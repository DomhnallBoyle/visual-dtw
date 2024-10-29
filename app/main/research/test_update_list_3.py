import argparse
import ast
import queue
import itertools
import os
import random
import re
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config, PAVAList
from main.research.test_full_update import get_session_ids
from main.research.test_update_list_2 import create_template
from main.services.transcribe import transcribe_signal
from main.utils.io import read_json_file
from matplotlib.ticker import MaxNLocator

VIDEOS_PATH = '/home/domhnall/Documents/sravi_dataset/liopa/pava/{}'
NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
ANALYSIS_REGEX = r'(\d+),(\d+),(.+),(\d+.\d+),(\[.+\])'
MAX_TESTS = 5

kvs = {}


def get_default_sessions():
    # calling this method multiple times will yield the same default sessions
    # already checked this
    from main.utils.db import db_session
    from sqlalchemy.orm import joinedload

    with db_session() as s:
        default_list = PAVAList.get(
            s, loading_options=(joinedload('phrases.templates')),
            filter=(PAVAList.id == configuration.DEFAULT_PAVA_LIST_ID),
            first=True)

    phrase_templates = {}
    for phrase in default_list.phrases:
        if phrase.content != 'None of the above':
            phrase_templates[phrase.content] = phrase.templates

    num_templates = sum([len(v) for k, v in phrase_templates.items()])
    num_sessions = num_templates // 20
    print('Num default sessions: ', num_sessions)

    default_sessions = []
    for i in range(num_sessions):
        default_session = []
        for phrase_content, templates in phrase_templates.items():
            default_session.append((phrase_content, templates[i]))

        assert len(default_session) == 20
        default_sessions.append((f'default_{i + 1}', default_session))

    return default_sessions


def get_user_sessions(videos_path):
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    # split unseen user templates into sessions
    user_sessions = {}
    for video in os.listdir(videos_path):
        if not video.endswith('.mp4'):
            continue

        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        video = os.path.join(videos_path, video)
        template = create_template(video)
        if not template:
            continue

        phrase = pava_phrases[phrase_id]
        session_videos = user_sessions.get(int(session_id), [])
        session_videos.append((phrase, template))
        user_sessions[int(session_id)] = session_videos

    user_sessions = [(f'added_{k}', v) for k, v in user_sessions.items()]

    return user_sessions


def to_labels(sessions):
    return frozenset([s[0] for s in sessions])


def kvs_accuracy_lookup(test_sessions, ref_sessions):
    test_session_labels = to_labels(test_sessions)
    ref_session_labels = to_labels(ref_sessions)

    accuracies = kvs.get(test_session_labels)
    if accuracies:
        return accuracies.get(ref_session_labels)

    return None


def get_accuracy(test_sessions, ref_sessions, kvs_lookup=True):
    if kvs_lookup:
        accuracy = kvs_accuracy_lookup(test_sessions, ref_sessions)
        if accuracy:
            return accuracy

    dtw_params = Config().__dict__

    test_signals = [(label, template.blob)
                    for session_label, test_session in test_sessions
                    for label, template in test_session]
    ref_signals = [(label, template.blob)
                   for session_label, ref_session in ref_sessions
                   for label, template in ref_session]

    # num_workers = 2
    # num_threads = num_workers + 1
    #
    # # divide test signals between threads
    # count_ar = np.linspace(0, len(test_signals), num_threads + 1, dtype=int)
    # test_signals_list = []
    # temp_list = []
    # i = 1
    # for entry in test_signals:
    #     temp_list.append(entry)
    #     if i in count_ar:
    #         test_signals_list.append(temp_list)
    #         temp_list = []
    #     i += 1
    #
    # q = queue.Queue()
    #
    # def transcribe_signals(_test_signals, _ref_signals, _queue):
    #     _num_correct = 0
    #     for actual_label, test_signal in _test_signals:
    #         try:
    #             predictions = transcribe_signal(_ref_signals, test_signal,
    #                                             None, **dtw_params)
    #         except Exception:
    #             continue
    #
    #         if actual_label == predictions[0]['label']:
    #             _num_correct += 1
    #     _queue.put(_num_correct)
    #
    # threads = []
    # for i in range(num_workers):
    #     thread = Thread(target=transcribe_signals,
    #                     args=(test_signals_list[i], ref_signals.copy(), q))
    #     thread.start()
    #     threads.append(thread)
    #
    # transcribe_signals(test_signals_list[-1], ref_signals.copy(), q)
    #
    # for i in range(num_workers):
    #     threads[i].join()  # main thread wait until workers finished
    #
    # total_num_correct = 0
    # while True:
    #     try:
    #         num_correct = q.get_nowait()
    #         total_num_correct += num_correct
    #     except queue.Empty:
    #         break
    #
    # accuracy = (total_num_correct / len(test_signals)) * 100

    num_correct = 0
    for actual_label, test_signal in test_signals:
        try:
            predictions = transcribe_signal(ref_signals, test_signal, None,
                                            **dtw_params)
        except Exception:
            continue

        if actual_label == predictions[0]['label']:
            num_correct += 1

    accuracy = (num_correct / len(test_signals)) * 100

    # kvs set
    if kvs_lookup:
        test_session_labels = to_labels(test_sessions)
        ref_session_labels = to_labels(ref_sessions)
        accuracies = kvs.get(test_session_labels, {})
        accuracies[ref_session_labels] = accuracy
        kvs[test_session_labels] = accuracies

    return accuracy


def create_mix(added_sessions, default_sessions, num_allowed_sessions):
    """In this case, each default session should be used as reference vs
    all user sessions. If the accuracy of the user sessions (test) vs
    default sessions (ref) is high, then that means it is good enough to
    be used in the user-default mix.

    REMEMBER: The point of this is to find the best default sessions to be
    combined with ALL the user sessions so far. We need to find the default
    sessions that make up the MAX allowed sessions for the user"""

    # get accuracies, default sessions = ref, user sessions = test
    accuracies = []
    for default_session in default_sessions:
        accuracy = get_accuracy(test_sessions=added_sessions,
                                ref_sessions=[default_session])
        accuracies.append(accuracy)

    # find best default sessions that make up MAX sessions
    num_default_to_add = num_allowed_sessions - len(added_sessions)
    num_default_added = 0
    final_combination = added_sessions.copy()

    while num_default_added != num_default_to_add:
        max_accuracy_index = accuracies.index(max(accuracies))
        final_combination.append(default_sessions.pop(max_accuracy_index))
        del accuracies[max_accuracy_index]
        num_default_added += 1

    assert len(final_combination) == num_allowed_sessions

    return final_combination


def create_mix_2(added_sessions, default_sessions):
    """Compare every session (ref) vs all others (test)"""
    combination = added_sessions + default_sessions
    best_sessions = find_best(sessions=combination.copy(),
                              num_allowed_sessions=len(default_sessions))

    return best_sessions


def find_best(sessions, num_allowed_sessions):
    """Find best n user sessions

    Instead of doing every combination of n user sessions like before,
    make every user session a ref vs all others as test, get the accuracies
    and use the n sessions that achieved the top n accuracies

    Choosing the best sessions like this shows they will probably be
    the best ref sessions to use going forward for that user
    """
    accuracies = []

    # get accuracies, each user session is a ref, others are tests
    for i in range(len(sessions)):
        ref_session = sessions[i]
        test_sessions = sessions[:i] + sessions[i+1:]

        accuracy = get_accuracy(test_sessions=test_sessions,
                                ref_sessions=[ref_session])
        accuracies.append(accuracy)

    # find best
    num_added = 0
    best_sessions = []
    while num_added != num_allowed_sessions:
        max_accuracy_index = accuracies.index(max(accuracies))
        best_sessions.append(sessions.pop(max_accuracy_index))
        del accuracies[max_accuracy_index]
        num_added += 1

    assert len(best_sessions) == num_allowed_sessions

    return best_sessions


def find_best_2(added_sessions, default_sessions, num_allowed_sessions):
    """Rank all defaults (test) vs every single added session (ref)"""
    accuracies = []
    for ref_session in added_sessions:
        accuracy = get_accuracy(default_sessions, [ref_session])
        accuracies.append(accuracy)

    # find best
    num_added = 0
    best_sessions = []
    while num_added != num_allowed_sessions:
        max_accuracy_index = accuracies.index(max(accuracies))
        best_sessions.append(added_sessions.pop(max_accuracy_index))
        del accuracies[max_accuracy_index]
        num_added += 1

    assert len(best_sessions) == num_allowed_sessions

    return best_sessions


def find_best_3(added_sessions, default_sessions, num_allowed_sessions):
    """Every combination of 5 user sessions (ref) vs default (test)
    Cache results to make it faster"""

    best_accuracy = 0
    best_sessions = None
    for ref_sessions in itertools.combinations(added_sessions,
                                               num_allowed_sessions):
        accuracy = get_accuracy(default_sessions, ref_sessions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_sessions = ref_sessions

    return best_sessions


def experiment(user_id, start_test_num=1):
    global kvs
    kvs = {}

    # find all session ids for that user
    session_ids = get_session_ids(VIDEOS_PATH.format(user_id))
    max_allowed_sessions = len(session_ids) - 1 if len(session_ids) <= 5 else 5
    print('Session IDs: ', session_ids)
    print('Max Allowed Sessions: ', max_allowed_sessions)

    default_sessions = get_default_sessions()
    user_sessions = get_user_sessions(VIDEOS_PATH.format(user_id))

    for num_tests in range(start_test_num, MAX_TESTS + 1):
        for i in range(len(user_sessions)):
            test_session = user_sessions[i]
            sessions_to_add = user_sessions[:i] + user_sessions[i+1:]

            test_session_label = test_session[0]

            default_accuracy = get_accuracy([test_session], default_sessions)

            # shuffle sessions to add
            random.shuffle(sessions_to_add)

            added_sessions = []
            results = []
            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]
                result = [session_id_to_add]

                # add user session
                added_sessions.append(session_to_add)
                num_added_sessions = len(added_sessions)

                if num_added_sessions < max_allowed_sessions:
                    # adding default sessions to as many max
                    chosen_sessions = create_mix(
                        added_sessions=added_sessions.copy(),
                        default_sessions=default_sessions.copy(),
                        num_allowed_sessions=max_allowed_sessions
                    )
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)

                    # adding default sessions to as many defaults
                    chosen_sessions = create_mix(
                        added_sessions=added_sessions.copy(),
                        default_sessions=default_sessions.copy(),
                        num_allowed_sessions=len(default_sessions)
                    )
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)

                    # combining user and default - finding best 13
                    chosen_sessions = create_mix_2(
                        added_sessions=added_sessions.copy(),
                        default_sessions=default_sessions.copy()
                    )
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)
                elif num_added_sessions == max_allowed_sessions:
                    chosen_sessions = added_sessions.copy()
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)
                else:
                    chosen_sessions = find_best(
                        sessions=added_sessions.copy(),
                        num_allowed_sessions=max_allowed_sessions
                    )
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)

                    chosen_sessions = find_best_2(
                        added_sessions=added_sessions.copy(),
                        default_sessions=default_sessions.copy(),
                        num_allowed_sessions=max_allowed_sessions
                    )
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)

                    chosen_sessions = find_best_3(
                        added_sessions=added_sessions.copy(),
                        default_sessions=default_sessions.copy(),
                        num_allowed_sessions=max_allowed_sessions
                    )
                    accuracy = get_accuracy([test_session], chosen_sessions)
                    result.append(accuracy)

                user_accuracy = get_accuracy([test_session], added_sessions)
                result.append(user_accuracy)

                results.append(result)

            with open('update_default_list_4.csv', 'a') as f:
                line = f'{user_id},{num_tests},{test_session_label},' \
                       f'{default_accuracy},{results}\n'
                f.write(line)


def analysis(file_path, new):
    columns = ['User ID', 'Test Num', 'Test Session', 'Default Accuracy',
               'Results']
    data = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, test_num, test_session, default_accuracy, results = \
                re.match(ANALYSIS_REGEX, line).groups()
            data.append([
                int(user_id),
                int(test_num),
                test_session,
                float(default_accuracy),
                ast.literal_eval(results)
            ])

    df = pd.DataFrame(columns=columns, data=data)

    users = df['User ID'].unique()
    print(len(users))

    max_rows, max_cols = 3, 3
    fig, axs = plt.subplots(max_rows, max_cols)
    fig.tight_layout()
    rows, columns = 0, 0

    usernames = {
        1: 'Fabian (SRAVI)',
        2: 'Alex (SRAVI)',
        3: 'Adrian (SRAVI)',
        5: 'Liam (SRAVI)',
        6: 'Richard (SRAVI)',
        7: 'Conor (SRAVI)',
        9: 'Richard',
        11: 'Fabian',
        12: 'Domhnall',
        17: 'Michael (SRAVI)'
    }

    max_added_sessions = {
        1: 5,
        2: 5,
        3: 5,
        5: 5,
        6: 5,
        7: 5,
        9: 4,
        11: 5,
        12: 5,
        17: 5
    }

    for user in users:
        sub_df = df[df['User ID'] == user]

        average_default_accuracy = sub_df['Default Accuracy'].mean()

        added_session_count = len(sub_df.iloc[0]['Results'])

        chosen_accuracies_1 = [[] for i in range(added_session_count)]
        chosen_accuracies_2 = [[] for i in range(added_session_count)]
        chosen_accuracies_3 = [[] for i in range(added_session_count)]
        user_accuracies = [[] for i in range(added_session_count)]

        for index, row in sub_df.iterrows():
            results = row['Results']
            for i, result in enumerate(results):
                num_added_sessions = i + 1

                if num_added_sessions < max_added_sessions[user]:
                    if len(result) == 3:
                        session_label, chosen_accuracy_1, user_accuracy \
                            = result
                    elif len(result) == 4:
                        session_label, chosen_accuracy_1, chosen_accuracy_2, \
                            user_accuracy = result
                        chosen_accuracies_2[i].append(chosen_accuracy_2)
                    else:
                        session_label, chosen_accuracy_1, chosen_accuracy_2, \
                            chosen_accuracy_3, user_accuracy = result
                        chosen_accuracies_2[i].append(chosen_accuracy_2)
                        chosen_accuracies_3[i].append(chosen_accuracy_3)

                    chosen_accuracies_1[i].append(chosen_accuracy_1)
                    user_accuracies[i].append(user_accuracy)
                elif num_added_sessions == max_added_sessions[user]:
                    session_label, chosen_accuracy, user_accuracy = result
                    chosen_accuracies_1[i].append(chosen_accuracy)
                    chosen_accuracies_2[i].append(chosen_accuracy)
                    chosen_accuracies_3[i].append(chosen_accuracy)
                    user_accuracies[i].append(user_accuracy)
                else:
                    session_label, chosen_accuracy_1, chosen_accuracy_2, \
                        chosen_accuracy_3, user_accuracy = result
                    chosen_accuracies_1[i].append(chosen_accuracy_1)
                    chosen_accuracies_2[i].append(chosen_accuracy_2)
                    chosen_accuracies_3[i].append(chosen_accuracy_3)
                    user_accuracies[i].append(user_accuracy)

        lists_average = \
            lambda ll: [sum(l) / len(l) for l in ll if l]

        chosen_accuracies_1 = lists_average(chosen_accuracies_1)
        chosen_accuracies_2 = lists_average(chosen_accuracies_2)
        chosen_accuracies_3 = lists_average(chosen_accuracies_3)
        user_accuracies = lists_average(user_accuracies)

        axs[rows, columns].plot(
            [i + 1 for i in range(len(chosen_accuracies_1))],
            chosen_accuracies_1, label='Chosen A')
        axs[rows, columns].plot(
            [i + 1 for i in range(len(chosen_accuracies_2))],
            chosen_accuracies_2, label='Chosen B')
        axs[rows, columns].plot(
            [i + 1 for i in range(len(chosen_accuracies_3))],
            chosen_accuracies_3, label='Chosen C')
        axs[rows, columns].plot(
            [i + 1 for i in range(len(user_accuracies))],
            user_accuracies, label='User')

        axs[rows, columns].axhline(average_default_accuracy, color='black',
                                   label='Default')
        axs[rows, columns].set_xlabel('Number of added sessions')
        axs[rows, columns].set_ylabel('Accuracy')
        axs[rows, columns].set_title(usernames[user])
        axs[rows, columns].legend()
        axs[rows, columns].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[rows, columns].set_ylim([60, 100])

        if columns == max_cols - 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.show()


def lst(s):
    return s.split(',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('experiment')
    parser_1.add_argument('users', type=lst)
    parser_1.add_argument('--start_test_num', type=int)

    parser_2 = sub_parsers.add_parser('analysis')
    parser_2.add_argument('file_path')
    parser_2.add_argument('--new', type=bool, default=False)

    args = parser.parse_args()
    run_type = args.run_type

    if run_type == 'experiment':
        users = args.users
        start_test_num = args.start_test_num

        if len(users) == 1 and start_test_num:
            experiment(users[0], start_test_num)
        else:
            for user_id in users:
                experiment(user_id)
    else:
        analysis(args.file_path, args.new)
