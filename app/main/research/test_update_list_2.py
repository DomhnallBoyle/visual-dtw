import argparse
import ast
import math
import os
import queue
import random
import re
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config
from main.research.test_full_update import save_default_list_to_file, \
    save_default_as_sessions, get_session_ids, create_template
from main.research.test_update_list import find_top_sessions
from main.services.transcribe import transcribe_signal
from main.utils.io import read_json_file, read_pickle_file, write_pickle_file
from matplotlib.ticker import MaxNLocator

DATASET_PATH = '/home/domhnall/Documents/sravi_dataset/liopa'

ALL_USER_SESSIONS_PATH = 'all_user_sessions.pkl'
ALL_DEFAULT_SESSIONS_PATH = 'all_default_sessions.pkl'
USER_DEFAULT_SESSIONS_PATH = 'user_default_sessions.pkl'

NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'


def tests(ref_signals, test_templates, dtw_params):
    num_worker_threads = 3
    num_threads = num_worker_threads + 1

    # divide test videos among threads
    count_ar = np.linspace(0, len(test_templates), num_threads + 1,
                           dtype=int)
    test_videos_list = []
    temp_list = []
    i = 1
    for video in test_templates:
        temp_list.append(video)
        if i in count_ar:
            test_videos_list.append(temp_list)
            temp_list = []
        i += 1

    results_queue = queue.Queue()

    def make_predictions(_test_templates, _queue):
        for actual_label, test_template in _test_templates:
            try:
                predictions = transcribe_signal(ref_signals,
                                                test_template.blob,
                                                None, **dtw_params)
                top_prediction_label = predictions[0]['label']
                _queue.put((actual_label, top_prediction_label))
            except Exception as e:
                continue

    threads = []
    for i in range(num_worker_threads):
        thread = Thread(target=make_predictions,
                        args=(test_videos_list[i], results_queue))
        thread.start()
        threads.append(thread)

    make_predictions(test_videos_list[-1], results_queue)

    # wait for threads to finish
    for i in range(num_worker_threads):
        threads[i].join()

    # continue on main thread
    accuracy = 0
    num_tests = results_queue.qsize()
    while True:
        try:
            actual_label, top_prediction_label = results_queue.get_nowait()
            if actual_label == top_prediction_label:
                accuracy += 1
        except queue.Empty:
            break

    accuracy /= num_tests

    return accuracy * 100


def get_accuracy(sessions, test_videos):
    ref_signals = [(label, template.blob)
                   for session_key, session in sessions
                   for label, template in session]
    dtw_params = Config().__dict__

    return tests(ref_signals, test_videos, dtw_params)


def loo_cv(sessions):
    accuracies = []

    for i in range(len(sessions)):
        test = sessions[i][1]
        reference = sessions[:i] + sessions[i+1:]

        accuracy = get_accuracy(reference, test)
        accuracies.append(accuracy)

    print('LOOCV: ', accuracies, flush=True)

    return accuracies


def add_user_session(session):
    if os.path.exists(ALL_USER_SESSIONS_PATH):
        previous_sessions = read_pickle_file(ALL_USER_SESSIONS_PATH)
    else:
        previous_sessions = []

    new_sessions = previous_sessions + [session]
    write_pickle_file(new_sessions, ALL_USER_SESSIONS_PATH)

    return new_sessions


def train_test_session_split(videos_path, test_session):
    """Split videos into training and test"""
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    # split unseen user templates into sessions
    sessions_to_add = {}
    test_videos = []
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
        if int(session_id) == test_session:
            test_videos.append((phrase, template))
        else:
            session_videos = sessions_to_add.get(int(session_id), [])
            session_videos.append((phrase, template))
            sessions_to_add[int(session_id)] = session_videos

    return sessions_to_add, test_videos


def create_user_default_mix(remove_multiply=1, session_accuracies=None):
    """Remove worst performing default session for every user added session"""
    all_user_sessions = read_pickle_file(ALL_USER_SESSIONS_PATH)
    all_default_sessions = read_pickle_file(ALL_DEFAULT_SESSIONS_PATH)

    num_to_remove = len(all_user_sessions) * remove_multiply
    if num_to_remove < len(all_default_sessions):
        num_should_remove = num_to_remove

        combination = all_user_sessions + all_default_sessions
        num_total = len(combination)

        # check if we've already calculate previous accuracies
        if not session_accuracies:
            accuracies = loo_cv(sessions=combination)
            session_accuracies = accuracies.copy()
        else:
            accuracies = session_accuracies.copy()

        print('Num to remove: ', num_to_remove)

        final_combination = []
        while True:
            if num_to_remove == 0:
                break

            print('Sessions: ', [c[0] for c in combination], flush=True)
            print('Accuracies: ', accuracies, flush=True)

            min_accuracy_index = accuracies.index(min(accuracies))
            min_accuracy_session = combination[min_accuracy_index]

            if 'default' in min_accuracy_session[0]:
                del combination[min_accuracy_index]
                del accuracies[min_accuracy_index]
                num_to_remove -= 1

                # record which defaults were removed
                with open('removed_default_sessions.csv', 'a') as f:
                    f.write(min_accuracy_session[0] + '\n')
            else:
                # added removed as the minimum accuracy
                final_combination.append(min_accuracy_session)
                del combination[min_accuracy_index]
                del accuracies[min_accuracy_index]

        # add the remaining combos
        final_combination.extend(combination)
        assert len(final_combination) == num_total - num_should_remove
    else:
        final_combination = all_user_sessions

    write_pickle_file(final_combination, USER_DEFAULT_SESSIONS_PATH)

    return final_combination, session_accuracies


def create_user_default_mix_2(max_num_sessions):
    """Instead of doing LOOCV on combination of all added users sessions
    and default sessions, just compare accuracies of defaults vs added
    user sessions (ref) and remove worst defaults

    In this case, each default session should be used as reference vs
    all user sessions. If the accuracy of the user sessions (test) vs
    default sessions (ref) is high, then that means it is good enough to
    be used in the user-default mix.

    REMEMBER: The point of this is to find the best default sessions to be
    combined with ALL the user sessions so far. We need to find the default
    sessions that make up the MAX allowed sessions for the user
    """

    all_user_sessions = read_pickle_file(ALL_USER_SESSIONS_PATH)
    all_default_sessions = read_pickle_file(ALL_DEFAULT_SESSIONS_PATH)

    accuracies = []
    dtw_params = Config().__dict__
    test_templates = [(label, template)
                      for session_label, user_session in all_user_sessions
                      for label, template in user_session]

    # TODO: History of accuracies of default sessions. Average them and pick
    #  best

    # get accuracies, default sessions = ref, user sessions = test
    for session_label, default_session in all_default_sessions:
        ref_signals = [(label, template.blob)
                       for label, template in default_session]
        accuracy = tests(ref_signals, test_templates, dtw_params)
        accuracies.append(accuracy)

    # find best default sessions that make up MAX sessions
    number_to_add = max_num_sessions - len(all_user_sessions)
    num_added = 0
    final_combination = all_user_sessions.copy()

    while num_added != number_to_add:
        max_accuracy_index = accuracies.index(max(accuracies))
        final_combination.append(all_default_sessions.pop(max_accuracy_index))
        del accuracies[max_accuracy_index]
        num_added += 1

    assert len(final_combination) == max_num_sessions

    return final_combination


def find_top_sessions_2(all_user_sessions, n):
    """Find best n user sessions

    Instead of doing every combination of n user sessions like before,
    make every user session a ref vs all others as test, get the accuracies
    and use the n sessions that achieved the top n accuracies

    Choosing the best sessions like this shows they will probably be
    the best ref sessions to use going forward for that user
    """
    if len(all_user_sessions) == n:
        return all_user_sessions

    accuracies = []
    dtw_params = Config().__dict__

    # get accuracies, each user session is a ref, others are tests
    for i in range(len(all_user_sessions)):
        ref_session = all_user_sessions[i]
        test_sessions = all_user_sessions[:i] + all_user_sessions[i+1:]

        ref_signals = [(label, template.blob)
                       for label, template in ref_session[1]]
        test_templates = [(label, template)
                          for session_label, test_session in test_sessions
                          for label, template in test_session]

        accuracy = tests(ref_signals, test_templates, dtw_params)
        accuracies.append(accuracy)

    # find best
    num_added = 0
    best_sessions = []
    while num_added != n:
        max_accuracy_index = accuracies.index(max(accuracies))
        best_sessions.append(all_user_sessions[max_accuracy_index])
        del accuracies[max_accuracy_index]
        num_added += 1

    return best_sessions


def clean():
    for f in [ALL_USER_SESSIONS_PATH, USER_DEFAULT_SESSIONS_PATH]:
        if os.path.exists(f):
            os.remove(f)


def update_default_list(user, session_ids=None, remove_multiply=1):
    """Update sessions for the default list"""
    save_default_list_to_file()
    save_default_as_sessions(ALL_DEFAULT_SESSIONS_PATH)
    clean()

    videos_path = os.path.join(DATASET_PATH, user)
    if not session_ids:
        session_ids = get_session_ids(videos_path)
    print(session_ids)
    print(remove_multiply)

    for test_session_id in session_ids:
        sessions_to_add, test_videos = \
            train_test_session_split(videos_path, test_session_id)

        # shuffle the sessions to add
        keys = list(sessions_to_add.keys())
        random.shuffle(keys)
        sessions_to_add = [(f'added_{key}', sessions_to_add[key])
                           for key in keys]

        for max_num_sessions in range(1, len(sessions_to_add) + 1):

            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]

                all_user_sessions = add_user_session(session_to_add)
                num_added_sessions = len(all_user_sessions)
                print('Num added sessions: ', num_added_sessions)

                if num_added_sessions < max_num_sessions:
                    print('Creating default user mix')
                    user_default_sessions = \
                        create_user_default_mix(remove_multiply)
                    accuracy = get_accuracy(user_default_sessions, test_videos)
                else:
                    print(f'Finding {max_num_sessions} best user sessions')
                    top_user_sessions = find_top_sessions(all_user_sessions,
                                                          n=max_num_sessions)
                    accuracy = get_accuracy(top_user_sessions, test_videos)

                # some tests
                default_sessions = read_pickle_file(ALL_DEFAULT_SESSIONS_PATH)
                default_accuracy = get_accuracy(default_sessions, test_videos)
                user_accuracy = get_accuracy(all_user_sessions, test_videos)

                with open('update_default_list.csv', 'a') as f:
                    f.write(f'{user},{test_session_id},{max_num_sessions},'
                            f'{session_id_to_add},{num_added_sessions},'
                            f'{accuracy},{default_accuracy},{user_accuracy}\n')
                print()

            clean()


def update_default_list_2(user, max_sessions_range=None):
    save_default_list_to_file()
    save_default_as_sessions(ALL_DEFAULT_SESSIONS_PATH)
    clean()

    videos_path = os.path.join(DATASET_PATH, user)
    session_ids = get_session_ids(videos_path)

    if not max_sessions_range:
        max_sessions_range = list(range(1, len(session_ids)))

    for max_num_sessions in max_sessions_range:
        for test_session_id in session_ids:
            sessions_to_add, test_videos = \
                train_test_session_split(videos_path, test_session_id)

            # get default accuracy
            default_sessions = read_pickle_file(ALL_DEFAULT_SESSIONS_PATH)
            default_accuracy = get_accuracy(default_sessions, test_videos)

            # shuffle the sessions to add
            keys = list(sessions_to_add.keys())
            random.shuffle(keys)

            sessions_to_add = [(f'added_{key}', sessions_to_add[key])
                               for key in keys]

            add_results = []

            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]
                add_result = [session_id_to_add]

                all_user_sessions = add_user_session(session_to_add)
                num_added_sessions = len(all_user_sessions)

                if num_added_sessions < max_num_sessions:
                    new_sessions_1, accuracies = create_user_default_mix()
                    new_sessions_2, accuracies = \
                        create_user_default_mix(remove_multiply=2,
                                                session_accuracies=accuracies)

                    chosen_accuracy_1 = get_accuracy(new_sessions_1,
                                                     test_videos)
                    chosen_accuracy_2 = get_accuracy(new_sessions_2,
                                                     test_videos)

                    add_result.extend([chosen_accuracy_1, chosen_accuracy_2])
                else:
                    new_sessions = find_top_sessions(all_user_sessions,
                                                     n=max_num_sessions)
                    chosen_accuracy = get_accuracy(new_sessions, test_videos)
                    add_result.append(chosen_accuracy)

                user_accuracy = get_accuracy(all_user_sessions, test_videos)
                add_result.append(user_accuracy)

                add_results.append(add_result)

            with open('update_default_list_2.csv', 'a') as f:
                f.write(f'{user},{max_num_sessions},{test_session_id},'
                        f'{default_accuracy},{add_results}\n')

            clean()


def update_default_list_3(user, max_sessions_range=None):
    save_default_list_to_file()
    save_default_as_sessions(ALL_DEFAULT_SESSIONS_PATH)
    clean()

    videos_path = os.path.join(DATASET_PATH, user)
    session_ids = get_session_ids(videos_path)

    if not max_sessions_range:
        max_sessions_range = list(range(1, len(session_ids) + 1))

    for max_num_sessions in max_sessions_range:
        for test_session_id in session_ids:
            sessions_to_add, test_videos = \
                train_test_session_split(videos_path, test_session_id)

            # get default accuracy
            default_sessions = read_pickle_file(ALL_DEFAULT_SESSIONS_PATH)
            default_accuracy = get_accuracy(default_sessions, test_videos)

            # shuffle the sessions to add
            keys = list(sessions_to_add.keys())
            random.shuffle(keys)

            sessions_to_add = [(f'added_{key}', sessions_to_add[key])
                               for key in keys]

            add_results = []
            for session_to_add in sessions_to_add:
                session_id = session_to_add[0]
                add_result = [session_id]

                all_user_sessions = add_user_session(session_to_add)
                num_added_sessions = len(all_user_sessions)

                if num_added_sessions < max_num_sessions:
                    chosen_sessions = \
                        create_user_default_mix_2(max_num_sessions)
                else:
                    chosen_sessions = find_top_sessions_2(all_user_sessions,
                                                          n=max_num_sessions)

                chosen_accuracy = get_accuracy(chosen_sessions, test_videos)
                user_accuracy = get_accuracy(all_user_sessions, test_videos)

                add_result.extend([chosen_accuracy, user_accuracy])
                add_results.append(add_result)

            with open('update_default_list_3.csv', 'a') as f:
                f.write(f'{user},{max_num_sessions},{test_session_id},'
                        f'{default_accuracy},{add_results}\n')

            clean()


def update_new_list(user, session_ids=None):
    """Update sessions for a new list"""
    clean()

    videos_path = os.path.join(DATASET_PATH, user)
    if not session_ids:
        session_ids = get_session_ids(videos_path)
    print(session_ids)

    for test_session_id in session_ids:
        sessions_to_add, test_videos = \
            train_test_session_split(videos_path, test_session_id)

        # shuffle the sessions to add
        keys = list(sessions_to_add.keys())
        random.shuffle(keys)
        sessions_to_add = [(f'added_{key}', sessions_to_add[key])
                           for key in keys]

        for max_num_sessions in range(1, len(sessions_to_add) + 1):

            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]

                all_user_sessions = add_user_session(session_to_add)
                num_added_sessions = len(all_user_sessions)
                print('Num added sessions: ', num_added_sessions)

                if num_added_sessions < max_num_sessions:
                    print('Using all user sessions')
                    sessions = all_user_sessions
                else:
                    print(f'Finding {max_num_sessions} best user sessions')
                    sessions = find_top_sessions(all_user_sessions,
                                                 n=max_num_sessions)

                accuracy_1 = get_accuracy(sessions, test_videos)
                accuracy_2 = get_accuracy(all_user_sessions, test_videos)

                with open('update_new_list.csv', 'a') as f:
                    f.write(f'{user},{test_session_id},{max_num_sessions},'
                            f'{session_id_to_add},{num_added_sessions},'
                            f'{accuracy_1},{accuracy_2}\n')

            clean()


def experiment(args):
    users = args.users
    session_ids = args.session_ids
    remove_multiply = args.remove_multiply

    if len(users) == 1 and session_ids:
        # update_new_list(users[0], session_ids)
        # update_default_list(users[0], session_ids, remove_multiply)
        # update_default_list_2(users[0], session_ids)
        update_default_list_3(users[0], session_ids)
    else:
        for user in users:
            # update_new_list(user)
            # update_default_list(user)
            # update_default_list(user, remove_multiply=remove_multiply)
            # update_default_list_2(user)
            update_default_list_3(user)


def analyse_update_default_list(file_path):
    columns = ['User ID', 'Test Session ID', 'Max Num Sessions',
               'Added Session', 'Num Added Sessions',
               'Chosen Session Accuracy', 'Default Accuracy', 'User Accuracy']
    data = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, test_session_id, max_num_sessions, added_session_id, \
                num_added_sessions, accuracy, default_accuracy, user_accuracy \
                = line.split(',')
            data.append([int(user_id), int(test_session_id),
                         int(max_num_sessions), added_session_id,
                         int(num_added_sessions), float(accuracy),
                         float(default_accuracy), float(user_accuracy)])

    df = pd.DataFrame(columns=columns, data=data)

    users = df['User ID'].unique()

    pc = lambda new, original: ((new - original) / original) * 100

    # shows accuracy as sessions are added for chosen, default and user
    # sessions per max allowed session for every user
    for user in users:
        sub_df = df[df['User ID'] == user]
        unique_max_allowed_sessions = sub_df['Max Num Sessions'].unique()
        num_test_sessions = sub_df['Test Session ID'].max()

        plots_per_row = 3
        num_rows = math.ceil(len(unique_max_allowed_sessions) / plots_per_row)
        fig, ax = plt.subplots(num_rows, plots_per_row)
        fig.tight_layout()

        xs = []
        ys = []

        for num_max_allowed_sessions in unique_max_allowed_sessions:
            sub_sub_df = sub_df[sub_df['Max Num Sessions'] ==
                                num_max_allowed_sessions]

            d_num_added_sessions = {}
            for index, row in sub_sub_df.iterrows():
                num_added_sessions = row['Num Added Sessions']
                chosen_sessions_accuracy = row['Chosen Session Accuracy']
                default_accuracy = row['Default Accuracy']
                user_accuracy = row['User Accuracy']

                d = d_num_added_sessions.get(num_added_sessions, {
                    'chosen': 0, 'default': 0, 'user': 0, 'num': 0
                })
                d['chosen'] += chosen_sessions_accuracy
                d['default'] += default_accuracy
                d['user'] += user_accuracy
                d['num'] += 1
                d_num_added_sessions[num_added_sessions] = d

            for num_added_sessions, d in d_num_added_sessions.items():
                for k in ['chosen', 'default', 'user']:
                    d[k] /= d['num']

            x = list(d_num_added_sessions.keys())
            y1 = [d['chosen'] for d in d_num_added_sessions.values()]
            y2 = [d['default'] for d in d_num_added_sessions.values()]
            y3 = [d['user'] for d in d_num_added_sessions.values()]

            xs.append(x)
            ys.append([y1, y2, y3])

        k = 0
        for i in range(num_rows):
            for j in range(plots_per_row):
                if k == len(xs):
                    break
                x = xs[k]
                y = ys[k]
                ax[i, j].plot(x, y[0], label='Chosen Sessions')
                ax[i, j].plot(x, y[1], label='Default Sessions')
                ax[i, j].plot(x, y[2], label='User Sessions')
                ax[i, j].set_xlabel('Num Added Sessions')
                ax[i, j].set_ylabel('% Accuracy')
                ax[i, j].set_title(f'User {user}, '
                                   f'Max Sessions: {unique_max_allowed_sessions[k]}')
                ax[i, j].set_ylim([85, 101])
                ax[i, j].legend()
                ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                k += 1
        plt.show()

        # show percentage increase/decrease per num added sessions from the
        # default accuracy
        fig, ax = plt.subplots(num_rows, plots_per_row)
        fig.suptitle('% Change in Accuracy from the Default Accuracy')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        k = 0
        width = 0.2
        for i in range(num_rows):
            for j in range(plots_per_row):
                if k == len(xs):
                    break
                x = xs[k]
                chosen_y = ys[k][0]
                default_y = ys[k][1]
                user_y = ys[k][2]

                chosen_bars = []
                user_bars = []
                for l in range(len(x)):
                    default_accuracy = default_y[l]
                    chosen_accuracy = chosen_y[l]
                    user_accuracy = user_y[l]

                    pc_chosen = pc(chosen_accuracy, default_accuracy)
                    pc_user = pc(user_accuracy, default_accuracy)

                    chosen_bars.append(pc_chosen)
                    user_bars.append(pc_user)

                ax[i, j].bar(np.arange(len(x)), chosen_bars, width=width,
                             label='Chosen Sessions')
                ax[i, j].bar(np.arange(len(x)) + width, user_bars,
                             width=width, label='User Sessions')
                ax[i, j].set_xlabel('Num Added Sessions')
                ax[i, j].set_title(f'Max Sessions: '
                                   f'{unique_max_allowed_sessions[k]}')
                ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i, j].axhline(0, color='black')
                ax[i, j].legend()

                k += 1

        plt.show()

    # # repeat the above with all users combined
    # # would need to be max number of sessions 1-4 for all users
    # minimum = None
    # for user in users:
    #     sub_df = df[df['User ID'] == user]
    #     max_num_sessions = sub_df['Max Num Sessions'].max()
    #
    #     if not minimum or max_num_sessions < minimum:
    #         minimum = max_num_sessions
    #
    # max_num_sessions_range = minimum
    #
    # plots_per_row = 2
    # num_rows = math.ceil(max_num_sessions_range / plots_per_row)
    # fig, ax = plt.subplots(num_rows, plots_per_row)
    # fig.tight_layout()
    #
    # xs = []
    # ys = []
    #
    # max_num_sessions_range = list(range(1, max_num_sessions_range + 1))
    # for max_num_sessions in max_num_sessions_range:
    #     sub_df = df[df['Max Num Sessions'] == max_num_sessions]
    #
    #     d_num_added_sessions = {}
    #     for index, row in sub_df.iterrows():
    #         num_added_sessions = row['Num Added Sessions']
    #         chosen_sessions_accuracy = row['Chosen Session Accuracy']
    #         default_accuracy = row['Default Accuracy']
    #         user_accuracy = row['User Accuracy']
    #
    #         d = d_num_added_sessions.get(num_added_sessions, {
    #             'chosen': 0, 'default': 0, 'user': 0, 'num': 0
    #         })
    #         d['chosen'] += chosen_sessions_accuracy
    #         d['default'] += default_accuracy
    #         d['user'] += user_accuracy
    #         d['num'] += 1
    #         d_num_added_sessions[num_added_sessions] = d
    #
    #     for num_added_sessions, d in d_num_added_sessions.items():
    #         d['chosen'] /= d['num']
    #         d['default'] /= d['num']
    #         d['user'] /= d['num']
    #         del d['num']
    #
    #     x = list(d_num_added_sessions.keys())
    #     y1 = [d['chosen'] for d in d_num_added_sessions.values()]
    #     y2 = [d['default'] for d in d_num_added_sessions.values()]
    #     y3 = [d['user'] for d in d_num_added_sessions.values()]
    #
    #     xs.append(x)
    #     ys.append([y1, y2, y3])
    #
    # k = 0
    # for i in range(num_rows):
    #     for j in range(plots_per_row):
    #         if k == len(xs):
    #             break
    #         x = xs[k]
    #         y = ys[k]
    #         ax[i, j].plot(x, y[0], label='Chosen Sessions')
    #         ax[i, j].plot(x, y[1], label='Default Sessions')
    #         ax[i, j].plot(x, y[2], label='User Sessions')
    #         ax[i, j].set_xlabel('Num Added Sessions')
    #         ax[i, j].set_ylabel('% Accuracy')
    #         ax[i, j].set_title(f'Max Sessions: {max_num_sessions_range[k]}')
    #         ax[i, j].legend()
    #         ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
    #         k += 1
    # plt.show()

    # analysis of the accuracies of where you chose between user + defaults vs
    # accuracies of just chosing from user sessions


def analyse_removed_default_sessions(file_path):
    with open(file_path, 'r') as f:
        results = f.readlines()

    removed_counts = {}
    for removed_type in results:
        removed_counts[removed_type] = removed_counts.get(removed_type, 0) + 1

    sizes = list(removed_counts.values())
    labels = list(removed_counts.keys())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Number of removed default sessions')
    plt.tight_layout()
    plt.show()


def analyse_update_new_list(file_path):
    columns = ['User ID', 'Test Session ID', 'Max Sessions', 'Added Session',
               'Num Added Sessions', 'Chosen Accuracy', 'User Accuracy']
    data = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, test_session_id, max_sessions, added_session, \
                num_added_sessions, chosen_accuracy, user_accuracy = \
                line.split(',')
            data.append([int(user_id),
                        int(test_session_id),
                        int(max_sessions),
                        added_session,
                        int(num_added_sessions),
                        float(chosen_accuracy),
                        float(user_accuracy)])

    df = pd.DataFrame(columns=columns, data=data)

    users = df['User ID'].unique()

    for user in users:
        sub_df = df[df['User ID'] == user]
        unique_max_sessions = sub_df['Max Sessions'].unique()

        plots_per_row = 3
        num_rows = math.ceil(len(unique_max_sessions) / plots_per_row)
        fig, ax = plt.subplots(num_rows, plots_per_row)
        fig.tight_layout()

        xs = []
        ys = []

        for max_session in unique_max_sessions:
            sub_sub_df = sub_df[sub_df['Max Sessions'] == max_session]

            d_num_added_sessions = {}
            for index, row in sub_sub_df.iterrows():
                num_added_sessions = row['Num Added Sessions']
                chosen_sessions_accuracy = row['Chosen Accuracy']
                user_accuracy = row['User Accuracy']

                d = d_num_added_sessions.get(num_added_sessions, {
                    'chosen': 0, 'user': 0, 'num': 0
                })
                d['chosen'] += chosen_sessions_accuracy
                d['user'] += user_accuracy
                d['num'] += 1
                d_num_added_sessions[num_added_sessions] = d

            for num_added_sessions, d in d_num_added_sessions.items():
                for k in ['chosen', 'user']:
                    d[k] /= d['num']

            x = list(d_num_added_sessions.keys())
            y1 = [d['chosen'] for d in d_num_added_sessions.values()]
            y2 = [d['user'] for d in d_num_added_sessions.values()]

            xs.append(x)
            ys.append([y1, y2])

        k = 0
        for i in range(num_rows):
            for j in range(plots_per_row):
                if k == len(xs):
                    break
                x = xs[k]
                y = ys[k]
                ax[i, j].plot(x, y[0], label='Chosen Sessions')
                ax[i, j].plot(x, y[1], label='User Sessions')
                ax[i, j].set_xlabel('Num Added Sessions')
                ax[i, j].set_ylabel('% Accuracy')
                ax[i, j].set_title(f'User {user}, '
                                   f'Max Sessions: {unique_max_sessions[k]}')
                ax[i, j].legend()
                ax[i, j].set_ylim([90, 101])
                ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))
                k += 1
        plt.show()


def analyse_update_default_list_2(file_path):
    columns = ['User ID', 'Max Sessions', 'Test Session', 'Default Accuracy',
               'Added Accuracies']
    data = []
    line_regex = r'(\d+),(\d+),(\d+),(\d+.\d+),(\[.+\])'

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, max_sessions, test_session, default_accuracy, \
                added_accuracies = re.match(line_regex, line).groups()
            added_accuracies = ast.literal_eval(added_accuracies)
            data.append([int(user_id), int(max_sessions), int(test_session),
                         float(default_accuracy), added_accuracies])

    df = pd.DataFrame(columns=columns, data=data)

    user_ids = df['User ID'].unique()
    for user_id in user_ids:
        sub_df = df[df['User ID'] == user_id]
        unique_max_sessions = sub_df['Max Sessions'].unique()

        plots_per_row = 3
        num_rows = math.ceil(len(unique_max_sessions) / plots_per_row)
        fig, ax = plt.subplots(num_rows, plots_per_row)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        rows, columns = 0, 0

        for max_session in unique_max_sessions:
            sub_sub_df = sub_df[sub_df['Max Sessions'] == max_session]

            average_default_accuracy = sub_sub_df['Default Accuracy'].mean()
            num_added_sessions = len(sub_sub_df.iloc[0]['Added Accuracies'])

            # list for every added session
            chosen_accuracies = [[] for i in range(num_added_sessions)]
            chosen_accuracies_2 = [[] for i in range(num_added_sessions)]
            user_accuracies = [[] for i in range(num_added_sessions)]

            for index, row in sub_sub_df.iterrows():
                added_accuracies = row['Added Accuracies']

                for i, added in enumerate(added_accuracies):
                    if len(added) == 3:
                        added_session, chosen_accuracy, user_accuracy = added
                    else:
                        added_session, chosen_accuracy, chosen_accuracy_2, \
                            user_accuracy = added
                        chosen_accuracies_2[i].append(chosen_accuracy_2)

                    chosen_accuracies[i].append(chosen_accuracy)
                    user_accuracies[i].append(user_accuracy)

            # print(chosen_accuracies)
            # print(chosen_accuracies_2)
            # print(user_accuracies)

            # get averages
            chosen_accuracies = [sum(l) / len(l) for l in chosen_accuracies]
            user_accuracies = [sum(l) / len(l) for l in user_accuracies]
            chosen_accuracies_2 = [sum(l) / len(l)
                                   for l in chosen_accuracies_2
                                   if l]

            x = [i + 1 for i in range(num_added_sessions)]

            # plot graph
            ax[rows, columns].plot(x, chosen_accuracies, label='Chosen')
            ax[rows, columns].plot(x, user_accuracies, label='User')
            ax[rows, columns].plot([i + 1 for i in range(len(chosen_accuracies_2))],
                                   chosen_accuracies_2, label='Chosen 2')
            ax[rows, columns].axhline(average_default_accuracy, color='black')
            ax[rows, columns].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[rows, columns].set_title(f'Max Sessions: {max_session}')
            ax[rows, columns].set_xlabel('Number of added user sessions')
            ax[rows, columns].legend()

            if columns == plots_per_row - 1:
                rows += 1
                columns = 0
            else:
                columns += 1

        fig.suptitle(f'User {user_id}')
        plt.show()


def analysis(args):
    file_path = args.file_path

    # analyse_update_default_list(file_path)
    # analyse_update_new_list(file_path)
    # analyse_removed_default_sessions(file_path)
    analyse_update_default_list_2(file_path)


def str_list(s):
    return [i for i in s.split(',')]


def int_list(s):
    return [int(i) for i in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser1 = sub_parsers.add_parser('experiment')
    parser1.add_argument('users', type=str_list)
    parser1.add_argument('--session_ids', type=int_list)
    parser1.add_argument('--remove_multiply', type=int)

    parser2 = sub_parsers.add_parser('analysis')
    parser2.add_argument('file_path', type=str)

    args = parser.parse_args()
    run_type = args.run_type

    if run_type == 'experiment':
        experiment(args)
    elif run_type == 'analysis':
        analysis(args)
    else:
        print(parser.print_help())
