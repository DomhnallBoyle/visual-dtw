import argparse
import ast
import multiprocessing
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main.research.test_update_list_5 import LIOPA_DATA_PATH, get_user_sessions, \
    get_test_data, get_default_sessions, create_mix_4, test_data_vs_sessions, \
    create_mix_2

PAVA_USER_DATA_PATH = \
    '/media/alex/Storage/Domhnall/datasets/sravi_dataset/pava_users/{user_id}'


def process_fold(_id, _training_data, _testing_data, _added_sessions,
                 _default_sessions):
    print('\nFold:', _id)
    print('Training:', len(_training_data))
    print('Testing:', len(_testing_data))

    start_time = time.time()
    _mix = create_mix_4(_training_data, _added_sessions, _default_sessions,
                        is_dtw_concurrent=False)
    end_time = time.time()

    time_taken = (end_time - start_time) / 60
    print(f'Fold {_id} took {time_taken} mins')

    _accuracy = test_data_vs_sessions(_testing_data, _mix,
                                      is_dtw_concurrent=False)

    return _accuracy, _mix, time_taken


def cross_validation(training_data, added_sessions, default_sessions, k=5,
                     multiple_max_test=False):
    subset_size = int(len(training_data) / k)

    random.shuffle(training_data)

    accuracies, mixes = [], []

    print('\nCross Validation:')

    tasks = []
    for i in range(k):
        testing_this_round = training_data[i * subset_size:][:subset_size]
        training_this_round = \
            training_data[:i * subset_size] \
            + training_data[(i + 1) * subset_size:]

        assert len(testing_this_round) + len(training_this_round) \
            == len(training_data)

        tasks.append([
            i+1,
            training_this_round,
            testing_this_round,
            added_sessions.copy(),
            default_sessions.copy()
        ])

    # multi core processing
    num_processes = k
    print('Num Processes:', num_processes)
    average_time_taken = 0
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_fold, tasks)
        for accuracy, session_set, time_taken in results:
            accuracies.append(accuracy)
            mixes.append(session_set)
            average_time_taken += time_taken
        average_time_taken /= k

    if multiple_max_test:
        max_accuracy = max(accuracies)
        mixes_to_test = [mixes[i] for i, accuracy in enumerate(accuracies)
                         if max_accuracy == accuracy]

        if len(mixes_to_test) == 1:
            return mixes_to_test[0], average_time_taken

        accuracies = [test_data_vs_sessions(training_data, mix)
                      for mix in mixes_to_test]
        mixes = mixes_to_test

    assert len(mixes) == len(accuracies)

    return mixes[accuracies.index(max(accuracies))], average_time_taken


def backward_session_selection(training_data, added_sessions,
                               default_sessions):
    """Backward session (feature) selection
    TAKES TOO LONG
    """

    # start off with the combination
    session_selection = added_sessions + default_sessions
    best_accuracy, best_sessions = None, []

    def check_accuracy(accuracy, sessions):
        nonlocal best_accuracy, best_sessions

        if not best_accuracy:
            best_accuracy = accuracy
            best_sessions = sessions

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_sessions = sessions

            print('New Accuracy: ', best_accuracy)
            counts = {}
            print('Selected: ', len(best_sessions))
            for session in best_sessions:
                _type = session[0].split('_')[0]
                counts[_type] = counts.get(_type, 0) + 1
            print('Makeup: ', counts)

    def dfs(data, sessions, i=1, accuracy_before=None):
        if len(data) == 1:
            return

        if i > len(sessions):
            return

        if not accuracy_before:
            accuracy_before = test_data_vs_sessions(data, sessions)

        session_split = sessions[:i-1] + sessions[i:]
        accuracy_after = test_data_vs_sessions(data, session_split)

        check_accuracy(accuracy_before, sessions)
        check_accuracy(accuracy_after, session_split)

        if accuracy_after >= accuracy_before:
            dfs(data, session_split, 1, accuracy_after)

        dfs(data, sessions, i+1, accuracy_before)

    dfs(training_data, session_selection)

    return best_sessions


def forward_session_selection(training_data, sessions):
    """
    This is flawed slightly in that if 2 sessions together gave an accuracy
    of 95%, this will be hard to beat and therefore prevent other sessions
    from being added.

    Doesn't work that well
    """
    session_selection = []
    best_accuracy = None
    while True:
        if len(sessions) == 0:
            break

        accuracies = []
        for session in sessions:
            accuracy = test_data_vs_sessions(test_data=training_data,
                                             sessions=[session] + session_selection)
            accuracies.append(accuracy)

        max_accuracy_index = accuracies.index(max(accuracies))
        max_accuracy = accuracies[max_accuracy_index]
        if not session_selection:
            session_selection.append(sessions.pop(max_accuracy_index))
            best_accuracy = max_accuracy
        else:
            if max_accuracy > best_accuracy:
                session_selection.append(sessions.pop(max_accuracy_index))
                best_accuracy = max_accuracy
            else:
                break

    print('Selected: ', len(session_selection))
    counts = {}
    for session in session_selection:
        _type = session[0].split('_')[0]
        counts[_type] = counts.get(_type, 0) + 1
    print('Makeup: ', counts)

    return session_selection


def experiment_1():
    """Cross Validation of Create Mix 4 algorithm"""
    users = {
        '1': '11',
        '6': '9'
    }

    for sravi_user, pava_user in users.items():

        # get the data first
        sessions_to_add = get_user_sessions(
            LIOPA_DATA_PATH.format(user_id=sravi_user))
        train_test_data = get_test_data(user_id=pava_user)
        default_sessions = get_default_sessions()

        # repeat adding sessions a number of times
        num_repeats = 5
        for repeat in range(1, num_repeats + 1):

            # do train, test split
            random.shuffle(train_test_data)
            train_split = int(len(train_test_data) * 0.6)
            training_data = train_test_data[:train_split]
            test_data = train_test_data[train_split:]
            print('Training Data: ', len(training_data))
            print('Test Data: ', len(test_data))
            assert len(training_data) + len(test_data) == len(train_test_data)

            # get default accuracy
            default_accuracy = test_data_vs_sessions(test_data,
                                                     default_sessions)

            # start adding sessions
            added_sessions = []
            results = []
            random.shuffle(sessions_to_add)
            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]
                result = [session_id_to_add]

                added_sessions.append(session_to_add)

                # cross validation with forward session selection
                mix_1 = cross_validation(training_data, added_sessions,
                                         default_sessions,
                                         multiple_max_test=True)
                mix_1_accuracy = test_data_vs_sessions(test_data, mix_1)
                print(mix_1_accuracy)

                result.append(mix_1_accuracy)
                results.append(result)

            with open('update_default_list_8.csv', 'a') as f:
                line = f'{sravi_user},{repeat},' \
                       f'{len(training_data)},{len(test_data)},' \
                       f'{default_accuracy},{results}\n'
                f.write(line)


def analysis_1(file_path):
    columns = ['User ID', 'Repeat', 'Num Training Data', 'Num Test Data',
               'Default Accuracy', 'Added Accuracies']
    data = []
    line_regex = r'(\d+),(\d+),(\d+),(\d+),(\d+.\d+),(\[.+\])'

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, repeat, num_training_data, num_test_data, \
                default_accuracy, added_accuracies = \
                re.match(line_regex, line).groups()
            added_accuracies = ast.literal_eval(added_accuracies)
            data.append([int(user_id), int(repeat), int(num_training_data),
                         int(num_test_data), float(default_accuracy),
                         added_accuracies])

    df = pd.DataFrame(data=data, columns=columns)

    for user_id in df['User ID'].unique():
        sub_df = df[df['User ID'] == user_id]

        max_rows, max_cols = 2, 3
        fig, axs = plt.subplots(max_rows, max_cols)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'User: {user_id}')
        rows, columns = 0, 0

        default_accuracies = []
        av_mix_accuracies = [[] for i in range(10)]

        for index, row in sub_df.iterrows():
            attempt = row['Repeat']
            default_accuracy = row['Default Accuracy']
            default_accuracies.append(default_accuracy)
            added_accuracies = row['Added Accuracies']
            mix_accuracies = [accuracy[1] for accuracy in added_accuracies]

            for i, accuracy in enumerate(mix_accuracies):
                av_mix_accuracies[i].append(accuracy)

            x = [i for i in range(1, len(added_accuracies) + 1)]
            axs[rows, columns].axhline(default_accuracy, color='black',
                                       label='Default')
            axs[rows, columns].plot(x, mix_accuracies,
                                    label='CV Forward Selection')
            axs[rows, columns].set_xlabel('Num Added Sessions')
            axs[rows, columns].set_ylabel('Accuracy')
            axs[rows, columns].set_title(f'User {user_id}, Attempt {attempt}')
            axs[rows, columns].set_ylim([60, 101])
            axs[rows, columns].set_xticks(np.arange(min(x), max(x)+1, 1.0))
            axs[rows, columns].legend()

            if columns == max_cols - 1:
                rows += 1
                columns = 0
            else:
                columns += 1

        # show average plots
        av_mix_accuracies = [sum(l) / len(l) for l in av_mix_accuracies]
        default_accuracies = sum(default_accuracies) / len(default_accuracies)

        x = [i for i in range(1, len(added_accuracies) + 1)]
        axs[rows, columns].axhline(default_accuracies, color='black',
                                   label='Default')
        axs[rows, columns].plot(x, av_mix_accuracies,
                                label='CV Forward Selection')
        axs[rows, columns].set_xlabel('Num Added Sessions')
        axs[rows, columns].set_ylabel('Accuracy')
        axs[rows, columns].set_title(f'User {user_id}, Average')
        axs[rows, columns].set_ylim([60, 101])
        axs[rows, columns].set_xticks(np.arange(min(x), max(x) + 1, 1.0))
        axs[rows, columns].legend()

        plt.show()

        # box plots over added sessions
        x = []
        for i in range(1):
            for j in range(10):
                x_i = []
                for index, row in sub_df.iterrows():
                    added_accuracies = row['Added Accuracies']
                    x_i.append(added_accuracies[j][i + 1])
                x.append(x_i)

        plt.boxplot(x)
        plt.xlabel('Num Added Sessions')
        plt.ylabel('Accuracy')
        plt.ylim([75, 101])
        plt.title(f'User {user_id}: Box Plot')
        plt.show()


def experiment_2():
    """Using another implementation of the forward selection
    """
    users = {
        '1': '11',
        '6': '9'
    }
    num_repeats = 5

    for sravi_user, pava_user in users.items():

        # get the data first
        sessions_to_add = get_user_sessions(
            LIOPA_DATA_PATH.format(user_id=sravi_user))
        train_test_data = get_test_data(user_id=pava_user)
        default_sessions = get_default_sessions()

        for repeat in range(1, num_repeats + 1):

            # do train, test split
            random.shuffle(train_test_data)
            train_split = int(len(train_test_data) * 0.6)
            training_data = train_test_data[:train_split]
            test_data = train_test_data[train_split:]
            print('Training Data: ', len(training_data))
            print('Test Data: ', len(test_data))
            assert len(training_data) + len(test_data) == len(train_test_data)

            # get default accuracy
            default_accuracy = test_data_vs_sessions(test_data,
                                                     default_sessions)

            # find best sessions from defaults only
            selected_best_default = forward_session_selection(
                training_data, default_sessions.copy())
            default_accuracy_2 = test_data_vs_sessions(test_data,
                                                       selected_best_default)

            # start adding sessions
            added_sessions = []
            results = []
            random.shuffle(sessions_to_add)
            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]
                result = [session_id_to_add]

                added_sessions.append(session_to_add)

                # forward session selection
                mix_1 = forward_session_selection(
                    training_data, added_sessions + default_sessions)
                mix_1_accuracy = test_data_vs_sessions(test_data, mix_1)

                # combination
                mix_2 = create_mix_2(added_sessions, default_sessions)
                mix_2_accuracy = test_data_vs_sessions(test_data, mix_2)

                result.extend([mix_1_accuracy, mix_2_accuracy])
                print(result)
                results.append(result)

            with open('update_default_list_9.csv', 'a') as f:
                line = f'{pava_user},{repeat},' \
                       f'{len(training_data)},{len(test_data)},' \
                       f'{default_accuracy},{default_accuracy_2},{results}\n'
                f.write(line)


def get_pava_user_sessions_and_test_data(videos_path, num_sessions=None):
    import os

    import pandas as pd
    from main import configuration
    from main.research.test_update_list_2 import create_template
    from main.utils.io import read_json_file

    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    groundtruth_path = os.path.join(videos_path, 'validated_groundtruth.txt')
    df = pd.read_csv(groundtruth_path, names=['Video', 'Phrase'])

    # first construct phrase templates
    phrase_templates = {}
    for index, row in df.iterrows():
        video_path = os.path.join(videos_path, row['Video'])

        template = create_template(video_path)
        if not template:
            continue

        templates = phrase_templates.get(row['Phrase'], [])
        templates.append(template)
        phrase_templates[row['Phrase']] = templates

    # construct our user sessions
    user_sessions = []

    if not num_sessions:
        num_sessions = min([len(templates)
                            for templates in phrase_templates.values()])

    while len(user_sessions) != num_sessions:
        session = []
        for phrase, templates in phrase_templates.items():
            random.shuffle(templates)
            session.append((phrase, templates.pop(0)))

        assert len(session) == len(pava_phrases)

        user_sessions.append((f'added_{len(user_sessions) + 1}', session))

    # construct our test data
    test_data = []
    for phrase, templates in phrase_templates.items():
        for template in templates:
            test_data.append((phrase, template))

    return user_sessions, test_data


def experiment_3():
    """Experimenting with PAVA groundtruth tool users"""

    user_id = '530bf0dd-4fb3-4ada-bd7f-3ee50d78ed98'

    default_sessions = get_default_sessions()

    # get user sessions and train test data
    sessions_to_add, train_test_data = \
        get_pava_user_sessions_and_test_data(
            PAVA_USER_DATA_PATH.format(user_id=user_id),
            num_sessions=2
        )

    print('Num sessions: ', len(sessions_to_add))
    print('Num train/test data: ', len(train_test_data))

    num_repeats = 5
    for repeat in range(1, num_repeats + 1):

        # do train, test split
        random.shuffle(train_test_data)
        train_split = int(len(train_test_data) * 0.6)
        training_data = train_test_data[:train_split]
        test_data = train_test_data[train_split:]
        print('Training Data: ', len(training_data))
        print('Test Data: ', len(test_data))
        assert len(training_data) + len(test_data) == len(train_test_data)

        # get default accuracy
        default_accuracy = test_data_vs_sessions(test_data,
                                                 default_sessions)

        # start adding sessions
        added_sessions = []
        results = []
        random.shuffle(sessions_to_add)
        for session_to_add in sessions_to_add:
            session_id_to_add = session_to_add[0]
            result = [session_id_to_add]

            added_sessions.append(session_to_add)

            # cross validation with forward session selection
            mix_1 = cross_validation(training_data, added_sessions,
                                     default_sessions, k=4)
            mix_1_accuracy = test_data_vs_sessions(test_data, mix_1)
            print(mix_1_accuracy)

            result.append(mix_1_accuracy)
            results.append(result)

        with open('update_default_list_10.csv', 'a') as f:
            line = f'{user_id},{repeat},' \
                   f'{len(training_data)},{len(test_data)},' \
                   f'{default_accuracy},{results}\n'
            f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('experiment')

    parser_2 = sub_parser.add_parser('analysis')
    parser_2.add_argument('file_path')

    args = parser.parse_args()
    run_type = args.run_type

    if run_type == 'experiment':
        experiment_1()
        # experiment_2()
        # experiment_3()
    else:
        analysis_1(args.file_path)
