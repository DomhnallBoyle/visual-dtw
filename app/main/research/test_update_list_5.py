import argparse
import ast
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config
from main.research.test_update_list_3 import \
    create_template, get_user_sessions, get_default_sessions, \
    NEW_RECORDINGS_REGEX, transcribe_signal, create_mix as create_mix_3
from main.utils.io import read_json_file

LIOPA_DATA_PATH = \
    '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/set_1/{user_id}'

kvs = {}


def get_test_data(user_id):
    test_data = []
    videos_path = LIOPA_DATA_PATH.format(user_id=user_id)
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    for video_filename in os.listdir(videos_path):
        if not video_filename.endswith('.mp4'):
            continue

        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video_filename).groups()

        template = create_template(os.path.join(videos_path, video_filename))
        if not template:
            continue

        phrase = pava_phrases[phrase_id]
        test_data.append((phrase, template))

    return test_data


def to_labels(templates):
    return frozenset([id(t) for t in templates])


def accuracy_lookup(test_templates, ref_templates):
    test_labels = to_labels(test_templates)
    ref_labels = to_labels(ref_templates)

    accuracies = kvs.get(test_labels)
    if accuracies:
        return accuracies.get(ref_labels)

    return None


def test_data_vs_sessions(test_data, sessions, is_dtw_concurrent=True):
    dtw_params = Config().__dict__
    dtw_params['is_dtw_concurrent'] = is_dtw_concurrent

    test_signals = [(label, template.blob)
                    for label, template in test_data]

    ref_signals = [(label, template.blob)
                   for session_label, ref_session in sessions
                   for label, template in ref_session]

    test_templates = [t[1] for t in test_data]
    ref_templates = [t[1] for session_label, ref_session in sessions
                     for t in ref_session]
    accuracy = accuracy_lookup(test_templates, ref_templates)
    if accuracy:
        return accuracy

    num_correct = 0
    for actual_label, test_signal in test_signals:
        try:
            predictions = transcribe_signal(ref_signals, test_signal, None,
                                            **dtw_params)
        except Exception as e:
            continue

        if actual_label == predictions[0]['label']:
            num_correct += 1

    accuracy = (num_correct / len(test_data)) * 100

    # kvs set
    test_labels = to_labels(test_templates)
    ref_labels = to_labels(ref_templates)
    accuracies = kvs.get(test_labels, {})
    accuracies[ref_labels] = accuracy
    kvs[test_labels] = accuracies

    return accuracy


def create_mix_1(training_data, added_sessions, default_sessions):
    """Are the added and default sessions good enough?

    Using the test data, get accuracy vs every other session
    Rank them and select top 13 sessions from the combo
    """
    session_combination = added_sessions + default_sessions

    # get accuracies using the test data vs every session
    accuracies = []
    for session in session_combination:
        accuracy = test_data_vs_sessions(training_data, [session])
        accuracies.append(accuracy)

    # sort and select top 13
    num_to_add = len(default_sessions)
    num_added = 0
    final_combination = []

    assert len(accuracies) == len(session_combination)

    while num_added != num_to_add:
        max_accuracy_index = accuracies.index(max(accuracies))
        final_combination.append(session_combination.pop(max_accuracy_index))
        del accuracies[max_accuracy_index]
        num_added += 1

    assert len(final_combination) == num_to_add

    return final_combination


def create_mix_2(added_sessions, default_sessions):
    """Just combine both added and default sessions"""
    return added_sessions + default_sessions


def create_mix_4(training_data, added_sessions, default_sessions,
                 initial_max=13, is_dtw_concurrent=True):
    """forward session selection"""
    session_selection = []
    session_combination = added_sessions + default_sessions

    while True:
        accuracies = []
        for session in session_combination:
            accuracy = \
                test_data_vs_sessions(test_data=training_data,
                                      sessions=[session] + session_selection,
                                      is_dtw_concurrent=is_dtw_concurrent)
            accuracies.append(accuracy)

        max_accuracy_index = accuracies.index(max(accuracies))
        best_session = session_combination.pop(max_accuracy_index)
        session_selection.append(best_session)

        if len(session_selection) == initial_max:
            best_accuracy = accuracies[max_accuracy_index]

            while True:
                if len(session_combination) == 0:
                    break

                accuracies = []
                for session in session_combination:
                    accuracy = \
                        test_data_vs_sessions(
                            test_data=training_data,
                            sessions=[session] + session_selection,
                            is_dtw_concurrent=is_dtw_concurrent
                        )
                    accuracies.append(accuracy)

                max_accuracy_index = accuracies.index(max(accuracies))
                max_accuracy = accuracies[max_accuracy_index]
                if max_accuracy >= best_accuracy:
                    best_session = session_combination.pop(max_accuracy_index)
                    session_selection.append(best_session)
                    best_accuracy = max_accuracy
                else:
                    break

            break

    # # TODO: Delete this after
    # counts = {}
    # print('Selected: ', len(session_selection))
    # print('Leftover: ', len(session_combination))
    # for session in session_selection:
    #     _type = session[0].split('_')[0]
    #     counts[_type] = counts.get(_type, 0) + 1
    # print('Makeup: ', counts)

    return session_selection


def experiment_1():
    users = {
        '1': '11',
        '6': '9'
    }

    num_repeats = 5

    for sravi_user, pava_user in users.items():

        sessions_to_add = get_user_sessions(LIOPA_DATA_PATH.format(user_id=sravi_user))
        test_data = get_test_data(user_id=pava_user)
        default_sessions = get_default_sessions()

        groups_of = int(len(test_data) / len(sessions_to_add))
        print('Adding test data in groups of: ', groups_of)

        for repeat in range(1, num_repeats + 1):
            test_data_copy = test_data.copy()

            added_test_data = []
            added_sessions = []

            random.shuffle(sessions_to_add)

            for session_to_add in sessions_to_add:
                # add in groups of randomly
                added_test_data.extend(
                    [test_data_copy.pop(random.randrange(len(test_data_copy)))
                     for _ in range(groups_of)]
                )

                default_accuracy = test_data_vs_sessions(added_test_data,
                                                         default_sessions)

                added_sessions.append(session_to_add)

                # use external training data to rank added and defaults
                mix_1 = create_mix_1(added_test_data, added_sessions, default_sessions)
                mix_1_accuracy = test_data_vs_sessions(added_test_data, mix_1)

                # just combine all sessions together
                mix_2 = create_mix_2(added_sessions, default_sessions)
                mix_2_accuracy = test_data_vs_sessions(added_test_data, mix_2)

                # use added sessions to rank defaults only
                mix_3 = create_mix_3(added_sessions, default_sessions.copy(), 13)
                mix_3_accuracy = test_data_vs_sessions(added_test_data, mix_3)

                # forward session (feature) selection
                mix_4 = create_mix_4(added_test_data, added_sessions, default_sessions)
                mix_4_accuracy = test_data_vs_sessions(added_test_data, mix_4)

                # print('Num test data: ', len(added_test_data))
                # print('Default: ', default_accuracy)
                # print('Mix 1: ', mix_1_accuracy)
                # print('Mix 2: ', mix_2_accuracy)
                # print('Mix 3: ', mix_3_accuracy)
                # print()

                with open('update_default_list_6.csv', 'a') as f:
                    line = f'{sravi_user},{repeat},{len(added_test_data)},' \
                           f'{default_accuracy},{mix_1_accuracy},' \
                           f'{mix_2_accuracy},{mix_3_accuracy},' \
                           f'{mix_4_accuracy}\n'
                    f.write(line)


def analyse_1(file_path):
    columns = ['User ID', 'Repeat', 'Num Test Data', 'Default Acc', 'Mix 1 Acc',
               'Mix 2 Acc', 'Mix 3 Acc', 'Mix 4 Acc']
    data = []

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, repeat, num_test_data, default_acc, mix_1_acc, mix_2_acc, \
                mix_3_acc, mix_4_acc = line.split(',')
            data.append([int(user_id), int(repeat), int(num_test_data),
                         float(default_acc),  float(mix_1_acc),
                         float(mix_2_acc), float(mix_3_acc), float(mix_4_acc)])

    df = pd.DataFrame(columns=columns, data=data)

    for user_id in df['User ID'].unique():

        max_rows, max_cols = 2, 3
        fig, axs = plt.subplots(max_rows, max_cols)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'User: {user_id}')
        rows, columns = 0, 0

        sub_df = df[df['User ID'] == user_id]
        unique_num_test_data = sub_df['Num Test Data'].unique()

        for repeat in range(1, 6):
            sub_sub_df = sub_df[sub_df['Repeat'] == repeat]

            x = list(sub_sub_df['Num Test Data'])
            y1 = list(sub_sub_df['Default Acc'])
            y2 = list(sub_sub_df['Mix 1 Acc'])
            y3 = list(sub_sub_df['Mix 2 Acc'])
            y4 = list(sub_sub_df['Mix 3 Acc'])
            y5 = list(sub_sub_df['Mix 4 Acc'])

            axs[rows, columns].plot(x, y1, label='Default')
            axs[rows, columns].plot(x, y2, label='External Rank All')
            axs[rows, columns].plot(x, y3, label='Combine')
            axs[rows, columns].plot(x, y4, label='Session Rank Defaults')
            axs[rows, columns].plot(x, y5, label='Forward Selection')
            axs[rows, columns].set_xlabel('Num Test Data')
            axs[rows, columns].set_ylabel('Accuracy')
            axs[rows, columns].set_title('Repeat: ' + str(repeat))
            axs[rows, columns].set_ylim([50, 101])
            axs[rows, columns].legend()

            if columns == max_cols - 1:
                rows += 1
                columns = 0
            else:
                columns += 1

        av_default_accuracies, av_mix_1_accuracies, av_mix_2_accuracies, \
            av_mix_3_accuracies, av_mix_4_accuracies = [], [], [], [], []
        for num_test_data in unique_num_test_data:
            x = sub_df[sub_df['Num Test Data'] == num_test_data]
            av_default_accuracies.append(x['Default Acc'].mean())
            av_mix_1_accuracies.append(x['Mix 1 Acc'].mean())
            av_mix_2_accuracies.append(x['Mix 2 Acc'].mean())
            av_mix_3_accuracies.append(x['Mix 3 Acc'].mean())
            av_mix_4_accuracies.append(x['Mix 4 Acc'].mean())

        x = unique_num_test_data
        axs[rows, columns].plot(x, av_default_accuracies, label='Default')
        axs[rows, columns].plot(x, av_mix_1_accuracies, label='External Rank All')
        axs[rows, columns].plot(x, av_mix_2_accuracies, label='Combine')
        axs[rows, columns].plot(x, av_mix_3_accuracies, label='Session Rank Defaults')
        axs[rows, columns].plot(x, av_mix_4_accuracies, label='Forward Selection')
        axs[rows, columns].set_xlabel('Num Test Data')
        axs[rows, columns].set_ylabel('Accuracy')
        axs[rows, columns].set_title('Average')
        axs[rows, columns].set_ylim([50, 101])
        axs[rows, columns].legend()

        plt.show()


def experiment_2():
    """Use pava dataset as training/test split
    Use the sravi dataset as the sessions to add
    Use external training data to select mixtures
    Use external test data to test against mixtures
    """

    # sravi: pava
    users = {
        # '1': '11',
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

            # start adding sesions
            added_sessions = []
            results = []
            random.shuffle(sessions_to_add)
            for session_to_add in sessions_to_add:
                session_id_to_add = session_to_add[0]
                result = [session_id_to_add]

                added_sessions.append(session_to_add)

                # use external training data to rank added and defaults
                mix_1 = create_mix_1(training_data, added_sessions, default_sessions)
                mix_1_accuracy = test_data_vs_sessions(test_data, mix_1)

                # just combine all sessions together
                mix_2 = create_mix_2(added_sessions, default_sessions)
                mix_2_accuracy = test_data_vs_sessions(test_data, mix_2)

                # use added sessions to rank defaults only
                mix_3 = create_mix_3(added_sessions, default_sessions.copy(), 13)
                mix_3_accuracy = test_data_vs_sessions(test_data, mix_3)

                # forward session selection
                mix_4 = create_mix_4(training_data, added_sessions, default_sessions)
                mix_4_accuracy = test_data_vs_sessions(test_data, mix_4)

                result.extend([mix_1_accuracy, mix_2_accuracy, mix_3_accuracy,
                               mix_4_accuracy])
                results.append(result)

            with open('update_default_list_7.csv', 'a') as f:
                line = f'{sravi_user},{repeat},' \
                       f'{len(training_data)},{len(test_data)},' \
                       f'{default_accuracy},{results}\n'
                f.write(line)


def analyse_2(file_path):
    columns = ['User ID', 'Attempt', 'Num Training', 'Num Test',
               'Default Accuracy', 'Added Accuracies']
    data = []
    regex = r'(\d+),(\d+),(\d+),(\d+),(\d+.\d+),(\[.+\])'

    user_mappings = {
        1: 'Fabian (SRAVI)',
        5: 'Liam (SRAVI)',
        12: 'Domhnall',
        6: 'Richard (SRAVI)',
        17: 'Michael (SRAVI)'
    }

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, attempt, num_training, num_test, default_accuracy, \
                added_accuracies = re.match(regex, line).groups()
            added_accuracies = ast.literal_eval(added_accuracies)
            data.append([int(user_id), int(attempt), int(num_training),
                         int(num_test), float(default_accuracy),
                         added_accuracies])

    df = pd.DataFrame(columns=columns, data=data)

    for user_id in df['User ID'].unique():
        sub_df = df[df['User ID'] == user_id]

        max_rows, max_cols = 2, 3
        fig, axs = plt.subplots(max_rows, max_cols)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'User: {user_id}')
        rows, columns = 0, 0

        av_mix_1_accuracies, av_mix_2_accuracies, av_mix_3_accuracies, \
            av_mix_4_accuracies, av_default_accuracies = [], [], [], [], []

        for index, row in sub_df.iterrows():
            attempt = row['Attempt']
            default_accuracy = row['Default Accuracy']
            added_accuracies = row['Added Accuracies']

            mix_1_accuracies, mix_2_accuracies, mix_3_accuracies, \
                mix_4_accuracies = [], [], [], []

            av_default_accuracies.append(default_accuracy)

            for added_session_id, mix_1_accuracy, mix_2_accuracy, \
                mix_3_accuracy, mix_4_accuracy in added_accuracies:
                mix_1_accuracies.append(mix_1_accuracy)
                mix_2_accuracies.append(mix_2_accuracy)
                mix_3_accuracies.append(mix_3_accuracy)
                mix_4_accuracies.append(mix_4_accuracy)

            av_mix_1_accuracies.append(mix_1_accuracies)
            av_mix_2_accuracies.append(mix_2_accuracies)
            av_mix_3_accuracies.append(mix_3_accuracies)
            av_mix_4_accuracies.append(mix_4_accuracies)

            x = [i for i in range(1, len(added_accuracies) + 1)]
            axs[rows, columns].axhline(default_accuracy, color='black',
                                       label='Default')
            axs[rows, columns].plot(x, mix_1_accuracies, label='External Rank All')
            axs[rows, columns].plot(x, mix_2_accuracies, label='Combine')
            axs[rows, columns].plot(x, mix_3_accuracies, label='Session Rank Defaults')
            axs[rows, columns].plot(x, mix_4_accuracies, label='Forward Selection')
            axs[rows, columns].set_xlabel('Num Added Sessions')
            axs[rows, columns].set_ylabel('Accuracy')
            axs[rows, columns].set_title(f'User {user_id}, Attempt {attempt}')
            axs[rows, columns].set_ylim([60, 101])
            axs[rows, columns].set_xticks(np.arange(min(x), max(x) + 1, 1.0))
            axs[rows, columns].legend()

            if columns == max_cols - 1:
                rows += 1
                columns = 0
            else:
                columns += 1

        def av_line(ll):
            av_line = []
            for i in range(len(ll[0])):
                av = 0
                for l in ll:
                    av += l[i]
                av /= len(ll)
                av_line.append(av)

            return av_line

        # show average over added sessions
        av_mix_1_accuracies = av_line(av_mix_1_accuracies)
        av_mix_2_accuracies = av_line(av_mix_2_accuracies)
        av_mix_3_accuracies = av_line(av_mix_3_accuracies)
        av_mix_4_accuracies = av_line(av_mix_4_accuracies)
        av_default_accuracy = \
            sum(av_default_accuracies) / len(av_default_accuracies)

        x = [i for i in range(1, len(added_accuracies) + 1)]
        axs[rows, columns].axhline(av_default_accuracy, color='black',
                                   label='Default')
        axs[rows, columns].plot(x, av_mix_1_accuracies, label='External Rank All')
        axs[rows, columns].plot(x, av_mix_2_accuracies, label='Combine')
        axs[rows, columns].plot(x, av_mix_3_accuracies, label='Session Rank Defaults')
        axs[rows, columns].plot(x, av_mix_4_accuracies, label='Forward Selection')
        axs[rows, columns].set_xlabel('Num Added Sessions')
        axs[rows, columns].set_ylabel('Accuracy')
        axs[rows, columns].set_title(f'User {user_id}, Average')
        axs[rows, columns].set_ylim([60, 101])
        axs[rows, columns].set_xticks(np.arange(min(x), max(x) + 1, 1.0))
        axs[rows, columns].legend()

        plt.show()

        # box plots over added sessions
        max_rows, max_cols = 2, 2
        fig, axs = plt.subplots(max_rows, max_cols)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'User: {user_id} - Technique Box Plots')
        rows, columns = 0, 0

        techniques = ['External Rank All', 'Combine', 'Session Rank Defaults',
                      'Forward Selection']

        for i in range(4):
            x = []
            for j in range(10):
                x_i = []
                for index, row in sub_df.iterrows():
                    added_accuracies = row['Added Accuracies']
                    x_i.append(added_accuracies[j][i+1])
                x.append(x_i)

            axs[rows, columns].boxplot(x)
            axs[rows, columns].set_xlabel('Num Added Sessions')
            axs[rows, columns].set_ylabel('Accuracy')
            axs[rows, columns].set_ylim([75, 101])
            axs[rows, columns].set_title(techniques[i])

            if columns == max_cols - 1:
                rows += 1
                columns = 0
            else:
                columns += 1

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('experiment')

    parser_2 = sub_parser.add_parser('analysis')
    parser_2.add_argument('file_path')

    args = parser.parse_args()
    run_type = args.run_type

    if run_type == 'experiment':
        # experiment_1()
        experiment_2()
    else:
        # analyse_1(args.file_path)
        analyse_2(args.file_path)
