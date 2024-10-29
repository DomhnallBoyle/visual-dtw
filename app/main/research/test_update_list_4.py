import argparse
import ast
import random
import re

import matplotlib.pyplot as plt
import pandas as pd
from main.models import Config
from main.research.test_update_list_3 import create_mix, \
    get_session_ids, get_default_sessions, get_user_sessions, VIDEOS_PATH, \
    MAX_TESTS
from main.services.transcribe import transcribe_signal
from matplotlib.ticker import MaxNLocator

kvs = {}


def to_labels(sessions):
    return frozenset([s[0] for s in sessions])


def kvs_accuracy_lookup(test_sessions, ref_sessions):
    test_session_labels = to_labels(test_sessions)
    ref_session_labels = to_labels(ref_sessions)

    accuracies = kvs.get(test_session_labels)
    if accuracies:
        return accuracies.get(ref_session_labels)

    return None


def get_accuracy(test_sessions, ref_sessions):
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
    test_session_labels = to_labels(test_sessions)
    ref_session_labels = to_labels(ref_sessions)
    accuracies = kvs.get(test_session_labels, {})
    accuracies[ref_session_labels] = accuracy
    kvs[test_session_labels] = accuracies

    return accuracy


def experiment(user_id, start_test_num=1):
    """Continuously use chosen B - does it tail off at any point?
    """
    # find all session ids for that user
    session_ids = get_session_ids(VIDEOS_PATH.format(user_id))
    max_allowed_sessions = len(session_ids) - 1 if len(session_ids) <= 5 else 5
    print('Session IDs: ', session_ids)
    print('Max Allowed Sessions: ', max_allowed_sessions)

    default_sessions = get_default_sessions()
    user_sessions = get_user_sessions(VIDEOS_PATH.format(user_id))

    global kvs
    kvs = {}

    for num_tests in range(start_test_num, MAX_TESTS + 1):
        for i in range(len(user_sessions)):
            test_session = user_sessions[i]
            sessions_to_add = user_sessions[:i] + user_sessions[i + 1:]

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

                # adding default sessions to as many defaults
                chosen_sessions = create_mix(
                    added_sessions=added_sessions.copy(),
                    default_sessions=default_sessions.copy(),
                    num_allowed_sessions=len(default_sessions)
                )
                accuracy = get_accuracy([test_session], chosen_sessions)
                result.append(accuracy)

                # just combine them
                chosen_sessions = added_sessions.copy() + \
                    default_sessions.copy()
                accuracy = get_accuracy([test_session], chosen_sessions)
                result.append(accuracy)

                user_accuracy = get_accuracy([test_session], added_sessions)
                result.append(user_accuracy)

                results.append(result)

            with open('update_default_list_5.csv', 'a') as f:
                line = f'{user_id},{num_tests},{test_session_label},' \
                       f'{default_accuracy},{results}\n'
                f.write(line)


def analysis(file_path):
    columns = ['User ID', 'Attempt', 'Test Session', 'Default Accuracy',
               'Added Accuracies']
    data = []
    regex = r'(\d+),(\d+),(.+),(\d+.\d+),(\[.+\])'

    user_mappings = {
        1: 'Fabian (SRAVI)',
        5: 'Liam (SRAVI)',
        12: 'Domhnall',
        6: 'Richard (SRAVI)',
        17: 'Michael (SRAVI)'
    }

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, attempt, test_session, default_accuracy, \
                added_accuracies = re.match(regex, line).groups()
            added_accuracies = ast.literal_eval(added_accuracies)
            data.append([int(user_id), int(attempt), test_session,
                         float(default_accuracy), added_accuracies])

    df = pd.DataFrame(columns=columns, data=data)

    max_rows, max_cols = 2, 3
    fig, axs = plt.subplots(max_rows, max_cols)
    fig.tight_layout()
    rows, columns = 0, 0

    users = df['User ID'].unique()
    for user in users:
        sub_df = df[df['User ID'] == user]
        average_default_accuracy = sub_df['Default Accuracy'].mean()

        added_session_count = len(sub_df.iloc[0]['Added Accuracies'])

        chosen_accuracies_1 = [[] for i in range(added_session_count)]
        chosen_accuracies_2 = [[] for i in range(added_session_count)]
        user_accuracies = [[] for i in range(added_session_count)]

        for index, row in sub_df.iterrows():
            added_accuracies = row['Added Accuracies']
            for i, accuracies in enumerate(added_accuracies):
                if len(accuracies) == 3:
                    session_label, chosen_accuracy, user_accuracy = accuracies
                    chosen_accuracies_1[i].append(chosen_accuracy)
                    user_accuracies[i].append(user_accuracy)
                elif len(accuracies) == 4:
                    session_label, chosen_accuracy_1, chosen_accuracy_2, \
                        user_accuracy = accuracies
                    chosen_accuracies_1[i].append(chosen_accuracy_1)
                    chosen_accuracies_2[i].append(chosen_accuracy_2)
                    user_accuracies[i].append(user_accuracy)

        def quantile(ll):
            results = []
            for l in ll:
                s = pd.Series(l)
                # results.append(s.min())
                results.append(s.mean())
                # results.append(s.quantile(0.1))
                # results.append(s.median())

            return results

        chosen_accuracies_1 = quantile(chosen_accuracies_1)
        chosen_accuracies_2 = quantile(chosen_accuracies_2)
        user_accuracies = quantile(user_accuracies)

        axs[rows, columns].plot(
            [i + 1 for i in range(len(chosen_accuracies_1))],
            chosen_accuracies_1, label='Chosen A')
        axs[rows, columns].plot(
            [i + 1 for i in range(len(chosen_accuracies_2))],
            chosen_accuracies_2, label='Chosen B')
        axs[rows, columns].plot(
            [i + 1 for i in range(len(user_accuracies))],
            user_accuracies, label='User')

        axs[rows, columns].axhline(average_default_accuracy, color='black',
                                   label='Default')
        axs[rows, columns].set_xlabel('Number of added sessions')
        axs[rows, columns].set_ylabel('Accuracy')
        axs[rows, columns].set_title(user_mappings[user])
        axs[rows, columns].legend()
        axs[rows, columns].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[rows, columns].set_ylim([60, 100])
        axs[rows, columns].set_xlim([1, 9])

        if columns == max_cols - 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.show()


def analysis_2(file_path):
    columns = ['User ID', 'Attempt', 'Test Session', 'Default Accuracy',
               'Added Accuracies']
    data = []
    regex = r'(\d+),(\d+),(.+),(\d+.\d+),(\[.+\])'

    user_mappings = {
        1: 'Fabian (SRAVI)',
        5: 'Liam (SRAVI)',
        12: 'Domhnall',
        6: 'Richard (SRAVI)',
        17: 'Michael (SRAVI)'
    }

    with open(file_path, 'r') as f:
        for line in f.readlines():
            user_id, attempt, test_session, default_accuracy, \
            added_accuracies = re.match(regex, line).groups()
            added_accuracies = ast.literal_eval(added_accuracies)
            data.append([int(user_id), int(attempt), test_session,
                         float(default_accuracy), added_accuracies])

    df = pd.DataFrame(columns=columns, data=data)

    users = df['User ID'].unique()
    for user in users:
        sub_df = df[df['User ID'] == user]
        unique_test_sessions = sub_df['Test Session'].unique()

        max_rows, max_cols = 4, 3
        fig, axs = plt.subplots(max_rows, max_cols)
        fig.tight_layout()
        rows, columns = 0, 0

        for test_session in unique_test_sessions:
            sub_sub_df = sub_df[sub_df['Test Session'] == test_session]
            average_default_accuracy = sub_sub_df['Default Accuracy'].mean()

            added_session_count = len(sub_sub_df.iloc[0]['Added Accuracies'])

            chosen_accuracies_1 = [[] for i in range(added_session_count)]
            chosen_accuracies_2 = [[] for i in range(added_session_count)]
            user_accuracies = [[] for i in range(added_session_count)]

            for index, row in sub_sub_df.iterrows():
                added_accuracies = row['Added Accuracies']
                for i, accuracies in enumerate(added_accuracies):
                    if len(accuracies) == 3:
                        session_label, chosen_accuracy, user_accuracy = accuracies
                        chosen_accuracies_1[i].append(chosen_accuracy)
                        user_accuracies[i].append(user_accuracy)
                    elif len(accuracies) == 4:
                        session_label, chosen_accuracy_1, chosen_accuracy_2, \
                        user_accuracy = accuracies
                        chosen_accuracies_1[i].append(chosen_accuracy_1)
                        chosen_accuracies_2[i].append(chosen_accuracy_2)
                        user_accuracies[i].append(user_accuracy)

            # lists_average = \
            #     lambda ll: [sum(l) / len(l) for l in ll if l]

            # chosen_accuracies_1 = lists_average(chosen_accuracies_1)
            # chosen_accuracies_2 = lists_average(chosen_accuracies_2)
            # user_accuracies = lists_average(user_accuracies)

            def quantile(ll):
                results = []
                for l in ll:
                    s = pd.Series(l)
                    # results.append(s.min())
                    results.append(s.mean())
                    # results.append(s.quantile(0.1))
                    # results.append(s.median())

                return results

            chosen_accuracies_1 = quantile(chosen_accuracies_1)
            chosen_accuracies_2 = quantile(chosen_accuracies_2)
            user_accuracies = quantile(user_accuracies)

            axs[rows, columns].plot(
                [i + 1 for i in range(len(chosen_accuracies_1))],
                chosen_accuracies_1, label='Chosen A')
            axs[rows, columns].plot(
                [i + 1 for i in range(len(chosen_accuracies_2))],
                chosen_accuracies_2, label='Chosen B')
            axs[rows, columns].plot(
                [i + 1 for i in range(len(user_accuracies))],
                user_accuracies, label='User')

            axs[rows, columns].axhline(average_default_accuracy, color='black',
                                       label='Default')
            axs[rows, columns].set_xlabel('Number of added sessions')
            axs[rows, columns].set_ylabel('Accuracy')
            axs[rows, columns].set_title(f'{user_mappings[user]} - '
                                         f'Test: {test_session}')
            axs[rows, columns].legend()
            axs[rows, columns].xaxis.set_major_locator(
                MaxNLocator(integer=True))
            axs[rows, columns].set_ylim([60, 100])
            axs[rows, columns].set_xlim([1, 9])

            if columns == max_cols - 1:
                rows += 1
                columns = 0
            else:
                columns += 1

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('experiment')

    parser_2 = sub_parsers.add_parser('analysis')
    parser_2.add_argument('file_path')

    args = parser.parse_args()
    run_type = args.run_type

    if run_type == 'experiment':
        users = [1, 5, 6, 12, 17]

        for user_id in users:
            experiment(user_id)
    else:
        analysis(args.file_path)
        # analysis_2(args.file_path)
