import argparse
import multiprocessing
import random
import time

from main.research.test_update_list_5 import test_data_vs_sessions, \
    LIOPA_DATA_PATH, get_user_sessions, get_test_data, get_default_sessions


def forward_session_selection(training_data, all_sessions, initial_max,
                              branch_out=True):
    """Recursive branching forward session selection"""
    selected_session_sets = []

    def recursive(_all_sessions, _session_selection=[], _best_accuracy=0):
        if len(_session_selection) < initial_max:
            accuracies = []
            for session in _all_sessions:
                accuracy = test_data_vs_sessions(
                    test_data=training_data,
                    sessions=[session] + _session_selection,
                    is_dtw_concurrent=False
                )
                accuracies.append(accuracy)

            # find max accuracy indexes
            max_accuracy = max(accuracies)
            max_accuracy_indexes = [i for i, accuracy in enumerate(accuracies)
                                    if accuracy == max_accuracy]
            if not branch_out:
                max_accuracy_indexes = [random.choice(max_accuracy_indexes)]

            for max_accuracy_index in max_accuracy_indexes:
                best_session = _all_sessions[max_accuracy_index]
                recursive(
                    _all_sessions=_all_sessions[:max_accuracy_index]
                                  + _all_sessions[max_accuracy_index+1:],
                    _session_selection=[best_session] + _session_selection,
                    _best_accuracy=max_accuracy
                )
        else:
            if len(_all_sessions) == 0:
                selected_session_sets.append(_session_selection)
                return

            accuracies = []
            for session in _all_sessions:
                accuracy = test_data_vs_sessions(
                    test_data=training_data,
                    sessions=[session] + _session_selection,
                    is_dtw_concurrent=False
                )
                accuracies.append(accuracy)

            # find max accuracy indexes
            max_accuracy = max(accuracies)
            max_accuracy_indexes = [i for i, accuracy in enumerate(accuracies)
                                    if accuracy == max_accuracy]
            if not branch_out:
                max_accuracy_indexes = [random.choice(max_accuracy_indexes)]

            for max_accuracy_index in max_accuracy_indexes:
                max_accuracy = accuracies[max_accuracy_index]
                if max_accuracy >= _best_accuracy:
                    best_session = _all_sessions[max_accuracy_index]
                    recursive(
                        _all_sessions=_all_sessions[:max_accuracy_index]
                                      + _all_sessions[max_accuracy_index+1:],
                        _session_selection=[best_session] + _session_selection,
                        _best_accuracy=max_accuracy
                    )
                else:
                    selected_session_sets.append(_session_selection)

    recursive(all_sessions.copy())

    return selected_session_sets


def selected_session_sets_analysis(all_sessions, selected_session_sets):
    # SESSION SET ANALYSIS
    # check what the state of the selection session sets are
    # are there sessions selected more times than others?
    selected_counts, leftover_counts = {}, {}
    num_session_sets = len(selected_session_sets)

    all_session_labels = [session[0] for session in all_sessions]
    selected_session_labels = [session[0]
                               for session_set in selected_session_sets
                               for session in session_set]

    leftover_session_labels = []
    for session_set in selected_session_sets:
        session_labels = [s[0] for s in session_set]
        leftover_session_labels.extend(
            list(set(all_session_labels) - set(session_labels))
        )

    selected_composition, leftover_composition = {}, {}

    for session_label in selected_session_labels:
        selected_counts[session_label] = \
            selected_counts.get(session_label, 0) + 1
        _type = session_label.split('_')[0]
        selected_composition[_type] = selected_composition.get(_type, 0) + 1

    for session_label in leftover_session_labels:
        leftover_counts[session_label] = \
            leftover_counts.get(session_label, 0) + 1
        _type = session_label.split('_')[0]
        leftover_composition[_type] = leftover_composition.get(_type, 0) + 1

    # % of times session label appeared/didn't appear in all the session sets
    selected_counts = {
        k: v / num_session_sets
        for k, v in selected_counts.items()
    }
    leftover_counts = {
        k: v / num_session_sets
        for k, v in leftover_counts.items()
    }

    # composition of selected/leftover i.e. added and default
    selected_composition = {
        k: v / len(selected_session_labels)
        for k, v in selected_composition.items()
    }
    leftover_composition = {
        k: v / len(leftover_session_labels)
        for k, v in leftover_composition.items()
    }

    print('Percentage selected: ', selected_counts)
    print('Percentage leftover: ', leftover_counts)

    print('Selected composition: ', selected_composition)
    print('Leftover composition: ', leftover_composition)


def process_fold(_id, _training_data, _testing_data, _all_sessions,
                 _branch_out):
    print('\nFold:', _id)
    print('Training:', len(_training_data))
    print('Testing:', len(_testing_data))
    print('Branch out: ', _branch_out)

    # getting multiple session sets back
    start_time = time.time()
    selected_session_sets = forward_session_selection(
        training_data=_training_data,
        all_sessions=_all_sessions,
        initial_max=13,
        branch_out=_branch_out
    )
    end_time = time.time()

    # analysis
    selected_session_sets_analysis(_all_sessions, selected_session_sets)

    print(f'Num session sets {_id}:', len(selected_session_sets))
    print(f'Time taken (mins) {_id}:', (end_time - start_time) / 60)

    accuracies = [
        test_data_vs_sessions(_testing_data, session_set)
        for session_set in selected_session_sets]

    return accuracies, selected_session_sets


def cross_validation(training_data, all_sessions, k=5, branch_out=True):
    """CV so selected sessions aren't over-fitting to the training data"""
    subset_size = int(len(training_data) / k)

    random.shuffle(training_data)

    print('\nCross Validation:')

    all_accuracies, all_session_mixes = [], []

    tasks = []
    for i in range(k):
        testing_this_round = training_data[i * subset_size:][:subset_size]
        training_this_round = \
            training_data[:i * subset_size] \
            + training_data[(i + 1) * subset_size:]

        assert \
            len(testing_this_round) + len(training_this_round) \
            == len(training_data)

        tasks.append([
            i+1,
            training_this_round,
            testing_this_round,
            all_sessions.copy(),
            branch_out
        ])

    # multi core processing
    num_processes = multiprocessing.cpu_count() - 1 or 1
    print('Num Processes: ', num_processes)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_fold, tasks)
        for accuracies, selected_session_sets in results:
            all_accuracies.extend(accuracies)
            all_session_mixes.extend(selected_session_sets)

    # # single core processing
    # for task in tasks:
    #     accuracies, selected_session_sets = process_fold(*task)
    #     all_accuracies.extend(accuracies)
    #     all_session_mixes.extend(selected_session_sets)

    assert len(all_accuracies) == len(all_session_mixes)

    # find max accuracies
    max_accuracy = max(all_accuracies)
    max_accuracy_indexes = [i for i, accuracy in enumerate(all_accuracies)
                            if accuracy == max_accuracy]

    if len(max_accuracy_indexes) == 1:
        return all_session_mixes[max_accuracy_indexes[0]]

    # test remaining session mixes vs training data
    sub_accuracies, sub_session_mixes = [], []
    for max_accuracy_index in max_accuracy_indexes:
        session_mix = all_session_mixes[max_accuracy_index]
        sub_accuracies.append(
            test_data_vs_sessions(training_data, session_mix)
        )
        sub_session_mixes.append(session_mix)

    # find max accuracies
    max_accuracy = max(sub_accuracies)
    max_accuracy_indexes = [i for i, accuracy in enumerate(sub_accuracies)
                            if accuracy == max_accuracy]

    if len(max_accuracy_indexes) == 1:
        return sub_session_mixes[max_accuracy_indexes[0]]

    # test remaining session mixes against themselves
    sub_accuracies = []
    for i in range(len(sub_session_mixes)):
        ref_sessions = sub_session_mixes[i]
        test_session_sets = sub_session_mixes[:i] + sub_session_mixes[i+1:]

        test_data = [(label, template)
                     for test_sessions in test_session_sets
                     for session_label, session in test_sessions
                     for label, template in session]

        accuracy = test_data_vs_sessions(test_data, ref_sessions)
        sub_accuracies.append(accuracy)

    # finally, return best performing session mix
    return sub_session_mixes[sub_accuracies.index(max(sub_accuracies))]


def experiment(args):
    users = {
        '1': '11',
        '6': '9'
    }

    for sravi_user, pava_user in users.items():
        # get the data first
        sessions_to_add = \
            get_user_sessions(LIOPA_DATA_PATH.format(user_id=sravi_user))
        train_test_data = get_test_data(user_id=pava_user)
        default_sessions = get_default_sessions()

        # repeat adding sessions a number of times
        for repeat in range(1, args.num_repeats + 1):

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
                selected_session_mix = cross_validation(
                    training_data=training_data,
                    all_sessions=added_sessions + default_sessions,
                    k=args.k,
                    branch_out=args.branch_out
                )
                mix_1_accuracy = \
                    test_data_vs_sessions(test_data, selected_session_mix)
                print('Test Accuracy: ', mix_1_accuracy, '\n')

                result.append(mix_1_accuracy)
                results.append(result)

            with open('update_default_list_11.csv', 'a') as f:
                line = f'{sravi_user},{repeat},' \
                       f'{len(training_data)},{len(test_data)},' \
                       f'{default_accuracy},{results}\n'
                f.write(line)


def main(args):
    experiment(args)


def bool_type(s):
    return s == 'True'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_repeats', default=5, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--branch_out', type=bool_type, default=True)

    main(parser.parse_args())
