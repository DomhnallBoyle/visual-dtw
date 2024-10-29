"""Experimenting with session updating on liopa new captured data
"""
import argparse
import ast
import random
import re
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import PAVAList, PAVAModel, PAVAModelSession, PAVASession, \
    PAVAUser
from main.research.test_update_list_3 import get_default_sessions, \
    get_user_sessions
from main.research.test_update_list_5 import test_data_vs_sessions
from main.research.test_update_list_6 import cross_validation as algorithm_1
from matplotlib.ticker import MaxNLocator
from sqlalchemy.orm import joinedload, noload


def experiment(args):
    default_sessions = get_default_sessions()

    user_sessions = get_user_sessions(args.videos_directory)
    half_index = len(user_sessions) // 2
    print('Num User Sessions:', len(user_sessions))
    print('Half Index:', half_index)

    liopa_user = args.videos_directory.split('/')[-1]
    if not liopa_user:
        liopa_user = args.videos_directory.split('/')[-2]

    for repeat in range(1, args.num_repeats + 1):
        # extract sessions to add and sessions for training/test
        random.shuffle(user_sessions)
        sessions_to_add = user_sessions[:half_index]
        train_test_sessions = user_sessions[half_index:]

        print('Num sessions to add:', len(sessions_to_add))
        print('Num train/test sessions:', len(train_test_sessions))

        # now extract training/test data
        # in production, this will be all user transcriptions
        # using sessions for now
        train_test_data = [(label, template)
                           for session_label, session in train_test_sessions
                           for label, template in session]
        random.shuffle(train_test_data)
        train_split = int(len(train_test_data) * 0.6)
        training_data = train_test_data[:train_split]
        test_data = train_test_data[train_split:]
        print('Training/Test Data:', len(train_test_data))
        print('Training Data:', len(training_data))
        print('Test Data:', len(test_data))
        assert len(training_data) + len(test_data) == len(train_test_data)

        # get default accuracy
        default_accuracy = test_data_vs_sessions(test_data,
                                                 default_sessions)
        print('Default Accuracy:', default_accuracy)

        # start adding sessions
        added_sessions = []
        results = []
        random.shuffle(sessions_to_add)
        for session_to_add in sessions_to_add:
            session_id_to_add = session_to_add[0]
            result = [session_id_to_add]

            added_sessions.append(session_to_add)

            # cross validation with forward session selection
            mix, average_time = algorithm_1(training_data, added_sessions,
                                            default_sessions, k=args.k,
                                            multiple_max_test=True)
            mix_accuracy = test_data_vs_sessions(test_data, mix)
            print('Mix Accuracy:', mix_accuracy)
            print('Average time:', average_time)

            mix_labels = [session_label
                          for session_label, session_templates in mix]

            result.extend([mix_accuracy, average_time, mix_labels])
            results.append(result)

        with open('update_default_list_12.csv', 'a') as f:
            line = f'{liopa_user},{repeat},{args.k},' \
                   f'{len(training_data)},{len(test_data)},' \
                   f'{default_accuracy},{results}\n'
            f.write(line)


def session_selection_algorithm(user_sessions, default_sessions,
                                training_data=None, split_index=0.7, k=4):
    user_sessions = [(session.str_id, [
        (template.phrase.content, template) for template in session.templates
    ]) for session in user_sessions]
    default_sessions = [(session.str_id, [
        (template.phrase.content, template) for template in session.templates
    ]) for session in default_sessions]

    if not training_data:
        split = int(len(user_sessions) * split_index)
        random.shuffle(user_sessions)

        sessions_to_add = user_sessions[:split]

        training_sessions = user_sessions[split:]
        training_data = [(label, template)
                         for session_label, session in training_sessions
                         for label, template in session]

        print('Num sessions to add:', len(sessions_to_add))
        print('Num training sessions:', len(training_sessions))
    else:
        sessions_to_add = user_sessions

    mix, average_time = algorithm_1(training_data, sessions_to_add,
                                    default_sessions, k=k,
                                    multiple_max_test=True)

    return [session_id for session_id, session_templates in mix]


def production_experiment(args):
    user_ids = args.user_ids
    if not user_ids:
        user_ids = PAVAUser.get(query=(PAVAUser.id,))

    for user_id in user_ids:

        # only grab the list ids belonging to the user that aren't archived:
        list_ids = PAVAList.get(
            query=(PAVAList.id,),
            filter=(
                (PAVAList.user_id == user_id)
                & (PAVAList.archived == False)
            )
        )

        for list_id in list_ids:
            lst = PAVAList.get(
                loading_options=(noload('phrases')),
                filter=(PAVAList.id == list_id),
                first=True
            )

            # check if sessions are new and completed
            # i.e. only run algorithm if there are new completed sessions
            if not any(session.completed and session.new
                       for session in lst.sessions):
                continue

            # get all completed sessions for the update (new or not)
            list_sessions = [session for session in lst.sessions
                             if session.completed]

            # grab default sessions if required
            if lst.default:
                default_sessions = PAVASession.get(
                    filter=(
                        PAVASession.list_id
                        == configuration.DEFAULT_PAVA_LIST_ID
                    )
                )

                print('User ID:', lst.user_id)
                print('User Sessions:', len(list_sessions))
                print('Default Sessions:', len(default_sessions))

                # run algorithm, get session ids out
                session_ids = session_selection_algorithm(
                    user_sessions=list_sessions,
                    default_sessions=default_sessions,
                    k=args.k
                )

                # create user model
                model = PAVAModel.create(
                    list_id=lst.id,
                )

                # create model sessions
                for session_id in session_ids:
                    PAVAModelSession.create(
                        model_id=model.id,
                        session_id=session_id
                    )

                # update list to point to new model
                PAVAList.update(
                    id=lst.id,
                    loading_options=(noload('phrases'), noload('sessions')),
                    current_model_id=model.id
                )

            # set applicable sessions to no longer be new
            for session in list_sessions:
                if session.new:
                    PAVASession.update(id=session.id,
                                       loading_options=(noload('templates')),
                                       new=False)


def production_test(args):
    """Test unseen test data with
    a) default list
    b) saved models in db
    """
    import glob
    import os
    from main import configuration
    from main.models import PAVATemplate
    from main.research.videos_vs_videos import get_accuracy, get_templates
    from main.utils.io import read_pickle_file

    unseen_videos_directory = args.videos_directory
    unseen_video_paths = \
        glob.glob(os.path.join(unseen_videos_directory, '*.mp4'))
    unseen_templates = get_templates(unseen_video_paths, None)

    # get default reference signals
    default_list = read_pickle_file(configuration.DEFAULT_LIST_PATH)
    default_ref_signals = [
        (phrase.content, template.blob.astype(np.float32))
        for phrase in default_list.phrases
        for template in phrase.templates
    ]

    # get model reference signals
    user_lst = PAVAList.get(
        loading_options=(
            joinedload('phrases').noload('templates'),
            noload('sessions')
        ),
        filter=(PAVAList.user_id == args.user_id),
        first=True
    )
    ref_templates = PAVATemplate.get(
        filter=(
            (PAVATemplate.session_id == PAVAModelSession.session_id)
            & (PAVAModelSession.model_id == user_lst.current_model_id)
        )
    )

    # always check updated database for archived phrases
    phrase_lookup = {
        phrase.content: phrase for phrase in user_lst.phrases
    }
    model_ref_signals = [
        (template.phrase.content, template.blob)
        for template in ref_templates
        if not phrase_lookup[template.phrase.content].archived
           and not phrase_lookup[template.phrase.content].is_nota
    ]

    get_accuracy(default_ref_signals, unseen_templates)
    get_accuracy(model_ref_signals, unseen_templates)


def analysis(args):
    data, columns = [], ['User ID', 'Repeat ID', 'K', 'Num Training',
                         'Num Testing', 'Default Accuracy', 'Added Results']
    line_regex = r'(\d+),(\d+),(\d+),(\d+),(\d+),(.+),(\[.+\])'

    user_id_mapping = {
        '2': 'Alex',
        '3': 'Adrian',
        '5': 'Liam',
        '9': 'Richard',
        '10': 'Chris',
        '12': 'Domhnall',
        '17': 'Michael',
        '18': 'Shane'
    }

    with open(args.results_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            user_id, repeat_id, k, num_training, num_testing, \
                default_accuracy, added_results = \
                re.match(line_regex, line).groups()
            data.append([
                user_id,
                repeat_id,
                k,
                num_training,
                num_testing,
                float(default_accuracy),
                ast.literal_eval(added_results)
            ])

    df = pd.DataFrame(columns=columns, data=data)

    unique_users = df['User ID'].unique()

    # show graphs
    if args.show_graphs:
        for user_id in unique_users:
            sub_df = df[df['User ID'] == user_id]
            av_added_accuracies, av_added_times = [], []
            av_default_accuracy = 0
            i, j = 0, 0
            num_rows, num_columns = 2, 3
            fig, ax = plt.subplots(num_rows, num_columns)
            fig.suptitle(f'User: {user_id_mapping[user_id]}')
            for index, row in sub_df.iterrows():
                # plot 1
                added_results = row['Added Results']
                num_added_sessions = len(added_results)
                added_labels = [
                    f'S{result[0].split("_")[1]}'
                    for result in added_results
                ]
                added_accuracies = [result[1] for result in added_results]
                av_added_accuracies.append(added_accuracies)
                av_default_accuracy += row['Default Accuracy']
                x = list(range(1, num_added_sessions + 1))
                ax[i, j].plot(x, added_accuracies, marker='x')
                ax[i, j].hlines(y=row['Default Accuracy'], xmin=1,
                                xmax=num_added_sessions)
                ax[i, j].set_title(f'Repeat: {row["Repeat ID"]}')
                ax[i, j].set_ylabel('Accuracy %')
                ax[i, j].set_xlabel('Num Added Sessions')
                ax[i, j].set_ylim((0, 110))
                ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                # plot 2
                added_times = [result[2] for result in added_results]
                av_added_times.append(added_times)
                ax_2 = ax[i, j].twinx()
                ax_2.plot(x, added_times, color='red')
                ax_2.set_ylabel('Time (mins)')
                ax_2.set_ylim((0, 110))

                for k, (x, y) in enumerate(zip(x, added_accuracies)):
                    ax[i, j].annotate(added_labels[k], (x, y),
                                      textcoords='offset points',
                                      xytext=(0, 10), ha='center')
                if j == num_columns - 1:
                    j = 0
                    i += 1
                else:
                    j += 1

            # show average plot
            i, j = num_rows - 1, num_columns - 1
            av_added_accuracies = list(np.asarray(av_added_accuracies).mean(axis=0))
            av_added_times = list(np.asarray(av_added_times).mean(axis=0))

            num_added_sessions = len(av_added_accuracies)
            av_default_accuracy /= len(sub_df)
            x = list(range(1, num_added_sessions + 1))

            # av plot 1
            ax[i, j].plot(x, av_added_accuracies, marker='x', label='Added')
            ax[i, j].hlines(y=av_default_accuracy, xmin=1, xmax=num_added_sessions,
                            label='Default')
            ax[i, j].set_title(f'Average')
            ax[i, j].set_ylabel('Accuracy %')
            ax[i, j].set_xlabel('Num Added Sessions')
            ax[i, j].set_ylim((0, 105))
            ax[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

            # av plot 2
            ax_2 = ax[i, j].twinx()
            ax_2.plot(x, av_added_times, color='red', label='Time Taken')
            ax_2.set_ylabel('Time (mins)')
            ax_2.set_ylim((0, 110))

            fig.legend()

            # maximise window
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()

            plt.subplots_adjust(
                top=0.902, bottom=0.106, left=0.06, right=0.945, hspace=0.378,
                wspace=0.375
            )
            plt.show()

    # analyse session mixes for patterns in added/removed sessions
    total_default_sessions = 13
    for user_id in unique_users:
        print('User:', user_id_mapping[user_id])
        sub_df = df[df['User ID'] == user_id]
        added_results = sub_df['Added Results'].values[0]

        most_common_first_13 = {}
        session_counts = {}
        for i, added_result in enumerate(added_results):
            added_session = added_result[0]
            mix_sessions = added_result[3]

            num_total_combination = total_default_sessions + (i + 1)
            num_chosen = len(mix_sessions)
            num_leftover = num_total_combination - num_chosen

            # is the added session always in the mix?
            if added_session not in mix_sessions:
                print(f'{added_session} not in mix...')

            print('Added session:', i+1, added_session)

            # check first 13 sessions
            first_13_sessions = mix_sessions[:total_default_sessions]
            for j, session in enumerate(first_13_sessions):
                # get positional look at sessions
                position_sessions = most_common_first_13.get(j+1, [])
                position_sessions.append(session)
                most_common_first_13[j+1] = position_sessions

                session_counts[session] = session_counts.get(session, 0) + 1

        # find patterns between most common 13
        pprint(most_common_first_13)

        # find most common defaults from 13
        default_counts = [(k, v) for k, v in session_counts.items()
                          if k.startswith('default')]
        default_counts = sorted(default_counts, key=lambda x: x[1],
                                reverse=True)
        pprint(default_counts)

        # can we assume to start with our added sessions first?
        added_counts = [(k, v) for k, v in session_counts.items()
                        if k.startswith('added')]
        added_counts = sorted(added_counts, key=lambda x: x[1],
                              reverse=True)
        pprint(added_counts)

        print()


def main(args):
    f = {
        'experiment': experiment,
        'production_experiment': production_experiment,
        'production_test': production_test,
        'analysis': analysis
    }

    if args.run_type not in f:
        parser.print_usage()
        exit()

    f[args.run_type](args)


def l(s):
    return s.split(',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('experiment')
    parser_1.add_argument('videos_directory')
    parser_1.add_argument('--num_repeats', type=int, default=5)
    parser_1.add_argument('--k', type=int, default=5)

    parser_2 = sub_parsers.add_parser('production_experiment')
    parser_2.add_argument('--user_ids', type=l)
    parser_2.add_argument('--k', type=int, default=5)

    parser_3 = sub_parsers.add_parser('production_test')
    parser_3.add_argument('videos_directory')
    parser_3.add_argument('user_id')

    parser_4 = sub_parsers.add_parser('analysis')
    parser_4.add_argument('results_path')
    parser_4.add_argument('--show_graphs', action='store_true')

    main(parser.parse_args())
