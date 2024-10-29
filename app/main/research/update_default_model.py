"""
Experiments to update the default model to improve accuracy for mainly
patient users using data from the pava groundtruth tool
"""
import argparse
import ast
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main.research.research_utils import get_default_sessions, \
    sessions_to_templates, create_templates, get_accuracy, get_phrases, \
    create_sessions, RECORDINGS_REGEX, SRAVI_EXTENDED_REGEX, create_template, \
    transcribe
from main.utils.io import read_csv_file, read_pickle_file, write_pickle_file
from scripts.session_selection import \
    session_selection_with_cross_validation_fast
from tqdm import tqdm

LIOPA_USER_IDS = [
    '05a93c37-358e-4d1e-986e-749d1b5124bd',
    '335ba19b-7e7c-4697-8284-2db149cb2fc3',
    '7bd67afd-0d47-46ac-b0fa-c6653b386e1a',
    '508765ab-5223-4947-9649-e58f579b1132',
    'aefe1462-07bc-4d0c-9600-fe049631164c',
    '858b2170-6ca5-4114-be27-94ce4ffcd702',
    '962a2a1d-2bb5-42e5-ad91-2e076773c9f3',
    'e4167017-614c-44ee-809b-484b438f50da'
]

# TODO: Run SSA for each user using all the sessions we have available
#  instead of covering all user accuracies together (not sure)


def analyse_users_vs_default(args):
    """Analyse the data from the groundtruth tool to find
    best and worst users vs the current default model
    """
    user_ids = os.listdir(args.groundtruth_directory)

    rank_weights = [0.6, 0.3, 0.1]
    data = []
    for user_id in user_ids:
        user_directory = os.path.join(args.groundtruth_directory, user_id)
        csv_path = os.path.join(user_directory, 'data.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)

        num_samples = len(df)
        rank_tallies = [
            len(df[df['Rank Position'] == position])
            for position in [1, 2, 3]
        ]
        num_notas = len(df[df['Rank Position'] == -1])
        assert sum(rank_tallies) + num_notas == num_samples

        rank_accuracies = [
            sum([rank_tallies[j] for j in range(i+1)]) / num_samples
            for i in range(len(rank_tallies))
        ]

        weighted_accuracy_sum = \
            sum([accuracy * weight
                 for accuracy, weight in zip(rank_accuracies, rank_weights)])

        if num_samples >= 30:
            data.append([user_id,
                         rank_accuracies,
                         weighted_accuracy_sum,
                         num_samples])

    # now sort by metric
    sorted_user_data = sorted(data, key=lambda x: x[2])[:args.user_limit]
    for data in sorted_user_data:
        print(data)


def construct_complete_sessions(dataset_sessions, user_templates, num_phrases):
    """Create complete sessions from all the groundtruth data and any
    datasets I have. Mixed sessions don't matter as long as they're complete
    """
    sessions = []
    leftover_phrase_templates = {}

    # add any complete dataset sessions first
    removed_indices = []
    for i, session in enumerate(dataset_sessions):
        if len(session[1]) == num_phrases:
            session_to_add = dataset_sessions[i]
            sessions.append((session_to_add[0],
                             [t[:2] for t in session_to_add[1]]))
            removed_indices.append(i)
    leftover_dataset_sessions = [s for i, s in enumerate(dataset_sessions)
                                 if i not in removed_indices]

    # add any leftover from the dataset sessions
    for label, session in leftover_dataset_sessions:
        for phrase, blob in zip(*list(zip(*session))[:2]):
            leftover_templates = leftover_phrase_templates.get(phrase, [])
            leftover_templates.append(blob)
            leftover_phrase_templates[phrase] = leftover_templates

    # construct any sessions from user templates
    for user_id, templates in user_templates.items():

        # create phrase templates dictionary
        phrase_templates = {}
        for phrase, blob in templates:
            blobs = phrase_templates.get(phrase, [])
            blobs.append(blob)
            phrase_templates[phrase] = blobs

        if len(phrase_templates.keys()) == num_phrases:
            # i.e. there are full sessions here
            max_num_sessions = min([
                len(l) for l in phrase_templates.values()
            ])

            print(f'Creating {max_num_sessions} user sessions...')

            # create sessions
            for i in range(max_num_sessions):
                user_session = []
                for phrase, templates in phrase_templates.items():
                    random.shuffle(templates)
                    user_session.append((phrase, templates[i]))
                    templates[i] = None  # marked as None for deletion
                sessions.append((f'{user_id}_S{i+1}', user_session))

        # add any leftover templates
        for phrase, templates in phrase_templates.items():
            # remove already used
            templates = [t for t in templates if t is not None]
            leftover_templates = leftover_phrase_templates.get(phrase, [])
            leftover_templates.extend(templates)
            leftover_phrase_templates[phrase] = leftover_templates

    max_leftover_sessions = min([
        len(l) for l in leftover_phrase_templates.values()
    ])

    # construct sessions from the leftover templates
    for i in range(max_leftover_sessions):
        leftover_session = []
        for phrase, templates in leftover_phrase_templates.items():
            random.shuffle(templates)
            leftover_session.append((phrase, templates[i]))
        sessions.append((f'leftover_S{i + 1}', leftover_session))

    assert all([len(s[1]) == num_phrases for s in sessions])

    return sessions


def find_default_models(args):
    """Find Default Models

    First construct all complete sessions from datasets and groundtruth data
    Any samples leftover, make sessions with them
    Continuously run SSA with these constructed sessions and training users
    Save models for analysis later
    """
    phrase_set = get_phrases('PAVA-DEFAULT')
    phrases = list(phrase_set.values())
    training_templates = {}

    # gather data capture sessions
    dataset_sessions = []
    for dataset_path in args.dataset_directories:
        for user_id in os.listdir(dataset_path):
            user_directory = os.path.join(dataset_path, user_id)
            if not os.path.isdir(user_directory):
                continue  # exclude zips etc
            user_id_tag = f'{os.path.basename(dataset_path)}_{user_id}'
            user_sessions = create_sessions(
                user_directory,
                regexes=[RECORDINGS_REGEX, SRAVI_EXTENDED_REGEX],
                phrase_lookup=phrase_set,
                save=True,
                debug=True,
                include_video_paths=True,
                user_id_tag=user_id_tag
            )

            # sort to training templates or sessions to add
            if user_id_tag in args.training_users:
                training_templates[user_id_tag] \
                    = sessions_to_templates(user_sessions)
            else:
                dataset_sessions.extend(user_sessions)

    # gather groundtruth user templates
    user_ids = os.listdir(args.groundtruth_directory)
    user_templates_d = {}
    for user_id in user_ids:
        data_directory = os.path.join(args.groundtruth_directory, user_id)
        user_templates = create_templates(data_directory, save=True,
                                          phrase_column='Groundtruth',
                                          include_video_paths=True,
                                          debug=True)
        user_templates = [t[:2] for t in user_templates
                          if t[0] in phrases]  # exclude NOTA etc

        # sort to training templates or templates for session creation
        if user_id in args.training_users:
            training_templates[user_id] = user_templates
        else:
            user_templates_d[user_id] = user_templates

    # show make-up of training templates
    print('Training templates make-up:')
    for user_id, templates in training_templates.items():
        print(user_id, len(templates))

    training_templates = [
        template
        for user_id, templates in training_templates.items()
        for template in templates
    ]

    data = []
    counter = 1
    while True:
        try:
            print(f'\nRepeat {counter}')

            # grab all the complete sessions
            sessions_to_add = construct_complete_sessions(dataset_sessions,
                                                          user_templates_d,
                                                          num_phrases=20)
            session_users = set([s[0].split('-')[0] for s in sessions_to_add])
            assert all([training_user not in session_users
                        for training_user in args.training_users])

            print('Sessions to add:', len(sessions_to_add))
            print('Training templates:', len(training_templates))

            if args.fast:
                selected_ids = session_selection_with_cross_validation_fast(
                    _sessions=sessions_to_add,
                    _training_templates=training_templates,
                    max_num_sessions=args.max_num_sessions
                )
            else:
                selected_ids = []

            print('Selected:', len(selected_ids))

            selected_sessions = [s for s in sessions_to_add
                                 if s[0] in selected_ids]

            data.append([
                counter,
                selected_sessions,
            ])
            counter += 1
        except KeyboardInterrupt:
            break

    if data:
        write_pickle_file(data, args.save_path)


def test_default_models(args):
    models = read_pickle_file(args.models_path)

    phrase_set = get_phrases('PAVA-DEFAULT')
    phrases = list(phrase_set.values())

    user_templates = {}

    # grab dataset templates
    for dataset_path in args.dataset_directories:
        for user_id in os.listdir(dataset_path):
            user_directory = os.path.join(dataset_path, user_id)
            if not os.path.isdir(user_directory):
                continue  # exclude zips etc
            user_id_tag = f'{os.path.basename(dataset_path)}_{user_id}'
            user_sessions = create_sessions(
                user_directory,
                regexes=[RECORDINGS_REGEX, SRAVI_EXTENDED_REGEX],
                phrase_lookup=phrase_set,
                save=args.save,
                include_video_paths=True,
                redo=args.redo,
                debug=True,
                user_id_tag=user_id_tag
            )
            user_templates[user_id_tag] = \
                sessions_to_templates(user_sessions)

    # grab groundtruth templates
    user_ids = os.listdir(args.groundtruth_directory)
    for user_id in user_ids:
        data_directory = os.path.join(args.groundtruth_directory, user_id)
        templates = create_templates(data_directory,
                                     save=args.save,
                                     phrase_column='Groundtruth',
                                     include_video_paths=True,
                                     redo=args.redo,
                                     debug=True)
        if not args.include_nota:
            templates = [(phrase, template, video_path)
                         for phrase, template, video_path in templates
                         if phrase in phrases]  # exclude NOTA etc
        user_templates[user_id.split('-')[0]] = templates

    # get default accuracies first
    default_templates = sessions_to_templates(get_default_sessions())
    old_default_accuracies_d = {}
    print('Getting default accuracies first...')
    for test_user, test_templates in tqdm(user_templates.items()):
        if len(test_templates) >= args.min_num_templates:
            labels, blobs, video_paths = zip(*test_templates)
            test_templates = list(zip(labels, blobs))
            old_default_accuracies_d[test_user] = \
                get_accuracy(default_templates, test_templates)

    # remove previous results path if exists
    for path in [args.output_path_1, args.output_path_2]:
        if os.path.exists(path):
            os.remove(path)

    # test models
    for counter, selected_sessions in models:
        print(f'\nModel {counter}')

        selected_users = set([s[0].split('-')[0] for s in selected_sessions])
        print(selected_users)

        selected_session_templates = sessions_to_templates(selected_sessions)

        # exclude selected users
        test_users = list(set(user_templates.keys()) - selected_users)
        for test_user in tqdm(test_users):
            test_templates = user_templates[test_user]
            if not len(test_templates) >= args.min_num_templates:
                continue

            labels, blobs, video_paths = zip(*test_templates)
            test_templates = list(zip(labels, blobs))

            old_default = old_default_accuracies_d[test_user]
            new_default = get_accuracy(selected_session_templates,
                                       test_templates)

            # output accuracies
            with open(args.output_path_1, 'a') as f:
                f.write(f'{counter},'
                        f'{test_user},'
                        f'{old_default[0]},'
                        f'{new_default[0]},'
                        f'{len(test_templates)}\n')

            # ensure there are equal numbers of labels, video paths and results
            assert len(labels) == \
                   len(video_paths) == \
                   len(old_default[2]) == \
                   len(new_default[2])

            # output individual predictions
            with open(args.output_path_2, 'a') as f:
                for label, video_path, old_prediction, new_prediction in \
                        zip(labels, video_paths,
                            old_default[2], new_default[2]):
                    f.write(f'{counter},'
                            f'{test_user},'
                            f'{os.path.basename(video_path)},'
                            f'{label},'
                            f'{old_prediction},'
                            f'{new_prediction}\n')


def analyse_default_models(args):
    df = read_csv_file(
        args.results_path,
        columns=['Model ID', 'User ID',
                 'Previous Default Accs', 'New Default Accs', 'Num Samples'],
        regex=r'(\d+),(.+),(\[.+\]),(\[.+\]),(\d+)',
        process_line_data=lambda row: [
            int(row[0]),
            row[1],
            ast.literal_eval(row[2]),
            ast.literal_eval(row[3]),
            int(row[4])
        ]
    )

    def get_users(in_prefixes):
        specific_user_ids = []
        prefixes = ['1_initial', '2_lighting', '3_more_phrases']
        for user_id in all_user_ids:
            if any([user_id.startswith(prefix) for prefix in prefixes]) \
                    == in_prefixes:
                specific_user_ids.append(user_id)
        return specific_user_ids

    all_user_ids = list(df['User ID'].unique())
    title = 'All Users'
    if args.pava_only:
        all_user_ids = get_users(False)
        title = 'PAVA Users'
    elif args.liopa_only:
        all_user_ids = get_users(True)
        title = 'Liopa Users'

    if args.exclude_liopa_pava_users:
        all_user_ids = [user_id for user_id in all_user_ids
                        if not any([user_id == liopa_user_id.split('-')[0]
                                    for liopa_user_id in LIOPA_USER_IDS])]

    rank_weights = [0.6, 0.3, 0.1]
    all_diffs = []
    print('Model ID | Improvement | Average Diff | Std Diff')
    for model_id in df['Model ID'].unique():
        sub_df = df[(df['Model ID'] == model_id)
                    & (df['User ID'].isin(all_user_ids))]
        diffs = []
        num_samples = len(sub_df)
        for index, row in sub_df.iterrows():
            previous_default = row['Previous Default Accs']
            new_default = row['New Default Accs']

            previous_default_weighted_sum = sum(
                [accuracy * rank
                 for accuracy, rank in zip(previous_default, rank_weights)]
            )
            new_default_weighted_sum = sum(
                [accuracy * rank
                 for accuracy, rank in zip(new_default, rank_weights)]
            )

            diff = new_default_weighted_sum - previous_default_weighted_sum
            diffs.append(diff)

        diffs = np.asarray(diffs)
        improvement_rate = len(np.where(diffs > 0)[0]) / num_samples
        average_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        print(model_id,
              improvement_rate,
              average_diff,
              std_diff)
        all_diffs.append(diffs)

    # box plot
    plt.boxplot(all_diffs)
    plt.ylim((-50, 100))
    plt.ylabel('Weighted Sum Improvement')
    plt.xlabel('Model ID')
    plt.title(title)
    plt.show()

    # show results for specific user ids
    x = np.array([1, 2, 3])
    width = 0.1
    colors = ['b', 'g', 'r', 'y', 'grey', 'black', 'orange']
    for user_id in all_user_ids:
        start = -0.1
        sub_df = df[df['User ID'] == user_id].reset_index()
        for index, row in sub_df.iterrows():
            previous_default = row['Previous Default Accs']
            new_default = row['New Default Accs']

            rank_diffs = [0, 0, 0]
            for i in range(len(rank_diffs)):
                rank_diffs[i] = new_default[i] - previous_default[i]

            plt.bar(x+start, rank_diffs, width=width,
                    color=colors[index-1], label=f'Model {index+1}')
            start += width
        plt.title(f'User {user_id} - Original Default: '
                  f'{[round(acc) for acc in previous_default]}')
        plt.legend()
        plt.xticks(x, x)
        plt.ylabel('Accuracy Improvement %')
        plt.xlabel('Ranks')
        plt.ylim((-100, 100))
        plt.axhline(y=0, color='black')
        plt.show()

    total_samples = df[df['Model ID'] == 1]['Num Samples'].sum()
    print('Total Samples', total_samples)

    # default vs model accuracies full comparison
    if args.show_accuracy_comparison:
        for user_id in all_user_ids:
            sub_df = df[df['User ID'] == user_id].reset_index()

            default_accuracies = sub_df.iloc[0]['Previous Default Accs']
            num_samples = sub_df.iloc[0]['Num Samples']
            plt.plot(x, default_accuracies, color=colors[-1], label=f'Default',
                     marker='o')

            for index, row in sub_df.iterrows():
                model_accuracies = row['New Default Accs']

                plt.plot(x, model_accuracies, color=colors[index],
                         label=f'Model {index+1}', marker='o')

            plt.title(f'User {user_id}, Num Samples {num_samples}')
            plt.ylabel('Accuracy %')
            plt.xlabel('Ranks')
            plt.xticks(x, x)
            plt.ylim((0, 101))
            plt.legend()
            plt.show()


def find_default_models_individually(args):
    """Find default models for an specific person"""
    pass


def test_default_models_2(args):
    models = read_pickle_file(args.models_path)

    test_templates = [create_template(video_path)
                      for video_path in args.video_paths]
    test_templates = [(v, t) for v, t in zip(args.video_paths, test_templates)
                      if t is not None]

    for counter, selected_sessions in models:
        print(f'\nModel {counter}')

        model_templates = sessions_to_templates(selected_sessions)

        for v, t in test_templates:
            predictions = transcribe(model_templates, t.blob)
            print(os.path.basename(v), predictions)


def main(args):
    f = {
        'analyse_users_vs_default': analyse_users_vs_default,
        'find_default_models': find_default_models,
        'test_default_models': test_default_models,
        'analyse_default_models': analyse_default_models,
        'find_default_models_individually': find_default_models_individually,
        'test_default_models_2': test_default_models_2
    }
    f[args.run_type](args)


if __name__ == '__main__':
    def lst_type(s):
        return s.split(',')

    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('analyse_users_vs_default')
    parser_1.add_argument('groundtruth_directory')
    parser_1.add_argument('--user_limit', type=int, default=5)

    parser_2 = sub_parsers.add_parser('find_default_models')
    parser_2.add_argument('groundtruth_directory')
    parser_2.add_argument('dataset_directories', type=lst_type)
    parser_2.add_argument('training_users', type=lst_type)
    parser_2.add_argument('--session_split', type=float, default=0.7)
    parser_2.add_argument('--fast', action='store_true')
    parser_2.add_argument('--max_num_sessions', type=int, default=20)
    parser_2.add_argument('--save_path', default='updated_default_models.pkl')

    parser_3 = sub_parsers.add_parser('test_default_models')
    parser_3.add_argument('groundtruth_directory')
    parser_3.add_argument('--dataset_directories', type=lst_type, default=[])
    parser_3.add_argument('--models_path',
                          default='updated_default_models.pkl')
    parser_3.add_argument('--min_num_templates', type=int, default=1)
    parser_3.add_argument('--output_path_1', default='update_default_model_test_default_models_accuracies.csv')
    parser_3.add_argument('--output_path_2', default='update_default_model_test_default_models_predictions.csv')
    parser_3.add_argument('--include_nota', action='store_true')
    parser_3.add_argument('--save', action='store_true')
    parser_3.add_argument('--redo', action='store_true')

    parser_4 = sub_parsers.add_parser('analyse_default_models')
    parser_4.add_argument('results_path')
    parser_4.add_argument('--pava_only', action='store_true')
    parser_4.add_argument('--liopa_only', action='store_true')
    parser_4.add_argument('--exclude_liopa_pava_users', action='store_true')
    parser_4.add_argument('--specific_user_ids', type=lst_type, default=[])
    parser_4.add_argument('--show_accuracy_comparison', action='store_true')

    parser_5 = sub_parsers.add_parser('test_default_models_2')
    parser_5.add_argument('models_path')
    parser_5.add_argument('video_paths', type=lst_type)

    main(parser.parse_args())
