"""
Expects directory with sub-folders named by user ids containing videos
in the name format XX_YY_ZZ.mp4 where
XX=user_id, YY=phrase_id, ZZ=session_id

One idea is to use the session selection algorithm but it would take too
long e.g. 15 users with 20 sessions containing 70 phrases = 300 sessions

Second idea similar to first time finding default model:
Find worst performing users - this is the benchmark
Find combination of user sessions that get highest in this benchmark

PREVIOUS WORK:
Method 1: by_phrases_2() in app/main/research/find_best.py
This function tested every combination of liopa users vs the patients and
found that Adrian, Alex and Conor (user ids: 2, 3, and 7) were the best users
IIRC the problem with this function was that we were unable to construct full
sessions from the resulting best templates from 2, 3 and 7

Method 2: full_session_templates() in app/main/research/ref_templates_composition.py
This function fixed the problem above by generating the full sessions we
needed for the default set. It only used users 2, 3 and 7 which were found to
be the best from the previous function
"""
import argparse
import ast
import multiprocessing
import os
import random

from main.models import Config
from main.research.research_utils import create_sessions, get_accuracy, \
    sessions_to_templates
from main.utils.io import read_csv_file, read_pickle_file, write_pickle_file
from scripts.session_selection import \
    session_selection_with_cross_validation, \
    session_selection_with_cross_validation_fast
from tqdm import tqdm

DTW_PARAMS = Config().__dict__
HARDEST_USERS_CSV = 'generate_default_model_hardest_users.csv'
FIND_DEFAULT_CSV = 'find_default_model.csv'
VIDEO_REGEX = r'(.+)_P0*(\d+)_S0*(\d+).mp4'

# globals
ref_templates = None
test_templates = None


def save_user_sessions(args):
    sub_directories = os.listdir(args.data_directory)

    for sub_directory in sub_directories:
        print('Creating sessions for:', sub_directory)
        user_data_directory = os.path.join(args.data_directory,
                                           sub_directory)
        create_sessions(user_data_directory, VIDEO_REGEX, save=True)


def find_hardest_test_users_process_fold(process_id, data_directory,
                                         ref_users):
    all_accuracies = []
    for ref_user in ref_users:
        ref_user_sessions = read_pickle_file(
            os.path.join(data_directory, ref_user, 'sessions.pkl')
        )
        ref_templates = sessions_to_templates(ref_user_sessions)

        print(f'Process {process_id}, against ref user {ref_user}')
        accuracies = get_accuracy(ref_templates, test_templates)[0]
        all_accuracies.append((ref_user, accuracies))

    return all_accuracies


def one_vs_one(args):
    """Every user against every other user"""
    user_ids = os.listdir(args.data_directory)
    print('All users:', user_ids)

    # don't process users already done
    already_processed_csv = read_csv_file(
        HARDEST_USERS_CSV,
        ['Test User', 'Ref User', 'Accuracies'],
        r'(.+),(.+),(\[.+\])',
        lambda l: [l[0], l[1], ast.literal_eval(l[2])]
    )
    already_processed_users = list(already_processed_csv['Test User'].unique())
    print('Already processed:', already_processed_users)

    for i in range(len(user_ids)):
        test_user = user_ids[i]
        if test_user in already_processed_users:
            continue
        ref_users = user_ids[:i] + user_ids[i+1:]
        assert len(ref_users) + 1 == len(user_ids)
        print(f'Testing {test_user}...')

        test_user_sessions = read_pickle_file(
            os.path.join(args.data_directory, test_user, 'sessions.pkl')
        )
        global test_templates
        test_templates = sessions_to_templates(test_user_sessions)

        # split ref users between processes
        process_tasks = [
            [
                i+1,
                args.data_directory,
                ref_users[i*args.num_users_per_process:(i*args.num_users_per_process)+args.num_users_per_process],
            ]
            for i in range(args.num_processes)
        ]

        with multiprocessing.Pool(args.num_processes) as pool:
            results = pool.starmap(find_hardest_test_users_process_fold,
                                   process_tasks)

            for all_accuracies in results:
                for ref_user, accuracies in all_accuracies:
                    with open(HARDEST_USERS_CSV, 'a') as f:
                        f.write(f'{test_user},{ref_user},{accuracies}\n')


def analyse_one_vs_one(args):
    """
    Hardest test users = benchmark
    Then find best ref users that maximises this benchmark
    """
    df = read_csv_file(
        HARDEST_USERS_CSV,
        ['Test User', 'Ref User', 'Accuracies'],
        r'(.+),(.+),(\[.+\])',
        lambda l: [l[0], l[1], ast.literal_eval(l[2])]
    )

    print('User Results:')

    by_user = f'{args.by_users.capitalize()} User'

    user_performances = []
    rank_weights = [0.6, 0.3, 0.1]  # more weight given to first ranks
    for user_id in df[by_user].unique():
        if args.excluded_users and user_id in args.excluded_users:
            continue
        sub_df = df[df[by_user] == user_id]
        average_ranks = [0, 0, 0]
        for index, row in sub_df.iterrows():
            accuracies = row['Accuracies']
            for i, rank_accuracy in enumerate(accuracies):
                average_ranks[i] += rank_accuracy
        for i in range(len(average_ranks)):
            average_ranks[i] /= len(sub_df)

        average_average_ranks = sum(average_ranks) / len(average_ranks)
        average_weighted_ranks = sum([accuracy * weight for accuracy, weight
                                      in zip(average_ranks, rank_weights)])
        print(user_id, average_ranks, average_average_ranks,
              average_weighted_ranks)

        # average ranks and weighted ranks
        user_performances.append([user_id,
                                  average_average_ranks,
                                  average_weighted_ranks])

    sort_indexes = {
        'average_ranks': 1,
        'weighted_ranks': 2
    }

    print(f'Sorting accuracy by {args.method}: ', end='')
    sort_index = sort_indexes[args.method]
    print([(item[0], item[sort_index])
           for item in sorted(user_performances, key=lambda x: x[sort_index],
                              reverse=args.descending)[:args.num_to_extract]])


def get_random_user_sessions(num_sessions, users, data_directory):
    random_sessions = []
    session_keys = []
    while len(random_sessions) < num_sessions:
        random_user = random.choice(users)
        user_sessions = create_sessions(
            os.path.join(data_directory, random_user),
            VIDEO_REGEX
        )
        random_session = random.choice(user_sessions)
        key = f'{random_user}_{random_session[0]}'
        if key in session_keys:
            continue
        random_sessions.append(random_session)
        session_keys.append(key)

    session_keys = frozenset(session_keys)
    assert len(random_sessions) == len(session_keys) == num_sessions

    return random_sessions, session_keys


def find_default_model_process_fold(start_index, end_index):
    accuracies = get_accuracy(ref_templates,
                              tqdm(test_templates[start_index:end_index]))[0]
    return accuracies


def get_user_sessions(data_directory, users):
    sessions = []
    for user in users:
        sessions.extend(
            create_sessions(os.path.join(data_directory, user), VIDEO_REGEX),
        )

    return sessions


def find_default_model_1(args):
    """
    find best combination of ref user sessions that maximises accuracy on the
    benchmark (hardest) test users AKA find the default model
    """
    # depending on the number of users and number of sessions per user,
    # I don't think the SSA can possibly run
    # e.g. 10 SO users, 20 sessions each = 200 sessions in memory
    # would also take a long time to run

    ref_users = args.ref_users
    test_users = args.test_users

    global test_templates
    test_templates = sessions_to_templates(
        get_user_sessions(args.data_directory, test_users)
    )

    # randomly picking ref sessions to test
    while True:
        num_sessions = random.randint(args.least_num_sessions,
                                      args.max_default_sessions)

        ref_sessions, ref_key = get_random_user_sessions(num_sessions,
                                                         ref_users,
                                                         args.data_directory)
        ref_key_hash = str(hash(ref_key))

        # check for duplicates
        if os.path.exists(FIND_DEFAULT_CSV):
            df = read_csv_file(FIND_DEFAULT_CSV,
                               ['Ref Users', 'Test Users', 'Ref Hash',
                                'Ref Sessions', 'Accuracies'],
                               r'(\[.+\]),(\[.+\]),(.+),(\[.+\]),(\[.+\])',
                               lambda l: [
                                   ast.literal_eval(l[0]),
                                   ast.literal_eval(l[1]),
                                   l[2],
                                   ast.literal_eval(l[3]),
                                   ast.literal_eval(l[4])
                               ])
            if df['Ref Hash'].str.contains(ref_key_hash).any():
                continue

        global ref_templates
        ref_templates = sessions_to_templates(ref_sessions)
        del ref_sessions

        print('Testing using ref sessions:', num_sessions, list(ref_key),
              len(ref_templates))

        # split test templates between processes
        templates_per_fold = len(test_templates) // args.num_processes
        process_tasks = []
        for i in range(args.num_processes):
            start = i * templates_per_fold
            end = start + templates_per_fold
            if i == args.num_processes - 1:
                end = len(test_templates)
            process_tasks.append([start, end])

        all_accuracies = []
        with multiprocessing.Pool(args.num_processes) as pool:
            results = pool.starmap(find_default_model_process_fold,
                                   process_tasks)
            for accuracies in results:
                all_accuracies.append(accuracies)

        # tallying the results from the multiprocessing
        ranks_tallies = [0, 0, 0]
        for i in range(args.num_processes):
            accuracies = all_accuracies[i]
            for j in range(len(accuracies)):
                # tallying how many we get correct
                rank_tally = (accuracies[j] / 100) * templates_per_fold
                ranks_tallies[j] += rank_tally
        rank_accuracies = [(t * 100) / len(test_templates)
                           for t in ranks_tallies]

        # save results
        with open(FIND_DEFAULT_CSV, 'a') as f:
            f.write(f'{ref_users},'
                    f'{test_users},'
                    f'{ref_key_hash},'
                    f'{list(ref_key)},'
                    f'{rank_accuracies}\n')


def find_default_model_2(args):
    """
    Run the session selection algorithm to find the default model

    E.g. Mandarin
    Ref users were top 3 ref users in the one vs one experiment
    Test users were the worst 5 test users in the one vs one

    The sessions are selected from the ref users
    The training templates are from the test users
    """
    ref_users = args.ref_users
    test_users = args.test_users

    if args.fast:
        selected_ids = session_selection_with_cross_validation_fast(
            _sessions=get_user_sessions(args.data_directory, ref_users),
            _training_templates=sessions_to_templates(
                get_user_sessions(args.data_directory, test_users)
            ),
            max_num_sessions=args.max_num_sessions
        )
    else:
        selected_ids = session_selection_with_cross_validation(
            _sessions=get_user_sessions(args.data_directory, ref_users),
            _training_templates=sessions_to_templates(
                get_user_sessions(args.data_directory, test_users)
            ),
            initial_max=args.initial_max
        )

    print(selected_ids)

    # save model to pickle file
    selected_sessions = []
    for selected_id in selected_ids:
        user_id, session_id = selected_id.split('-')
        user_sessions = create_sessions(
            os.path.join(args.data_directory, user_id),
            VIDEO_REGEX
        )
        user_session_ids = [s[0] for s in user_sessions]
        selected_session_index = user_session_ids.index(selected_id)
        selected_sessions.append(user_sessions[selected_session_index])
    write_pickle_file(selected_sessions, args.save_path)


def main(args):
    f = {
        'save_user_sessions': save_user_sessions,
        'one_vs_one': one_vs_one,
        'analyse_one_vs_one': analyse_one_vs_one,
        'find_default_model_1': find_default_model_1,
        'find_default_model_2': find_default_model_2
    }
    if args.run_type not in f:
        print('Choose from:', list(f.keys()))
    else:
        f[args.run_type](args)


if __name__ == '__main__':
    list_type = lambda s: s.split(',')

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('save_user_sessions')
    parser_1.add_argument('data_directory')

    parser_2 = sub_parsers.add_parser('one_vs_one')
    parser_2.add_argument('data_directory')
    parser_2.add_argument('--num_processes', type=int, default=5)
    parser_2.add_argument('--num_users_per_process', type=int, default=3)

    parser_3 = sub_parsers.add_parser('analyse_one_vs_one')
    parser_3.add_argument('--num_to_extract', type=int, default=5)
    parser_3.add_argument('--by_users', choices=['test', 'ref'],
                          default='test')
    parser_3.add_argument('--method', choices=['average_ranks',
                                               'weighted_ranks'],
                          default='average_ranks')
    parser_3.add_argument('--descending', action='store_true')
    parser_3.add_argument('--excluded_users', type=list_type)

    parser_4 = sub_parsers.add_parser('find_default_model_1')
    parser_4.add_argument('data_directory')
    parser_4.add_argument('ref_users', type=list_type)
    parser_4.add_argument('test_users', type=list_type)
    parser_4.add_argument('max_default_sessions', type=int)
    parser_4.add_argument('--num_processes', type=int, default=5)
    parser_4.add_argument('--least_num_sessions', type=int, default=5)

    parser_5 = sub_parsers.add_parser('find_default_model_2')
    parser_5.add_argument('data_directory')
    parser_5.add_argument('ref_users', type=list_type)
    parser_5.add_argument('test_users', type=list_type)
    parser_5.add_argument('--initial_max', type=int, default=10)
    parser_5.add_argument('--fast', action='store_true')
    parser_5.add_argument('--max_num_sessions', type=int, default=10)
    parser_5.add_argument('--save_path', default='generated_default_model.pkl')

    main(parser.parse_args())
