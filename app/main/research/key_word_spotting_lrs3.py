import argparse
import ast
import multiprocessing
import os
import pprint
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from main.models import Config
from main.research.key_word_spotting import get_dtw_distance, clean_text, create_template, extract_search_template, find_ref_and_search_video_paths_2, find_search_area, get_duration, get_pose_estimation, kws, syllable_count
from main.utils.dtw import DTW
from main.utils.io import read_pickle_file, write_pickle_file

MIN_NUM_SYLLABLES = 4
MAX_DURATION = 10
NUM_SEARCH_TEMPLATES = 5
MIN_NUM_USERS_TO_KEYWORDS = NUM_SEARCH_TEMPLATES + 1
MAX_NUM_USERS_TO_KEYWORDS = 10
LRS3_LIOPA_REGEX = r'0*(\d+)_LDL30*(\d+)_S0*(\d+)'

random.seed(2021)

"""
Testing using the LRS3 Test and LRS3 Liopa Test Sets
NOTE: NO LRS3 test users are in the LRS3 training set
NOTE: Need to update the LRS3 test set (like it says on website)

Testing is always going to be INDEPENDENT so don't need to include dependent samples in the training
...So for constructing the training dataset, I need to find independent in-phrase and not in-phrase samples
"""


def extract_key_words(phrase):
    words = phrase.split(' ')
    key_words = []
    for i in range(len(words)):
        j = i + 1
        while True:
            sub_words = words[i:j]
            sub_words_str = ' '.join(sub_words)
            num_syllables = syllable_count(sub_words_str)
            if num_syllables >= MIN_NUM_SYLLABLES:
                key_words.append(sub_words_str)
                break
            else:
                j += 1
                if j > len(words):
                    break

    return key_words


def get_sorted_list(path, g):
    return sorted(list(path.glob(g)))


def get_user_video_paths(user_directories):
    user_video_paths = {}
    for user_directory in tqdm(user_directories):
        user_video_paths[user_directory.name] = {}
        video_paths, transcripts = get_sorted_list(user_directory, '*.mp4'), get_sorted_list(user_directory, '*.txt')
        assert len(video_paths) == len(transcripts)
        if len(video_paths) < 5:
            continue
        for video_path, transcript in zip(video_paths, transcripts):
            video_duration = get_duration(video_path)
            if video_duration > MAX_DURATION:
                continue

            with open(transcript, 'r') as f:
                phrase = clean_text(f.readline().strip().replace('Text:', '').strip())
                key_words = extract_key_words(phrase)
                for key_word in key_words:
                    video_samples = user_video_paths[user_directory.name].get(key_word, [])
                    video_samples.append([phrase, str(video_path)])
                    user_video_paths[user_directory.name][key_word] = video_samples

    return user_video_paths


def extract_user_kws_paths_lrs3_training(**kwargs):
    dataset_path = Path(kwargs['dataset_path'])
    output_dir = kwargs['output_dir']
    redo = kwargs['redo']

    user_directories = list(set(dataset_path.glob('*')) - set(dataset_path.glob('removed*')))  # exclude removed users

    # construct user video paths i.e. {'DB': {'hello': [phrase, video path]}}
    user_kw_paths_path = Path(os.path.join(output_dir, 'user_video_paths.pkl'))
    if user_kw_paths_path.exists() and not redo:
        user_kw_paths = read_pickle_file(user_kw_paths_path)
    else:
        num_processes = 6
        tasks = []
        num_users_per_process = len(user_directories) // num_processes
        for i in range(num_processes):
            start = i * num_users_per_process
            end = start + num_users_per_process
            tasks.append([user_directories[start:end]])
        user_kw_paths = {}
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(get_user_video_paths, tasks)
            for result in results:
                user_kw_paths.update(result)
        write_pickle_file(user_kw_paths, user_kw_paths_path)

    return user_kw_paths


def extract_user_kws_paths_lrs3_liopa(**kwargs):
    dataset_path = kwargs['dataset_path']
    phrases_path = kwargs['phrases_path']

    with open(phrases_path, 'r') as f:
        phrases = f.read().splitlines()

    # extract key-words for each phrase
    phrase_kws_d = {}
    for phrase in phrases:
        key_words = extract_key_words(phrase)
        phrase_kws_d[phrase] = key_words

    # extract samples for each key-word by user id
    user_kw_paths_d = {}
    for video_path in Path(dataset_path).glob('*.mp4'):
        video_name = video_path.name
        user_id, phrase_id, session_id = re.match(LRS3_LIOPA_REGEX, video_name).groups()
        phrase = phrases[int(phrase_id)-1]
        key_word_paths_d = user_kw_paths_d.get(user_id, {})
        for key_word in phrase_kws_d[phrase]:
            video_samples = key_word_paths_d.get(key_word, [])
            video_samples.append([phrase, str(video_path)])
            key_word_paths_d[key_word] = video_samples
        user_kw_paths_d[user_id] = key_word_paths_d

    return user_kw_paths_d


def create_dataset(args):
    pprint.pprint(args.__dict__)

    if args.liopa_test_set:
        user_kw_paths = extract_user_kws_paths_lrs3_liopa(**args.__dict__)
    else:
        user_kw_paths = extract_user_kws_paths_lrs3_training(**args.__dict__)

    user_ids = list(user_kw_paths.keys())

    # shuffle user kw paths
    for user_id, key_word_paths in user_kw_paths.items():
        for key_word, video_samples in key_word_paths.items():
            random.shuffle(video_samples)

    # find users that say the same phrase
    key_words_to_users = {}
    for user_id, key_word_paths in user_kw_paths.items():
        for key_word in key_word_paths.keys():
            users = key_words_to_users.get(key_word, [])
            users.append(user_id)
            key_words_to_users[key_word] = users
    key_words = []
    for key_word, users in key_words_to_users.items():
        assert len(set(users)) == len(users)  # just checking there isn't duplicate user ids per keyword
        if len(users) > 1:
            key_words.append(key_word)
    assert len(set(key_words)) == len(key_words)
    print(f'{len(key_words)} instances of different users saying the same keyword')

    # # show distribution of key-word syllables
    # kws_num_syllables = []
    # for key_word in key_words:
    #     num_syllables = syllable_count(key_word)
    #     kws_num_syllables.append(num_syllables)
    # plt.hist(kws_num_syllables)
    # plt.show()

    # show distribution of number of users to same keyword counts
    d_num_users_keyword_counts = {}
    l_num_users = []
    for key_word, users in key_words_to_users.items():
        num_users = len(users)
        if MIN_NUM_USERS_TO_KEYWORDS <= num_users <= MAX_NUM_USERS_TO_KEYWORDS:
            d_num_users_keyword_counts[num_users] = d_num_users_keyword_counts.get(num_users, 0) + 1
            l_num_users.append(num_users)
    print('Num Users To Key-Word Counts:', d_num_users_keyword_counts)

    # selecting our key-words based on how many unique users say them (for multiple search templates)
    key_words = [key_word for key_word, users in key_words_to_users.items()
                 if MIN_NUM_USERS_TO_KEYWORDS <= len(users) <= MAX_NUM_USERS_TO_KEYWORDS]
    random.shuffle(key_words)
    assert len(key_words) == sum([num_key_words for num_key_words in d_num_users_keyword_counts.values()])

    # how many unique users are involved in the keywords
    unique_key_word_users = set()
    for key_word in key_words:
        unique_key_word_users = unique_key_word_users.union(set(key_words_to_users[key_word]))
    unique_user_percentage = float(len(unique_key_word_users)) / float(len(user_ids))
    print(f'Using {round(unique_user_percentage, 2)} of the total users')

    columns = [
        'Ref User', 'Search User', 'In Phrase',
        'Ref Phrase', 'Ref Video Path', 'Num Ref Frames', 'Ref FPS', 'Ref Duration', 'Ref Pose Direction',
        'Search Phrase', 'Search Term', 'Search Video Path', 'Num Search Frames', 'Search FPS', 'Search Duration', 'Search Pose Direction', 'Original Search Template Size', 'Clipped Search Template Size', 'Window Size',
        'FA Start', 'FA End', 'FA Likelihood',
        'Best Padding', 'Tried Paddings'
    ]

    df_path = Path(os.path.join(args.output_dir, args.csv_name))
    if df_path.exists() and not args.redo:
        df = pd.read_csv(df_path)
    else:
        df = pd.DataFrame(columns=columns)

    # not doing dependent because it would be hard to find examples where the same user
    # is saying the same thing
    dependency = 'independent'

    # start the process from the last search key-word from the previous run if any
    if len(df) > 0:
        last_row = df.iloc[[-1]]
        last_search_key_word = last_row['Search Term'].values[0]
        key_words = key_words[key_words.index(last_search_key_word):]

    for search_key_word in tqdm(key_words):

        key_word_users = key_words_to_users[search_key_word]
        random.shuffle(key_word_users)  # don't want the same user being selected as a ref user all the time
        ref_user_id, search_user_ids = key_word_users[0], key_word_users[1:]
        user_mappings = {ref_user_id: search_user_ids}

        for in_phrase in [True, False]:

            # check if keyword done or not
            if ((df['Search Term'] == search_key_word) & (df['In Phrase'] == in_phrase)).any():
                continue

            results = find_ref_and_search_video_paths_2(
                user_video_paths=user_kw_paths,
                user_mappings=user_mappings,
                ref_user_id=ref_user_id,
                search_word=search_key_word,
                dependency=dependency,
                in_phrase=in_phrase,
                multiple_search=args.use_multiple_search_templates,
                debug=args.debug
            )
            if results is None:
                continue
            search_criteria, ref_phrase, ref_video_path = results

            # grab majority pose in ref video
            # TODO: For in-phrase, check search area of forced alignment of key word instead of majority pose
            ref_pose_estimation = get_pose_estimation(ref_video_path)
            if ref_pose_estimation is None:
                continue
            ref_pose_direction = ref_pose_estimation['direction']
            if args.pose_direction and args.pose_direction != ref_pose_direction:
                continue

            lowest_distance = np.inf
            best_index = None
            all_results = []
            for i, (search_user_id, search_phrase, search_video_path) in enumerate(search_criteria):
                results = kws(
                    ref_video_path=ref_video_path,
                    search_video_path=search_video_path,
                    search_transcript=search_phrase,
                    search_words=search_key_word,
                    preprocess=args.preprocess,
                    core_search_template=args.core_search_template,
                    use_pad_finding=args.use_pad_finding,
                    pose_direction=args.pose_direction,
                    debug=args.debug
                )
                all_results.append(results)
                if results is None:
                    continue

                distances = results['Tried Paddings'][results['Best Padding']]['Distances']
                min_distance = min(distances)
                if min_distance < lowest_distance:
                    lowest_distance = min_distance
                    best_index = i

            if best_index is None:
                continue

            results = all_results[best_index]
            search_user_id, search_phrase, search_video_path = search_criteria[best_index]

            data = [[
                ref_user_id, search_user_id, in_phrase,
                ref_phrase, ref_video_path, results['Num Ref Frames'], results['Ref FPS'], results['Ref Duration'], ref_pose_direction,
                search_phrase, search_key_word, search_video_path, results['Num Search Frames'], results['Search FPS'], results['Search Duration'], results['Search Pose Direction'], results['Original Search Template Size'], results['Clipped Search Template Size'], results['Window Size'],
                results['FA Start'], results['FA End'], results['FA Likelihood'],
                results['Best Padding'], results['Tried Paddings']
            ]]

            new_rows = pd.DataFrame(data=data, columns=columns)
            df = pd.concat([df, new_rows])
            df.to_csv(df_path, index=False)


def test(args):
    with open(args.phrases_path, 'r') as f:
        phrases = f.read().splitlines()

    # extract key-words for each phrase
    phrase_kws_d = {}
    for phrase in phrases:
        key_words = extract_key_words(phrase)
        phrase_kws_d[phrase] = key_words

    # extract samples for each key-word by user id
    user_kw_paths_d = {}
    for video_path in Path(args.dataset_path).glob('*.mp4'):
        video_name = video_path.name
        user_id, phrase_id, session_id = re.match(LRS3_LIOPA_REGEX, video_name).groups()
        phrase = phrases[int(phrase_id)-1]
        key_word_paths_d = user_kw_paths_d.get(user_id, {})
        for key_word in phrase_kws_d[phrase]:
            video_samples = key_word_paths_d.get(key_word, [])
            video_samples.append([phrase, str(video_path)])
            key_word_paths_d[key_word] = video_samples
        user_kw_paths_d[user_id] = key_word_paths_d

    # find users that say the same key-words
    kws_to_users_d = {}
    for user_id, key_word_paths_d in user_kw_paths_d.items():
        for key_word in key_word_paths_d.keys():
            users = kws_to_users_d.get(key_word, [])
            users.append(user_id)
            kws_to_users_d[key_word] = users
    kws_to_users_d = {
        key_word: users
        for key_word, users in kws_to_users_d.items()
        if MIN_NUM_USERS_TO_KEYWORDS <= len(users) <= MAX_NUM_USERS_TO_KEYWORDS
    }
    key_words = list(kws_to_users_d.keys())

    groundtruth, predictions = [], []

    for search_key_word in tqdm(key_words):

        key_word_users = kws_to_users_d[search_key_word]
        random.shuffle(key_word_users)
        ref_user_id, search_user_ids = key_word_users[0], key_word_users[1:]
        user_mappings = {ref_user_id: search_user_ids}

        for in_phrase in [True, False]:

            results = find_ref_and_search_video_paths_2(
                user_video_paths=user_kw_paths_d,
                user_mappings=user_mappings,
                ref_user_id=ref_user_id,
                search_word=search_key_word,
                dependency='independent',
                in_phrase=in_phrase,
                multiple_search=args.use_multiple_search_templates,
                debug=args.debug
            )
            if results is None:
                continue
            search_criteria, ref_phrase, ref_video_path = results

            # grab ref pose direction
            ref_pose_estimation = get_pose_estimation(ref_video_path)
            if ref_pose_estimation is None:
                continue
            ref_pose_direction = ref_pose_estimation['direction']
            if args.pose_direction and args.pose_direction != ref_pose_direction:
                continue

            all_results = []
            all_search_pose_directions = []
            for i, (search_user_id, search_phrase, search_video_path) in enumerate(search_criteria):

                # grab search pose direction
                search_pose_estimation = get_pose_estimation(search_video_path)
                if search_pose_estimation is None:
                    continue
                search_pose_direction = search_pose_estimation['direction']
                if args.pose_direction and args.pose_direction != search_pose_direction:
                    continue

                results = kws(
                    ref_video_path=ref_video_path,
                    search_video_path=search_video_path,
                    search_transcript=search_phrase,
                    search_words=search_key_word,
                    preprocess=args.preprocess,
                    core_search_template=args.core_search_template,
                    use_pad_finding=args.use_pad_finding,
                    debug=args.debug
                )
                if results is None:
                    continue

                all_results.append(results)
                all_search_pose_directions.append(search_pose_direction)

            lowest_distance = np.inf
            best_index = None
            for i, results in enumerate(all_results):
                distances = results['Tried Paddings'][results['Best Padding']]['Distances']
                min_distance = min(distances)
                if min_distance < lowest_distance:
                    lowest_distance = min_distance
                    best_index = i
            if best_index is None:
                continue

            label = 1 if in_phrase else 0
            groundtruth.append(label)
            if lowest_distance <= args.threshold:
                prediction = 1
            else:
                prediction = 0
            predictions.append(prediction)

    cm = confusion_matrix(groundtruth, predictions)
    columns = ['In Phrase', 'Not In Phrase']
    df_cm = pd.DataFrame(cm, index=columns, columns=columns)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)


def examine(args):
    df = pd.read_csv(args.dataset_path)
    df['Tried Paddings'] = df.apply(lambda row: ast.literal_eval(row['Tried Paddings']), axis=1)

    row = df.iloc[args.index-2]

    print(row['Ref Phrase'], row['Ref Video Path'])
    print(row['Search Term'], row['Search Video Path'])

    # show distance graph
    for padding, results in row['Tried Paddings'].items():
        plt.plot(results['Xs'], results['Distances'])
    plt.show()

    dtw = DTW(**Config().__dict__)
    dtw.find_path = True
    ref_signal = create_template(row['Ref Video Path'], preprocess=True).blob
    start_time, end_time, score = find_search_area(row['Search Video Path'], row['Search Phrase'], row['Search Term'])
    test_signal = extract_search_template(row['Search Video Path'], start_time, end_time, preprocess=True)[3]
    path, distance, cost_matrix, path_length = get_dtw_distance(dtw, test_signal.astype(np.float32), ref_signal.astype(np.float32))
    cost_matrix = np.where(cost_matrix > 1000000, 0, cost_matrix)
    sns.heatmap(cost_matrix, cmap='YlGnBu')
    xs, ys = zip(*path)
    plt.plot(ys, xs)
    plt.show()


def main(args):
    f = {
        'create_dataset': create_dataset,
        'test': test,
        'examine': examine,
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('create_dataset')
    parser_1.add_argument('dataset_path')
    parser_1.add_argument('output_dir')
    parser_1.add_argument('csv_name')
    parser_1.add_argument('--preprocess', action='store_true')
    parser_1.add_argument('--pose_direction')
    parser_1.add_argument('--core_search_template', type=float, default=1.0)
    parser_1.add_argument('--use_pad_finding', action='store_true')
    parser_1.add_argument('--use_multiple_search_templates', action='store_true')
    parser_1.add_argument('--debug', action='store_true')
    parser_1.add_argument('--redo', action='store_true')
    parser_1.add_argument('--liopa_test_set', action='store_true')
    parser_1.add_argument('--phrases_path')

    parser_2 = sub_parsers.add_parser('test')
    parser_2.add_argument('dataset_path')
    parser_2.add_argument('phrases_path')
    parser_2.add_argument('threshold', type=float)
    parser_2.add_argument('--pose_direction')

    parser_3 = sub_parsers.add_parser('examine')
    parser_3.add_argument('dataset_path')
    parser_3.add_argument('index', type=int)

    main(parser.parse_args())
