"""
KWS: Key Word/Phrase Spotting

- Keep functions as modular as possible
- Focus on original DTW algorithm
- Use video sliding window approach
- Sliding window conversion to similar no. reference frames
- Padding on the sliding window - should focus in on the best padding (need algorithm for this to find direction)
- Use core x% of search template removes artifacts (in practice, cropping of search template is required)
- Use of multiple search templates for the independent case (just min across all search templates should do)
- KPS: should search templates be the whole phrase or use separate templates for each word and look for
consecutive low distance regions
- GOAL is 0 FN and reduce FP as much as possible
"""
import argparse
import ast
import datetime
import multiprocessing
import os
import pprint
import random
import subprocess
import tempfile
import textwrap
import time
from http import HTTPStatus
from pathlib import Path
from string import punctuation

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
random.seed(2021)

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from fastdtw import fastdtw
from tqdm import tqdm
from scipy import stats
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.spatial.distance import sqeuclidean
from sklearn.metrics import classification_report

from main.models import Config
from main.research.process_videos import get_video_rotation, fix_frame_rotation
from main.research.research_utils import create_template, create_templates
from main.utils.dtw import DTW


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1

    return count


def get_fps(video_path):
    vr = cv2.VideoCapture(video_path)
    fps = vr.get(cv2.CAP_PROP_FPS)
    vr.release()

    return fps


def get_duration(video_path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", video_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)

    return float(result.stdout)


def get_frames(video_path):
    reader = cv2.VideoCapture(video_path)
    rotation = get_video_rotation(video_path)
    frames = []
    while True:
        success, frame = reader.read()
        if not success:
            break
        frame = fix_frame_rotation(frame, rotation)
        frames.append(frame)

    reader.release()

    return frames


def get_num_frames(video_path):
    return len(get_frames(video_path))


def extract_audio_from_video(video_path, audio_path, audio_codec='copy'):
    command = f'ffmpeg -hide_banner -loglevel error -y -i {video_path} -vn -acodec {audio_codec} {audio_path}'

    subprocess.call(command, shell=True, stdout=None)


def precise_slice(video_path, start, end, output_path):
    start = str(datetime.timedelta(seconds=start))
    end = str(datetime.timedelta(seconds=end))

    command = f'ffmpeg -hide_banner -loglevel error -y -i {video_path} -ss {start} -to {end} {output_path}'

    return subprocess.call(command, shell=True, stdout=None)


def clean_text(text):
    return text.lower().strip(punctuation)


def run_forced_alignment(video_path, transcript, debug=False):
    # do forced alignment to on audio/text
    transcript = clean_text(transcript)

    with tempfile.NamedTemporaryFile('w', suffix='.txt') as f1, tempfile.NamedTemporaryFile(suffix='.wav') as f2:
        f1.write(transcript)
        f1.seek(0)

        extract_audio_from_video(video_path, f2.name, audio_codec='pcm_s16le')
        f2.seek(0)

        forced_alignment_response = \
            requests.post('http://127.0.0.1:8082/align/',
                          files={'audio': open(f2.name, 'rb'),
                                 'transcript': open(f1.name, 'rb')})
        if forced_alignment_response.status_code != HTTPStatus.OK:
            return

        forced_alignment = forced_alignment_response.json()['alignment']

    if debug:
        print(transcript)
        print(forced_alignment)
        print(len(transcript.split(' ')), len(forced_alignment))

    # assert len(forced_alignment) == len(transcript.split(' '))

    return forced_alignment


def find_search_area(video_path, transcript, search_words, debug=False):
    # extracts start and end time of search word in transcript
    # if search word appears multiple times in transcript, it will select the first time is appears
    forced_alignment = run_forced_alignment(video_path, transcript, debug=debug)
    if forced_alignment is None:
        return

    search_words = clean_text(search_words).split(' ')
    if debug:
        print(search_words)

    for i in range((len(forced_alignment) - len(search_words)) + 1):
        if all([
            forced_alignment[i+j][0] == search_words[j].upper()
            for j in range(len(search_words))
        ]):
            start_time = forced_alignment[i][1]
            end_time = forced_alignment[i+(len(search_words)-1)][2]
            score = np.mean([
                forced_alignment[i+j][3]
                for j in range(len(search_words))
            ])

            return start_time, end_time, score

    return


def extract_search_template(video_path, start_time, end_time, get_ark=True, preprocess=False, pose_direction=None,
                            pose_port='8085', debug=False):
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        slice_result = precise_slice(video_path, start_time, end_time, f.name)
        if slice_result != 0:
            return
        f.seek(0)

        # grab search pose direction
        search_pose_estimation = get_pose_estimation(f.name, port=pose_port)
        if search_pose_estimation is None:
            return
        search_pose_direction = search_pose_estimation['direction']
        if pose_direction and pose_direction != search_pose_direction:
            return

        search_window_size = get_num_frames(f.name)
        search_fps = get_fps(f.name)
        search_duration = get_duration(f.name)

        if debug:
            print('Search Window Size:', search_window_size)
            print('Search FPS:', search_fps)
            print('Search Duration:', search_duration)
            print('Search Pose:', search_pose_direction)
            print()

        if get_ark:
            search_template = create_template(f.name, preprocess=preprocess, debug=debug)
            if not search_template:
                return
            search_template = search_template.blob.astype(np.float32)
        else:
            search_template = None

        return search_window_size, search_fps, search_duration, search_template, search_pose_direction


def create_window_template(window_frames, fps, preprocess=False, extract_pose_direction=False, debug=False):
    height, width = window_frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        video_writer = cv2.VideoWriter(f.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in window_frames:
            video_writer.write(frame)
        video_writer.release()

        # grab window pose direction
        if extract_pose_direction:
            window_pose_estimation = get_pose_estimation(f.name)
            if window_pose_estimation is None:
                return
            window_pose_direction = window_pose_estimation['direction']
        else:
            window_pose_direction = None

        window_template = create_template(f.name, preprocess=preprocess, debug=debug)

    return window_template, window_pose_direction


def clip_template(template, core=1.0, debug=False):
    if core == 1.0:
        return template

    # extract core x% of template frames i.e. remove x/2% from start and end
    # more focus given to start of template

    template_length = template.shape[0]
    num_core_frames = template_length * core
    diff = template_length - num_core_frames
    diff_i = int(diff)

    half = diff_i // 2
    if diff.is_integer() and diff_i % 2 == 0:
        from_start = from_end = half
    else:
        from_end = half
        from_start = diff_i - from_end

    template = template[from_start:template_length-from_end, ]

    if debug:
        print('Original Search Template:', template_length)
        print('Start-End:', from_start, from_end)
        print('Core Search Template:', template.shape[0])
        print()

    return template


def get_dtw_distance(dtw, test, ref):
    cost_matrix = dtw._calculate_cost_matrix(test_signal=test, ref_signal=ref)
    path, distance, cost_matrix, path_length = dtw._calculate_path_and_distance(cost_matrix=cost_matrix)

    return path, distance, cost_matrix, path_length


def sliding_window(ref_frames, search_template, window_size, ref_fps, preprocess=False, index_range=None,
                   pose_direction=None, debug=False):
    num_ref_frames = len(ref_frames)
    window_positions = range(num_ref_frames - window_size)

    if index_range:
        assert len(index_range) == len(window_positions), f'{len(index_range)} != {len(window_positions)}'

    dtw = DTW(**Config().__dict__)

    xs, distances, path_lengths, ref_template_sizes, window_pose_directions = [], [], [], [], []
    for i in window_positions:
        ref_window = ref_frames[i:i + window_size]
        ref_template, window_pose_direction = create_window_template(ref_window, ref_fps, preprocess=preprocess,
                                                                     debug=debug)
        if ref_template is None or (pose_direction and pose_direction != window_pose_direction):
            continue
        ref_template = stats.zscore(ref_template.blob.astype(np.float32))

        path, distance, cost_matrix, path_length = get_dtw_distance(dtw, search_template, ref_template)

        if index_range:
            i = index_range[i]
        xs.append(i)
        distances.append(distance)
        path_lengths.append(path_length)
        ref_template_sizes.append(ref_template.shape[0])
        window_pose_directions.append(window_pose_direction)

    return xs, distances, path_lengths, ref_template_sizes, window_pose_directions


def padded_sliding_windows(ref_video_path, search_template, search_duration, preprocess=False, use_pad_finding=False):
    """Pad focusing by running pads from start to end and selecting best based on min distance"""
    ref_frames = get_frames(ref_video_path)
    ref_fps = get_fps(ref_video_path)
    ref_duration = get_duration(ref_video_path)

    search_window_size = int(ref_fps * search_duration)
    tried_paddings = {}
    best_padding = None

    if use_pad_finding:
        # first find initial direction to take in parallel
        initial_paddings = [-3, 0, 3]
        lowest_distance = np.inf
        num_processes = len(initial_paddings)
        with multiprocessing.Pool(processes=num_processes) as pool:
            process_tasks = [[
                ref_frames,
                search_template,
                search_window_size + padding,
                ref_fps,
                preprocess
            ] for padding in initial_paddings]
            results = pool.starmap(sliding_window, process_tasks)
            for padding, (xs, distances, path_lengths, ref_template_sizes) in zip(initial_paddings, results):
                tried_paddings[padding] = None
                if not distances or any([np.isnan(d) for d in distances]):
                    continue
                min_distance = min(distances)
                if min_distance < lowest_distance:
                    lowest_distance = min_distance
                    best_padding = padding
                tried_paddings[padding] = {
                    'Xs': xs,
                    'Distances': distances,
                    'Path Lengths': path_lengths,
                    'Ref Template Sizes': ref_template_sizes
                }

        # if all initial paddings have failed, then return
        if not best_padding:
            return

        while True:
            min_distances = []
            paddings = []
            padding_range = [padding for padding in list(range(best_padding-1, best_padding+2))
                             if padding not in tried_paddings]
            num_processes = len(padding_range)

            if num_processes == 0:
                break

            if num_processes > 1:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    process_tasks = [[
                        ref_frames,
                        search_template,
                        search_window_size + padding,
                        ref_fps,
                        preprocess
                    ] for padding in padding_range]
                    results = pool.starmap(sliding_window, process_tasks)
            else:
                results = [sliding_window(ref_frames, search_template, search_window_size + padding_range[0], ref_fps)]

            for padding, (xs, distances, path_lengths, ref_template_sizes) in zip(padding_range, results):
                tried_paddings[padding] = None
                if not distances or any([np.isnan(d) for d in distances]):
                    continue
                min_distances.append(min(distances))
                paddings.append(padding)
                tried_paddings[padding] = {
                    'Xs': xs,
                    'Distances': distances,
                    'Path Lengths': path_lengths,
                    'Ref Template Sizes': ref_template_sizes
                }

            # if no min distances here, all of padding range failed
            # break with already found best padding
            if not min_distances:
                break

            best_index = min_distances.index(min(min_distances))
            if min_distances[best_index] < lowest_distance:
                lowest_distance = min_distances[best_index]
                best_padding = paddings[best_index]
            else:
                # no improvement in min distance, break with already found best padding
                break
    else:
        best_padding = 0
        xs, distances, path_lengths, ref_template_sizes = \
            sliding_window(ref_frames, search_template, search_window_size, ref_fps, preprocess)
        if not distances or any([np.isnan(d) for d in distances]):
            return
        tried_paddings[best_padding] = {
            'Xs': xs,
            'Distances': distances,
            'Path Lengths': path_lengths,
            'Ref Template Sizes': ref_template_sizes
        }

    return {
        'Best Padding': best_padding,
        'Tried Paddings': tried_paddings,
        'Num Ref Frames': len(ref_frames),
        'Ref FPS': ref_fps,
        'Ref Duration': ref_duration,
        'Window Size': search_window_size
    }


def padded_sliding_windows_2(ref_video_path, search_template, search_window_size, search_duration, preprocess=False,
                             use_pad_finding=False, pose_direction=None, debug=False):
    """This approach focuses more on the min distance of pad 0 rather than running paddings from the start to the end"""
    ref_frames = get_frames(ref_video_path)
    ref_fps = get_fps(ref_video_path)
    ref_duration = get_duration(ref_video_path)

    search_window_size = int(ref_fps * search_duration)  # find ref search window size
    tried_paddings = {}

    xs, distances, path_lengths, ref_template_sizes, window_pose_directions = \
        sliding_window(ref_frames, search_template, search_window_size, ref_fps, preprocess=preprocess,
                       pose_direction=pose_direction, debug=debug)
    if debug:
        print(distances)
        print(len(xs), len(distances), len(ref_frames) - search_window_size)
    if not distances or any([np.isnan(d) for d in distances]):
        return
    if len(xs) != (len(ref_frames) - search_window_size):  # ensure full results for 0 padding - affects the rest
        return
    tried_paddings[0] = {
        'Xs': xs,
        'Distances': distances,
        'Path Lengths': path_lengths,
        'Ref Template Sizes': ref_template_sizes,
        'Ref Pose Directions': window_pose_directions
    }

    if use_pad_finding:
        min_distance_index = distances.index(min(distances))
        from_index, to_index = min_distance_index - 5, min_distance_index + 5
        if from_index < 0:
            from_index = 0
        if to_index > len(xs) - 1:
            to_index = len(xs) - 1
        from_index, to_index = xs[from_index], xs[to_index]

        # min_distance_index = distances.index(min(distances))
        # # print(min_distance_index, xs[min_distance_index])
        # from_index, to_index = min_distance_index - 5, min_distance_index + 5
        # if from_index < 0:
        #     from_index = 0
        # if to_index > len(ref_frames) - 1:
        #     to_index = len(ref_frames) - 1

        best_padding_direction = None
        lowest_distance = min(distances)
        for padding in [-1, 1]:
            # print(to_index, search_window_size, padding)

            # upper_bound_frame = to_index + search_window_size + padding
            # if upper_bound_frame > len(ref_frames) - 1:
            #     continue
            # sliding_window_area = ref_frames[from_index:upper_bound_frame]
            # index_range = list(range(from_index, to_index))

            upper_bound_frame = to_index + search_window_size + padding
            sliding_window_area = ref_frames[from_index:upper_bound_frame]
            index_range = list(range(from_index, to_index))
            if upper_bound_frame > len(ref_frames):
                index_range = list(range(from_index, (to_index - (upper_bound_frame - len(ref_frames)))))

            # print(padding,
            #       len(ref_frames),
            #       min_distance_index,
            #       from_index, to_index,
            #       search_window_size,
            #       len(sliding_window_area),
            #       len(index_range), '\n')

            xs, distances, path_lengths, ref_template_sizes, window_pose_directions = \
                sliding_window(sliding_window_area, search_template, search_window_size + padding, ref_fps,
                               preprocess=preprocess, index_range=index_range, pose_direction=pose_direction,
                               debug=debug)
            if not distances or any([np.isnan(d) for d in distances]):
                continue
            min_distance = min(distances)
            if min_distance < lowest_distance:
                lowest_distance = min_distance
                best_padding_direction = padding
            tried_paddings[padding] = {
                'Xs': xs,
                'Distances': distances,
                'Path Lengths': path_lengths,
                'Ref Template Sizes': ref_template_sizes,
                'Ref Pose Directions': window_pose_directions
            }

        if best_padding_direction:
            padding = best_padding_direction
            while True:
                padding += best_padding_direction

                # if padding == -6:  # -5 is the min padding that we'll allow (prevents focusing on viseme level)
                #     padding -= best_padding_direction
                #     break

                upper_bound_frame = to_index + search_window_size + padding

                # if upper_bound_frame > len(ref_frames) - 1:
                #     padding -= best_padding_direction
                #     break
                # sliding_window_area = ref_frames[from_index:upper_bound_frame]
                # index_range = list(range(from_index, to_index))

                sliding_window_area = ref_frames[from_index:upper_bound_frame]
                index_range = list(range(from_index, to_index))
                if upper_bound_frame > len(ref_frames):
                    index_range = list(range(from_index, (to_index - (upper_bound_frame - len(ref_frames)))))

                if debug:
                    print(len(sliding_window_area), search_window_size, padding, index_range)

                xs, distances, path_lengths, ref_template_sizes, window_pose_directions = \
                    sliding_window(sliding_window_area, search_template, search_window_size + padding, ref_fps,
                                   preprocess=preprocess, index_range=index_range, pose_direction=pose_direction,
                                   debug=debug)
                if not distances or any([np.isnan(d) for d in distances]):
                    padding -= best_padding_direction
                    break
                min_distance = min(distances)
                if min_distance >= lowest_distance:
                    padding -= best_padding_direction
                    break
                else:
                    lowest_distance = min_distance
                tried_paddings[padding] = {
                    'Xs': xs,
                    'Distances': distances,
                    'Path Lengths': path_lengths,
                    'Ref Template Sizes': ref_template_sizes,
                    'Ref Pose Directions': window_pose_directions
                }
            best_padding = padding
        else:
            best_padding = 0
    else:
        best_padding = 0

    return {
        'Best Padding': best_padding,
        'Tried Paddings': tried_paddings,
        'Num Ref Frames': len(ref_frames),
        'Ref FPS': ref_fps,
        'Ref Duration': ref_duration,
        'Window Size': search_window_size
    }


def kws(ref_video_path, search_video_path, search_transcript, search_words, preprocess=False, core_search_template=1.0,
        use_pad_finding=False, pose_direction=None, debug=False):
    results = find_search_area(
        video_path=search_video_path,
        transcript=search_transcript,
        search_words=search_words,
        debug=debug
    )
    if debug:
        print(results)
    if not results:
        return
    start_time, end_time, score = results

    results = extract_search_template(
        video_path=search_video_path,
        start_time=start_time,
        end_time=end_time,
        preprocess=preprocess,
        pose_direction=pose_direction,
        debug=debug
    )
    if debug:
        print(results)
    if not results:
        return
    search_window_size, search_fps, search_duration, search_template, search_pose_direction = results

    if search_template.shape[0] < 10:
        return

    clipped_search_template = clip_template(
        template=search_template,
        core=core_search_template,
        debug=debug
    )

    results = padded_sliding_windows_2(
        ref_video_path=ref_video_path,
        search_template=clipped_search_template,
        search_window_size=search_window_size,
        search_duration=search_duration,
        preprocess=preprocess,
        use_pad_finding=use_pad_finding,
        pose_direction=None,
        debug=debug
    )
    if debug:
        print(results)
    if results is None:
        return

    results.update({
        'Num Search Frames': search_window_size,
        'Search FPS': search_fps,
        'Search Duration': search_duration,
        'Search Pose Direction': search_pose_direction,
        'Original Search Template Size': search_template.shape[0],
        'Clipped Search Template Size': clipped_search_template.shape[0],
        'FA Start': start_time,
        'FA End': end_time,
        'FA Likelihood': score
    })

    return results


def find_ref_and_search_video_paths(user_video_paths, ref_user_id, search_word, dependency, in_phrase):
    """
    - Dependent in-phrase
    - Dependent not in-phrase
    - Independent in-phrase
    - Independent not in-phrase
    """
    user_ids = sorted(list(user_video_paths.keys()))

    if dependency == 'dependent':
        search_user_id = ref_user_id
    else:
        search_user_id = random.choice([user_id for user_id in user_ids if user_id != ref_user_id])
    search_phrase, search_video_path = random.choice(user_video_paths[search_user_id][search_word])

    while True:
        if in_phrase:
            ref_word = search_word
        else:
            ref_word = random.choice([word for word in user_video_paths[ref_user_id].keys() if word != search_word])
        ref_phrase, ref_video_path = random.choice(user_video_paths[ref_user_id][ref_word])

        if not in_phrase and search_word in ref_phrase.split(' '):
            continue

        if ref_video_path != search_video_path:
            break

    if dependency == 'dependent':
        assert ref_user_id == search_user_id
    else:
        assert ref_user_id != search_user_id

    if in_phrase:
        assert search_word in ref_phrase.split(' ')
    else:
        assert search_word not in ref_phrase.split(' ')

    assert ref_video_path != search_video_path

    return search_user_id, ref_phrase, ref_video_path, search_phrase, search_video_path


def find_ref_and_search_video_paths_2(user_video_paths, user_mappings, ref_user_id, search_word, dependency,
                                      in_phrase, multiple_search=False, debug=False):
    if dependency == 'dependent':
        search_user_ids = [ref_user_id]
    else:
        search_user_ids = user_mappings[ref_user_id]  # multiple search users for multiple search templates
        random.shuffle(search_user_ids)
        if not multiple_search:
            search_user_ids = [search_user_ids[0]]

    search_criteria = []
    for search_user_id in search_user_ids:
        search_phrase, search_video_path = user_video_paths[search_user_id][search_word][0]  # always selects the first, these are shuffled beforehand
        search_criteria.append((search_user_id, search_phrase, search_video_path))

    # the search criteria should only be 1 if dependent or not using multiple search templates in the independent case
    if dependency == 'dependent' or not multiple_search:
        assert len(search_criteria) == 1

    num_attempts = 5
    while True:
        if in_phrase:
            ref_word = search_word
        else:
            ref_word = random.choice([word for word in user_video_paths[ref_user_id].keys() if word != search_word])
        ref_phrase, ref_video_path = random.choice(user_video_paths[ref_user_id][ref_word])

        # ensure if not in phrase, that search word is not in selected ref phrase
        if not in_phrase and search_word in ref_phrase:
            if num_attempts == 0:
                return
            num_attempts -= 1
            continue

        # ensure selected ref video path is not same as any search video path
        if ref_video_path not in [x[-1] for x in search_criteria]:
            break

    if debug:
        print(user_mappings)
        print(search_word)
        print(ref_phrase)
        print(in_phrase)
        print(search_criteria)
        print(multiple_search)

    # checking user ids
    if dependency == 'dependent':
        assert ref_user_id == search_criteria[0][0]
    else:
        assert ref_user_id not in [x[0] for x in search_criteria]

    # checking search words in phrase or not
    assert (search_word in ref_phrase) == in_phrase

    # ensure no search video paths are the same as the ref video path
    assert ref_video_path not in [x[-1] for x in search_criteria]

    return search_criteria, ref_phrase, ref_video_path


def create_dataset(args):
    pprint.pprint(args.__dict__)

    with open(args.phrases_path, 'r') as f:
        phrases = f.read().splitlines()

    # create vocab
    vocab = set()
    for phrase in phrases:
        words = clean_text(phrase).split(' ')
        vocab = vocab.union(set(words))
    vocab = sorted(list(vocab))
    print('Vocab size:', len(vocab))

    # get syllable count of words
    syllable_d = {}
    for word in vocab:
        num_syllables = syllable_count(word)
        syllable_d[num_syllables] = syllable_d.get(num_syllables, 0) + 1
    max_words_per_syllable = min(syllable_d.values())
    print('Num Syllables:', list(syllable_d.keys()))
    print('Max words per syllable:', max_words_per_syllable)

    # get user video paths
    videos_directory = Path(args.videos_path)
    phrase_lookup = {
        str(i+1): phrase
        for i, phrase in enumerate(phrases)
    }
    user_video_paths = {}
    for video_sub_directory in videos_directory.glob('*'):
        user_id = video_sub_directory.name
        phrase_video_paths = create_templates(str(video_sub_directory), [r'SRAVIExtended(.+)P(\d+)-S(\d+)'],
                                              phrase_lookup, preprocess=args.preprocess, include_templates=False,
                                              include_video_paths=True)
        word_to_video_paths = {}
        for phrase, video_path in phrase_video_paths:
            for word in clean_text(phrase).split(' '):
                word_video_paths = word_to_video_paths.get(word, [])
                word_video_paths.append([clean_text(phrase), video_path])
                random.shuffle(word_video_paths)  # should be the same seeded random shuffle
                word_to_video_paths[word] = word_video_paths
        user_video_paths[user_id] = word_to_video_paths
    user_ids = sorted(list(user_video_paths.keys()))

    # create user mappings
    user_mappings = {}
    for user_id in user_ids:
        other_user_ids = [other_user_id for other_user_id in user_ids if other_user_id != user_id]
        random.shuffle(other_user_ids)  # should be the same seeded random shuffle
        user_mappings[user_id] = other_user_ids

    # create word mappings for each user
    word_mappings = {}
    for user_id in user_ids:
        word_mappings[user_id] = {}
        for word in vocab:
            other_words = [other_word for other_word in vocab if other_word != word]
            random.shuffle(other_words)  # should be the same seeded random shuffle
            word_mappings[user_id][word] = other_words[0]

    data, columns = [], [
        'Ref User', 'Search User',
        'Phrase', 'Ref Video Path', 'Num Ref Frames', 'Ref FPS', 'Ref Duration',
        'Word', 'Search Video Path', 'Num Search Frames', 'Search FPS', 'Search Duration', 'Original Search Template Size', 'Clipped Search Template Size', 'Window Size',
        'FA Start', 'FA End', 'FA Likelihood',
        'Best Padding', 'Tried Paddings'
    ]

    # just concentrate on 3 syllables
    del syllable_d[1]
    del syllable_d[2]

    for num_syllables in sorted(list(syllable_d.keys())):
        print('Num Syllables:', num_syllables)
        search_words = sorted([word for word in vocab
                               if syllable_count(word) == num_syllables][:max_words_per_syllable])
        for search_word in search_words:  # same order of search words
            for ref_user_id in user_ids:  # same order of ref user ids
                for dependency, in_phrase in [('dependent', True),
                                              ('dependent', False),
                                              ('independent', True),
                                              ('independent', False)]:

                    search_criteria, ref_phrase, ref_video_path = find_ref_and_search_video_paths_2(
                        user_video_paths=user_video_paths,
                        user_mappings=user_mappings,
                        ref_user_id=ref_user_id,
                        search_word=search_word,
                        dependency=dependency,
                        in_phrase=in_phrase,
                        multiple_search=args.use_multiple_search_templates
                    )
                    # print(ref_user_id, search_word, dependency, in_phrase, ref_phrase, ref_video_path, search_criteria)
                    # continue

                    lowest_distance = np.inf
                    best_index = None
                    all_results = []
                    for i, (search_user_id, search_phrase, search_video_path) in enumerate(search_criteria):
                        results = kws(
                            ref_video_path=ref_video_path,
                            search_video_path=search_video_path,
                            search_transcript=search_phrase,
                            search_words=search_word,
                            preprocess=args.preprocess,
                            core_search_template=args.core_search_template,
                            use_pad_finding=args.use_pad_finding,
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
                    search_user_id = search_criteria[best_index][0]
                    search_video_path = search_criteria[best_index][-1]

                    data.append([
                        ref_user_id, search_user_id,
                        ref_phrase, ref_video_path, results['Num Ref Frames'], results['Ref FPS'], results['Ref Duration'],
                        search_word, search_video_path, results['Num Search Frames'], results['Search FPS'], results['Search Duration'], results['Original Search Template Size'], results['Clipped Search Template Size'], results['Window Size'],
                        results['FA Start'], results['FA End'], results['FA Likelihood'],
                        results['Best Padding'], results['Tried Paddings']
                    ])

                    df = pd.DataFrame(data=data, columns=columns)
                    df.to_csv(args.output_path, index=False)


def get_roc(distances, labels, upper_bound):
    # # false positive rate
    # fprs = []
    # # true positive rate
    # tprs = []
    #
    # # get number of positive and negative examples in the dataset
    # P = sum(labels)
    # N = len(labels) - P
    #
    # if not P or not N:
    #     return None
    #
    # thresholds = np.arange(0.0, upper_bound, 0.01)
    #
    # # iterate through all thresholds and determine fraction of true positives
    # # and false positives found at this threshold
    # for thresh in thresholds:
    #     fp = 0
    #     tp = 0
    #     for label, distance in zip(labels, distances):
    #         if distance < thresh:
    #             if label == 1:
    #                 tp += 1
    #             if label == 0:
    #                 fp += 1
    #
    #     fprs.append(fp / float(N))
    #     tprs.append(tp / float(P))
    #
    # # get auc
    # auc = np.trapz(tprs, fprs)
    #
    # # get optimal thresholds (where tpr is high and fpr is low i.e. tpr - (1 - fpr) is closest to zero
    # best_threshold = None
    # best_score = 1000
    # best_x, best_y = None, None
    # for threshold, tpr, fpr in zip(thresholds, tprs, fprs):
    #     score = abs(tpr - (1 - fpr))
    #     if score < best_score:
    #         best_score = score
    #         best_threshold = threshold
    #         best_x = fpr
    #         best_y = tpr
    #
    # return tprs, fprs, auc, best_threshold, best_x, best_y

    # false negative and positive rates
    fnrs, fprs = [], []

    # get num of +ve and -ve samples in the dataset
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    thresholds = np.arange(0.0, upper_bound, 0.01)
    for threshold in thresholds:
        fn_count, fp_count = 0, 0
        for label, distance in zip(labels, distances):
            if distance < threshold and label == 0:  # pred +ve when -ve
                fp_count += 1
            elif distance >= threshold and label == 1:  # pred -ve when +ve
                fn_count += 1

        fprs.append(fp_count / float(num_neg))
        fnrs.append(fn_count / float(num_pos))

    # get auc
    auc = np.trapz(fnrs, fprs)

    # get optimal thresholds (where fnr and fpr is low i.e.)
    best_threshold = None
    best_score = 1000
    best_x, best_y = None, None
    for threshold, fnr, fpr in zip(thresholds, fnrs, fprs):
        score = abs(fnr - fpr)
        if score < best_score:
            best_score = score
            best_threshold = threshold
            best_x = fpr
            best_y = fnr

    return fnrs, fprs, auc, best_threshold, best_x, best_y, thresholds


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def get_valley_stats(xs, distances, debug=False):
    initial_xs = np.asarray(xs)
    initial_distances = np.asarray(distances)

    if debug:
        plt.plot(initial_xs, initial_distances)
        plt.plot(initial_distances.argmin(), initial_distances.min(), 'or', label='Initial Min')

    distances = initial_distances.copy()

    peaks, _ = find_peaks(-distances)
    prominences = peak_prominences(-distances, peaks)[0]
    contour_heights = distances[peaks] + prominences
    if debug:
        plt.plot(peaks, distances[peaks], "x", label='Troughs')
        plt.vlines(x=peaks, ymin=contour_heights, ymax=distances[peaks], label='Valley Prominences')

    results_half = peak_widths(-distances, peaks, rel_height=0.5)
    results_full = peak_widths(-distances, peaks, rel_height=1)

    results_half = list(results_half[1:])
    results_half[0] = -results_half[0]

    results_full = list(results_full[1:])
    results_full[0] = -results_full[0]

    if debug:
        plt.hlines(*results_half, color="C2", label='Valley Half Widths')
        plt.hlines(*results_full, color="C3", label='Vally Full Widths')

    # calculate area of valleys using shoelace formula
    # find largest valley by this area
    max_area = 0
    largest_valley = None
    largest_valley_xs = None
    largest_valley_ys = None
    areas = []
    for i, (line_y, line_xmin, line_xmax) in enumerate(zip(*results_full)):
        polygon_xs = [line_xmin]
        polygon_ys = [line_y]
        for x, y in zip(initial_xs, initial_distances):
            if line_xmin <= x <= line_xmax:
                polygon_xs.append(x)
                polygon_ys.append(y)
        polygon_xs.append(line_xmax)
        polygon_ys.append(line_y)

        area = poly_area(polygon_xs, polygon_ys)
        # print(i+1, area)
        if area > max_area:
            max_area = area
            largest_valley = peaks[i]
            largest_valley_xs = polygon_xs
            largest_valley_ys = polygon_ys

        areas.append(area)

    if largest_valley is None:
        return
    largest_valley_x, largest_valley_y = initial_xs[largest_valley], initial_distances[largest_valley]
    largest_valley_area = max_area
    areas = sorted(areas, reverse=True)
    diff_between_largest_2_areas = areas[0] - areas[1]
    area_range = areas[0] - areas[-1]
    area_mean = np.mean(areas)

    # calculate largest valley half-width
    half_widths = []
    for i, (line_y, line_xmin, line_xmax) in enumerate(zip(*results_half)):
        half_width = line_xmax - line_xmin
        half_widths.append(half_width)
    half_widths = sorted(half_widths, reverse=True)
    largest_half_width = half_widths[0]
    diff_between_largest_2_half_widths = half_widths[0] - half_widths[1]
    half_width_range = half_widths[0] - half_widths[-1]
    half_width_mean = np.mean(half_widths)

    if debug:
        print(initial_xs, initial_distances, largest_valley)
        print(f'\nBiggest valley found at '
              f'x = {initial_xs[largest_valley]}, y = {initial_distances[largest_valley]}')
        plt.plot(initial_xs[largest_valley], initial_distances[largest_valley], 'o', label='Largest Valley (Min)')
        plt.fill(largest_valley_xs, largest_valley_ys)  # fill the best valley
        plt.ylim((0, 4.0))
        plt.legend()
        plt.show()

    return {
        'largest_valley_coords': [largest_valley_x, largest_valley_y],
        'largest_valley_area': largest_valley_area,
        'largest_half_width': largest_half_width,
        'diff_between_largest_2_areas': diff_between_largest_2_areas,
        'diff_between_largest_2_half_widths': diff_between_largest_2_half_widths,
        'area_range': area_range,
        'half_width_range': half_width_range,
        'area_mean': area_mean,
        'half_width_mean': half_width_mean,
        'num_troughs': len(peaks),
    }


def show_results(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print(pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))


def compute_eer(fpr, fnr, thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold.
    EER => where FPR == FNR
    """
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))

    return eer, thresholds[min_index]


def get_pose_estimation(video_path, port='8085'):
    num_attempts = 3

    while num_attempts != 0:
        with open(video_path, 'rb') as f:
            try:
                pose_response = requests.post(f'http://127.0.0.1:{port}/estimate', files={'video': f.read()})
            except Exception as e:
                print(e)
                num_attempts -= 1
                time.sleep(5)
                continue
            if pose_response.status_code != HTTPStatus.OK:
                return

        return pose_response.json()  # contains median direction in entire video

    return


def get_pitch_and_yaw_directions(direction):
    try:
        pitch, yaw = direction.split(' ')
    except ValueError:
        pitch, yaw = 'centre', 'centre'

    return pitch, yaw


def analyse_dataset(args):
    random.seed()  # reset the seed to be random again

    df = pd.read_csv(args.dataset_path)

    def wps(word, duration):
        num_words = len(word.split(' '))
        return float(num_words) / float(duration)

    variable = 'Distances w/ Path Norm'
    # variable = 'Distances Un-norm'
    # variable = 'Distances w/ Largest Size Norm'
    # variable = 'Distance w/ Width + Height Norm'
    # variable = 'Distance w/ Optimal Path Length Ratio'

    # compatibility for earlier versions
    if 'Ref Phrase' not in df.columns:
        df['Ref Phrase'] = df.apply(lambda row: row['Phrase'], axis=1)
    if 'In Phrase' not in df.columns:
        df['In Phrase'] = df.apply(lambda row: 1 if row['Search Term'] in row['Ref Phrase'] else 0, axis=1)

    # add new columns/features
    df['Dependency'] = df.apply(lambda row: 'Dependent' if row['Ref User'] == row['Search User'] else 'Independent', axis=1)
    df['Label'] = df.apply(lambda row: 1 if row['Search Term'] in row['Ref Phrase'] else 0, axis=1)
    df['Search Term Syllables'] = df.apply(lambda row: syllable_count(row['Search Term']), axis=1)
    df['Tried Paddings'] = df.apply(lambda row: ast.literal_eval(row['Tried Paddings']), axis=1)
    df['Best Results'] = df.apply(lambda row: row['Tried Paddings'][row['Best Padding']], axis=1)
    df['Distances w/ Path Norm'] = df.apply(lambda row: row['Best Results']['Distances'], axis=1)
    df['Path Lengths'] = df.apply(lambda row: row['Best Results']['Path Lengths'], axis=1)
    df['Ref Template Sizes'] = df.apply(lambda row: row['Best Results']['Ref Template Sizes'], axis=1)
    df['Search Template Sizes'] = df.apply(lambda row: [row['Clipped Search Template Size']] * len(row['Path Lengths']), axis=1)
    df['Distances Un-norm'] = df.apply(lambda row: [d * l for d, l in zip(row['Distances w/ Path Norm'], row['Path Lengths'])], axis=1)
    df['Distances w/ Largest Size Norm'] = df.apply(
        lambda row: [d / r if r > s else d / s for d, r, s in zip(row['Distances Un-norm'], row['Ref Template Sizes'], row['Search Template Sizes'])],
        axis=1
    )
    df['Distance w/ Width + Height Norm'] = df.apply(
        lambda row: [d / (r + s) for d, r, s in zip(row['Distances Un-norm'], row['Ref Template Sizes'], row['Search Template Sizes'])],
        axis=1
    )
    df['Optimal Path Lengths'] = df.apply(
        lambda row: [r if r > s else s for r, s in zip(row['Ref Template Sizes'], row['Search Template Sizes'])],
        axis=1
    )
    df['Distance w/ Optimal Path Length Ratio'] = df.apply(
        lambda row: [d / o for d, l, o in zip(row['Distances Un-norm'], row['Path Lengths'], row['Optimal Path Lengths'])],
        axis=1
    )
    df['Min Distance'] = df.apply(lambda row: min(row[variable]), axis=1)
    df['Min Distance / Med Distances'] = df.apply(lambda row: min(row[variable]) / np.median(row[variable]), axis=1)
    df['Window Size w/ Padding'] = df.apply(lambda row: row['Window Size'] + row['Best Padding'], axis=1)
    # for index, row in df.iterrows():
    #     best_padding_results = row['Tried Paddings'][row['Best Padding']]
    #     valley_stats = get_valley_stats(best_padding_results['Xs'], best_padding_results['Distances'])
    #     if valley_stats is None:
    #         df.drop(index, inplace=True)
    #         continue
    #     df.at[index, 'Largest Valley Area'] = valley_stats['largest_valley_area']
    #     df.at[index, 'Largest Valley Distance'] = valley_stats['largest_valley_coords'][1]
    #     df.at[index, 'Num Troughs'] = valley_stats['num_troughs']
    #     df.at[index, 'Largest Valley Half-Width'] = valley_stats['largest_half_width']
    #     df.at[index, 'Diff Between Largest 2 Areas'] = valley_stats['diff_between_largest_2_areas']
    #     df.at[index, 'Diff Between Largest 2 Half-Widths'] = valley_stats['diff_between_largest_2_half_widths']
    #     df.at[index, 'Area Range'] = valley_stats['area_range']
    #     df.at[index, 'Area Mean'] = valley_stats['area_mean']
    #     df.at[index, 'Half-Width Range'] = valley_stats['half_width_range']
    #     df.at[index, 'Half-Width Mean'] = valley_stats['half_width_mean']
    df['Ref Phrase WPS'] = df.apply(lambda row: wps(row['Ref Phrase'], row['Ref Duration']), axis=1)
    df['Search Term WPS'] = df.apply(lambda row: wps(row['Search Term'], row['Search Duration']), axis=1)
    df['Search Term Word Length'] = df.apply(lambda row: len(row['Search Term'].split(' ')), axis=1)

    # only include samples where the search term is a single word rather than multiple
    # df = df[df['Search Term Word Length'] == 1]

    # only include samples where forced alignment score is high
    if args.exclude_less_than_fa_score:
        df = df[df['FA Likelihood'] > args.exclude_less_than_fa_score]

    # only allow similar words per second
    # df = df[abs(df['Phrase WPS'] - df['Word WPS']) < 0.2]

    # get num search words
    unique_search_words = set(df['Search Term'])
    print('Num Unique Search Words:', len(unique_search_words))

    # check for any possible duplicates
    d = {}
    for index, row in df.iterrows():
        tried_paddings = str(row['Tried Paddings'])
        count = d.get(tried_paddings, 0)
        if count == 1:
            print(f'Duplicate found at index: {index}')
            exit()
        d[tried_paddings] = count + 1

    # show how many samples have incomplete 0 padding xs
    counter = 0
    for index, row in df.iterrows():
        num_ref_frames = row['Num Ref Frames']
        search_window_size = row['Window Size']
        num_window_positions = len(list(range(num_ref_frames - search_window_size)))

        default_padding_results = row['Tried Paddings'][0]  # focusing on 0 paddingq
        if len(default_padding_results['Xs']) == num_window_positions:
            counter += 1
        else:
            if args.remove_incomplete:
                df.drop(index, inplace=True)
    if not args.remove_incomplete:
        # higher value here indicates more samples have ref videos where all window positions are utilised
        # skipping window positions could be because of angle of person speaking and CFE therefore breaks
        print('% completed 0 padding Xs:', round(counter / len(df), 2))

    # get histogram of pose estimation diffs between reference and search videos
    if args.do_pose_estimation:
        pitch_indices = {'upper': 0, 'centre': 1, 'lower': 2}
        yaw_indices = {'left': 0, 'centre': 1, 'right': 2}
        pose_diffs = []
        for index, row in tqdm(list(df.iterrows())):
            ref_video_path, search_video_path = row['Ref Video Path'], row['Search Video Path']
            search_word = row['Search Term']
            with open(search_video_path.replace('.mp4', '.txt'), 'r') as f:
                search_transcript = f.readline().replace('Text:', '').strip().lower()

            results = find_search_area(search_video_path, search_transcript, search_word)
            if not results:
                continue
            start_time, end_time, score = results

            with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
                slice_result = precise_slice(search_video_path, start_time, end_time, f.name)
                if slice_result != 0:
                    continue
                f.seek(0)

                search_direction = get_pose_estimation(f.name)
                if search_direction is None:
                    continue
                search_direction = search_direction['direction']

            ref_direction = get_pose_estimation(ref_video_path)
            if ref_direction is None:
                continue
            ref_direction = ref_direction['direction']

            if args.keep_specific_direction is not None and not \
                    (args.keep_specific_direction == ref_direction == search_direction):
                df.drop(index, inplace=True)
                continue

            ref_pitch, ref_yaw = get_pitch_and_yaw_directions(ref_direction)
            search_pitch, search_yaw = get_pitch_and_yaw_directions(search_direction)

            diff = abs(pitch_indices[ref_pitch] - pitch_indices[search_pitch]) + \
                   abs(yaw_indices[ref_yaw] - yaw_indices[search_yaw])

            if args.keep_specific_diff is not None and args.keep_specific_diff != diff:
                df.drop(index, inplace=True)
            else:
                pose_diffs.append(diff)

        if args.show_pose_diffs_hist:
            plt.hist(pose_diffs)
            plt.show()

    # # show example of the valley stats
    # random_df = df.sample(n=1)
    # # random_df = df[(df['Ref User'] == 'LMQ') & (df['Search User'] == 'LMQ') & (df['Phrase'] == 'i\'m going on holiday') & (df['Word'] == 'holiday')]
    # print(random_df['Ref User'].values[0], random_df['Search User'].values[0],
    #       random_df['Phrase'].values[0], random_df['Word'].values[0],
    #       random_df['Num Troughs'].values[0])
    # get_valley_stats(random_df['Xs'].values[0], random_df['Distances'].values[0], debug=True)

    # # show specific example of results
    # ref_user = 'RMC'
    # search_user = 'RMC'
    # phrase = 'what time is it'
    # word = 'headphones'
    # random_df = df[(df['Ref User'] == ref_user) &
    #                (df['Search User'] == search_user) &
    #                (df['Phrase'] == phrase) &
    #                (df['Word'] == word)]
    # for index, row in random_df.iterrows():
    #     for padding, results in row['Tried Paddings'].items():
    #         if results is None:
    #             continue
    #         plt.plot(results['Xs'], results['Distances'], label=f'{padding}')
    #     plt.title(f'{row["Ref User"]} vs {row["Search User"]}, '
    #               f'"{row["Phrase"]}" vs "{row["Word"]}", {row["Best Padding"]}',
    #               fontdict={'color': 'green' if row['Label'] == 1 else 'red'})
    # plt.legend()
    # plt.show()

    try:
        # show dependency counts in pie chart
        df['Dependency'].value_counts().plot.pie()
        plt.title(f'Total Samples: {len(df)}')
        plt.show()
    except TypeError:
        # no numeric data to plot for dependent/independent case
        pass

    # plot multiple random distance line graphs
    for dependency in ['Dependent', 'Independent']:
        sub_df = df[df['Dependency'] == dependency]
        if len(sub_df) == 0:
            continue
        num_rows, num_columns = 3, 3
        num_plots = num_rows * num_columns
        if len(sub_df) < num_plots:
            random_samples = sub_df
        else:
            random_samples = sub_df.sample(n=num_rows * num_columns)
        random_samples = random_samples.reset_index()
        fig, axs = plt.subplots(num_rows, num_columns)
        fig.suptitle(f'{dependency}')
        fig.tight_layout(pad=2)
        row_i, column_i = 0, 0
        annots, plots = [], []
        for index, row in random_samples.iterrows():

            annot = axs[row_i, column_i].annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                                  bbox=dict(boxstyle="round", fc="w"),
                                                  arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)
            annots.append(annot)

            for padding, results in row['Tried Paddings'].items():
                if results is None:
                    continue
                distances = results['Distances']
                plot = axs[row_i, column_i].plot(results['Xs'], distances, label=f'{padding}', marker='o')
                if padding == 0:
                    plots.append(plot)

            if len(row['Ref Phrase']) > 20:
                title = f'{row["Ref User"]} vs {row["Search User"]}, {row["Best Padding"]}'
            else:
                title = f'{row["Ref User"]} vs {row["Search User"]}, "{row["Ref Phrase"]}" vs "{row["Search Term"]}", {row["Best Padding"]}'
            axs[row_i, column_i].set_title(title, fontdict={'color': 'green' if row['Label'] == 1 else 'red'})
            axs[row_i, column_i].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            if column_i == num_columns - 1:
                column_i = 0
                row_i += 1
            else:
                column_i += 1

        def update_annot(event, row, ind, annot):
            x, y = event.xdata, event.ydata

            # just show results of pad 0
            ind = ind['ind'][0]
            pad_zero_results = row['Tried Paddings'][0]
            unnorm_distance = pad_zero_results['Distances'][ind] * pad_zero_results['Path Lengths'][ind]
            path_length = pad_zero_results['Path Lengths'][ind]
            ref_template_size = pad_zero_results['Ref Template Sizes'][ind]
            search_template_size = row['Clipped Search Template Size']

            annot.xy = (x, y)
            annot.set_text(f'{round(unnorm_distance, 2)}, {path_length}, {search_template_size, ref_template_size}')

        def hover(event):
            for i, (ax, plot, annot) in enumerate(zip(axs.flatten(), plots, annots)):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    # print(ax, plot)
                    cont, ind = plot[0].contains(event)
                    row = random_samples.iloc[i]
                    if cont:
                        update_annot(event, row, ind, annot)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        break
                    else:
                        # show ref phrase and search phrase instead if just hovering over axis
                        annot.xy = (event.xdata, event.ydata)
                        text = f'"{row["Ref Phrase"]}" vs "{row["Search Term"]}"'
                        text = textwrap.fill(text, 20)
                        annot.set_text(text)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        break
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        fig.canvas.mpl_connect('motion_notify_event', hover)

        plt.show()

    # plot distribution and ROC curves
    num_syllables = sorted(df['Search Term Syllables'].unique())
    variable = 'Min Distance'
    # variable = 'Min Distance / Med Distances'
    x_axis_limit = df[variable].max() + 2
    for num_syllable in num_syllables:
        dependent_df = df[(df['Search Term Syllables'] == num_syllable) & (df['Dependency'] == 'Dependent')]
        independent_df = df[(df['Search Term Syllables'] == num_syllable) & (df['Dependency'] == 'Independent')]

        for sub_df, title in zip([dependent_df, independent_df], ['Dependent', 'Independent']):
            if len(sub_df) > 0:
                sub_df['In Phrase'].value_counts().plot.pie()
                plt.title(title)
                plt.show()

        fig, axs = plt.subplots(2, 3)
        fig.suptitle(f'Using "{variable}", Num Syllables: {num_syllable}')

        # plot boxplots
        for sub_df, ax, title in zip([dependent_df, independent_df], [axs[0, 0], axs[0, 1]], ['Dependent', 'Independent']):
            ax.boxplot([sub_df[sub_df['Label'] == 1][variable], sub_df[sub_df['Label'] == 0][variable]])
            ax.set_title(f'{title}, # Samples: {len(sub_df)}')
            ax.set_xticklabels(['Hit', 'Miss'])
            ax.legend()

        # plot ROC curves
        for sub_df, dependency in zip([dependent_df, independent_df], ['Dependent', 'Independent']):
            distances, labels = sub_df[variable], sub_df['Label']
            try:
                fnrs, fprs, auc, best_threshold, best_x, best_y, thresholds = \
                    get_roc(distances, labels, upper_bound=x_axis_limit)
            except ZeroDivisionError:
                continue
            eer, eer_threshold = compute_eer(np.asarray(fprs), np.asarray(fnrs), thresholds)
            min_fnr_index = fnrs.index(min(fnrs))
            min_fnr = fnrs[min_fnr_index]
            fpr_at_min_fnr = fprs[min_fnr_index]

            # label = f'{dependency}, AUC = {round(auc, 2)}, Thres = {round(best_threshold, 2)}, EER = {round(eer, 2)}'
            label = f'{dependency}\n' \
                    f'AUC = {round(auc, 2)}\n' \
                    f'EER = {round(eer, 2)}\n' \
                    f'Min FNR = {round(min_fnr, 2)}\n' \
                    f'FPR @ Min FNR = {round(fpr_at_min_fnr, 2)}\n' \
                    f'Best Threshold = {round(best_threshold, 2)}'

            axs[0, 2].plot(fprs, fnrs, label=label)
            # axs[0, 2].plot([best_x], [best_y], 'xb')
            axs[0, 2].set_ylabel('False Negative Rate')
            axs[0, 2].set_xlabel('False Positive Rate')
            axs[0, 2].set_title(f'ROC Curves')
            axs[0, 2].legend()

            # show some wrong examples i.e. FP and FN
            fp_examples = sub_df[(sub_df[variable] < best_threshold) & (sub_df['Label'] == 0)]  # predicted +ve when -ve
            fn_examples = sub_df[(sub_df[variable] >= best_threshold) & (sub_df['Label'] == 1)]  # predicted -ve when +ve
            for label, examples in zip(['FP', 'FN'], [fp_examples, fn_examples]):
                print(f'{dependency} - {label} examples:')
                print('****************************')
                for index, row in examples.iterrows():
                    print(f'{row["Ref User"]} vs {row["Search User"]}, '
                          f'"{row["Search Term"]}" vs "{row["Ref Phrase"]}", '
                          f'{row[variable]}')
                print()

            # show classification report
            y_true = sub_df['Label']
            y_pred = [1 if distance < best_threshold else 0 for distance in sub_df[variable]]
            print(len(y_true), len(y_pred))
            show_results(np.array(y_true), np.array(y_pred))

        # plot distribution graphs
        for sub_df, ax, title in zip([dependent_df, independent_df], [axs[1, 0], axs[1, 1]], ['Dependent', 'Independent']):
            for label_int, label_str in zip([1, 0], ['Hit', 'Miss']):
                if len(sub_df[sub_df['Label'] == label_int][variable]) > 1:
                    sns.distplot(sub_df[sub_df['Label'] == label_int][variable], ax=ax, label=label_str)
            ax.set_xlim((0, x_axis_limit))
            ax.legend()

        axs[1, 2].set_axis_off()

        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        plt.show()

        # save ROC curve
        roc_save_path = args.dataset_path.replace('.csv', f'_{num_syllable}.png')
        extent = axs[0, 2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(roc_save_path, bbox_inches=extent.expanded(1.3, 1.3))

    # plot padding graphs
    for num_syllable in num_syllables:
        dependent_df = df[(df['Search Term Syllables'] == num_syllable) & (df['Dependency'] == 'Dependent')]
        independent_df = df[(df['Search Term Syllables'] == num_syllable) & (df['Dependency'] == 'Independent')]

        # plot distribution graphs
        for sub_df, label in zip([dependent_df, independent_df], ['Dependent', 'Independent']):
            sns.distplot(sub_df['Best Padding'], kde=False, label=label)
            plt.xlim((-20, 20))
            plt.ylabel('Frequency')
            plt.title(f'Best Window Padding By Min Distance Histogram: Num Syllables {num_syllable}')
            plt.legend()
        plt.show()

    # # rate of speech analysis
    # plt.scatter(df['Padding'], df['Window Size'])
    # plt.xlabel('Best Padding Found')
    # plt.ylabel('Sliding Window Size (before padding)')
    # plt.show()

    exit()

    print('*************************************************')

    # create small classifier with some feature engineering
    random_state = 2021
    # random_state = 42
    # dependency = 'Independent'
    dependency = 'Dependent'
    features = [
        'Min Distance',
        'Largest Valley Area',
        'Largest Valley Distance',
        'Largest Valley Half-Width',
        'Diff Between Largest 2 Areas',
        'Diff Between Largest 2 Half-Widths',
        'Area Range',
        'Half-Width Range',
        'Area Mean',
        'Half-Width Mean',
        'Num Troughs'
    ]
    X = df[df['Dependency'] == dependency][features]
    Y = df[df['Dependency'] == dependency]['Label']
    assert len(X) == len(Y)

    from main.research.phrase_in_list import auto_feature_engineering
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import chi2, SelectKBest
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

    # min-max normalisation
    scaler = MinMaxScaler()
    X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns)

    # # try some auto feature engineering
    # X = auto_feature_engineering(X, Y, selection_percent=0.8,
    #                              num_depth_steps=3,
    #                              transformatives=[
    #                                  'multiply_numeric',
    #                                  'percentile',
    #                                  # 'divide_numeric',
    #                              ])

    # try polynomial features
    poly = PolynomialFeatures(degree=3)
    poly_features = poly.fit_transform(X)
    X = pd.DataFrame(poly_features, columns=poly.get_feature_names(X.columns))

    print('Num Columns Before:', len(X.columns))

    # look at best features
    n = 1
    best_features = SelectKBest(score_func=chi2, k='all')
    fit = best_features.fit(X, Y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(X.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']  # naming the dataframe columns
    with pd.option_context('display.max_colwidth', -1):
        print(feature_scores.nlargest(10, 'Score'))
    X = X[feature_scores.nlargest(n, 'Score')['Specs']]

    # this gives best performance for Dependent - i.e. best features and no correlation between them i.e. no redundancy
    # X = X[['Min Distance', 'Diff Between Largest 2 Areas', 'Num Troughs']]

    # # best dependent
    # X = X[['Diff Between Largest 2 Areas^3',
    #        'Min Distance^3',
    #        'Num Troughs']]

    # # best independent
    # X = X[['Min Distance^3',
    #        'Diff Between Largest 2 Areas^3',
    #        'Largest Valley Area Diff Between Largest 2 Areas^2']]

    print('Num Columns After:', len(X.columns))

    # show feature correlation
    correlation_mat = X.corr()
    sns.heatmap(correlation_mat, annot=True)
    plt.show()

    # # normalise features
    # scaler = MinMaxScaler()
    # X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=random_state, stratify=Y)
    print('Num Train:', len(x_train))
    print('Num Test:', len(x_test))
    print('Train Label Counts:', y_train.value_counts().to_string())
    print('Test Label Counts:', y_test.value_counts().to_string())

    # clf = KNeighborsClassifier()
    # clf = RandomForestClassifier(random_state=random_state)
    clf = SVC(random_state=random_state)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    show_results(y_test, y_pred)

    print('*************************************************')

    # grid search SVM
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [50, 10, 5, 1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    # param_grid = {
    #     'n_estimators': [1, 2, 5, 10, 50, 100, 200],
    #     'max_depth': [None, 1, 2, 5, 10, 50],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }
    est = SVC()
    # est = RandomForestClassifier()
    clf = GridSearchCV(est, param_grid, scoring='recall_macro', cv=5, n_jobs=4, verbose=True)
    clf.fit(x_train, y_train)  # uses k-fold cross-validation on training set
    print('Best score:', clf.best_score_)
    print('Best params:', clf.best_params_)
    y_pred = clf.predict(x_test)  # uses best estimator from grid search
    show_results(y_test, y_pred)


def check_template_normalisation(args):
    """
    Check out normalisation of current DTW implementation:
    - crop search area from ref video and run sliding window (in-phrase)
    - use an anti-template vs the ref video and run sliding window (not in-phrase)
    - use search area from a different ref video and run sliding window (in-phrase)
    """
    ref_video_path = args.reference_video_path
    ref_frames = get_frames(ref_video_path)
    ref_fps = get_fps(ref_video_path)

    search_video_path = args.search_video_path

    phrase = args.phrase
    search_word = args.search_word

    un_normalise = args.un_normalise
    normalise_by_search_template = args.normalise_by_search_template

    forced_alignment = run_forced_alignment(ref_video_path, phrase)
    for word, start_time, end_time, score in forced_alignment:
        if word == search_word:
            break
    search_window_size, search_fps, search_duration, ref_template = \
        extract_search_template(ref_video_path, start_time, end_time, debug=True)
    # xs, distances = sliding_window(ref_frames, same_video_search_template, search_window_size, ref_fps,
    #                                un_normalise=un_normalise,
    #                                normalise_by_search_template=normalise_by_search_template)
    # plt.plot(xs, distances, label='Search Template From Ref Video')

    forced_alignment = run_forced_alignment(search_video_path, phrase)
    for word, start_time, end_time, score in forced_alignment:
        if word == search_word:
            break
    search_window_size, search_fps, search_duration, search_template = \
        extract_search_template(search_video_path, start_time, end_time, debug=True)
    window_size = int(ref_fps * search_duration)
    # xs, distances = sliding_window(ref_frames, diff_video_search_template, window_size, ref_fps,
    #                                un_normalise=un_normalise,
    #                                normalise_by_search_template=normalise_by_search_template)
    # plt.plot(xs, distances, label='Search Template From Search Video')

    # max_search_template_value = search_template.max()
    # anti_search_template = np.full(search_template.shape, max_search_template_value)
    # xs, distances = sliding_window(ref_frames, anti_search_template, window_size, ref_fps,
    #                                normalise=normalise,
    #                                normalise_by_search_template=normalise_by_search_template)
    # plt.plot(xs, distances, label='Anti-Search Template')

    # plt.xlabel('Window Position')
    # plt.ylabel('DTW Distance')
    # plt.legend()
    # plt.title(f'Un-norm: {un_normalise}, Norm By Search Template: {normalise_by_search_template}')
    # plt.show()

    dtw = DTW(**Config().__dict__)
    dtw.find_path = 1
    path, distance, cost_matrix, path_length = get_dtw_distance(dtw, search_template, ref_template)
    print(search_template.shape, ref_template.shape, path, distance, cost_matrix.shape, path_length, len(path))
    cost_matrix = np.where(cost_matrix > 100, 0, cost_matrix)
    sns.heatmap(cost_matrix, cmap='YlGnBu')
    xs, ys = zip(*path)
    plt.plot(ys, xs)
    plt.show()

    # distance, path = fastdtw(diff_video_search_template, same_video_search_template, dist=sqeuclidean)
    # print(distance, path)


def show_dtw_cost_matrix(args):
    ref_video_path = args.reference_video_path
    search_video_path = args.search_video_path

    # first extract word from search video
    forced_alignment = run_forced_alignment(search_video_path, args.phrase)
    for word, start_time, end_time, score in forced_alignment:
        if word.lower() == args.search_word:
            break
    search_window_size, search_fps, search_duration, search_template = \
        extract_search_template(search_video_path, start_time, end_time, preprocess=True, debug=True)

    ref_template = create_template(ref_video_path, preprocess=True).blob.astype(np.float32)

    dtw = DTW(**Config().__dict__)
    dtw.find_path = 1
    # dtw.transition_cost = 0

    vec_dist = dtw._calculate_cost_matrix(search_template, ref_template)
    sns.heatmap(vec_dist, cmap='YlGnBu')
    plt.show()

    path, distance, cost_matrix, path_length = get_dtw_distance(dtw, search_template, ref_template)
    print(search_template.shape, ref_template.shape, distance, cost_matrix.shape, path_length, len(path))
    cost_matrix = np.where(cost_matrix == 1e10, 0, cost_matrix)
    sns.heatmap(cost_matrix, cmap='YlGnBu')
    xs, ys = zip(*path)
    plt.plot(ys, xs)
    plt.show()

    num_search_frames = search_template.shape[0]
    num_ref_frames = ref_template.shape[0]
    distances = []
    for i in range(num_ref_frames - num_search_frames):
        vec_dist_window = vec_dist[:, i:i+num_search_frames]
        assert vec_dist_window.shape == (num_search_frames, num_search_frames)

        path, distance, cost_matrix, path_length = dtw._calculate_path_and_distance(vec_dist_window)
        distances.append(distance)

    plt.plot(distances)
    plt.show()


def analyse_fast_sliding_window(args):
    ref_video_path = args.reference_video_path
    search_video_path = args.search_video_path

    forced_alignment = run_forced_alignment(search_video_path, args.phrase)
    for word, start_time, end_time, score in forced_alignment:
        if word.lower() == args.search_word.lower():
            break
    search_window_size, search_fps, search_duration, search_template = \
        extract_search_template(search_video_path, start_time, end_time, preprocess=True, debug=True)

    ref_template = create_template(ref_video_path, preprocess=True).blob
    print(ref_template.shape, search_template.shape)

    num_ref_frames = ref_template.shape[0]
    num_search_frames = search_template.shape[0]

    dtw = DTW(**Config().__dict__)

    distances = []
    for i in range(num_ref_frames - num_search_frames):
        window_template = ref_template[i:i + num_search_frames, ]

        path, distance, cost_matrix, path_length = get_dtw_distance(dtw, search_template.astype(np.float32),
                                                                    window_template.astype(np.float32))

        distances.append(distance)

    plt.plot(distances)
    plt.show()


def main(args):
    f = {
        'create_dataset': create_dataset,
        'analyse_dataset': analyse_dataset,
        'check_template_normalisation': check_template_normalisation,
        'show_dtw_cost_matrix': show_dtw_cost_matrix,
        'analyse_fast_sliding_window': analyse_fast_sliding_window
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('create_dataset')
    parser_1.add_argument('videos_path')
    parser_1.add_argument('phrases_path')
    parser_1.add_argument('--preprocess', action='store_true')
    parser_1.add_argument('--output_path', default='kws_sravi_dataset.csv')
    parser_1.add_argument('--core_search_template', type=float, default=1.0)
    parser_1.add_argument('--use_multiple_search_templates', action='store_true')
    parser_1.add_argument('--use_pad_finding', action='store_true')
    parser_1.add_argument('--debug', action='store_true')

    parser_2 = sub_parsers.add_parser('analyse_dataset')
    parser_2.add_argument('dataset_path')
    parser_2.add_argument('--remove_incomplete', action='store_true')
    parser_2.add_argument('--do_pose_estimation', action='store_true')
    parser_2.add_argument('--show_pose_diffs_hist', action='store_true')
    parser_2.add_argument('--keep_specific_diff', type=int)
    parser_2.add_argument('--keep_specific_direction')
    parser_2.add_argument('--exclude_less_than_fa_score', type=int)

    parser_3 = sub_parsers.add_parser('check_template_normalisation')
    parser_3.add_argument('reference_video_path')
    parser_3.add_argument('search_video_path')
    parser_3.add_argument('phrase')
    parser_3.add_argument('search_word')
    parser_3.add_argument('--un_normalise', action='store_true')
    parser_3.add_argument('--normalise_by_search_template', action='store_true')

    parser_4 = sub_parsers.add_parser('show_dtw_cost_matrix')
    parser_4.add_argument('reference_video_path')
    parser_4.add_argument('search_video_path')
    parser_4.add_argument('phrase')
    parser_4.add_argument('search_word')

    parser_5 = sub_parsers.add_parser('analyse_fast_sliding_window')
    parser_5.add_argument('reference_video_path')
    parser_5.add_argument('search_video_path')
    parser_5.add_argument('phrase')
    parser_5.add_argument('search_word')

    main(parser.parse_args())
