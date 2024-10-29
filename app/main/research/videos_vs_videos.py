"""Ref videos vs test videos"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from main import configuration
from main.research.research_utils import create_template, get_accuracy
from main.utils.io import read_json_file, read_pickle_file, write_pickle_file

VIDEO_REGEX_1 = r'(\d+)_(S\w?[A-Z])(\d+)_S(\d+).mp4'
VIDEO_REGEX_2 = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
ALL_PHRASES = read_json_file(configuration.PHRASES_PATH)
PAVA_PHRASES = ALL_PHRASES['PAVA-DEFAULT']
REF_SIGNALS = 'ref_templates.pkl'


def get_label(video_path, groundtruth_file):
    relative_path = os.path.basename(video_path)
    if groundtruth_file:
        df = pd.read_csv(groundtruth_file, names=['Path', 'Said', 'Actual'])
        phrase = df[df['Path'] == relative_path]['Actual']

        if len(phrase) == 0:
            return None

        is_null = pd.isnull(phrase).values[0]

        if is_null:
            phrase = df[df['Path'] == relative_path]['Said']

        if phrase.values:
            return phrase.values[0]

        return None
    else:
        try:
            phrase_id = re.match(VIDEO_REGEX_1, relative_path).groups()[1:3]
            phrase_set, phrase_set_id = phrase_id[0], str(int(phrase_id[1]))
            phrase = ALL_PHRASES[phrase_set][phrase_set_id]
            if phrase not in PAVA_PHRASES.values():
                phrase = None
        except AttributeError:
            phrase_id = re.match(VIDEO_REGEX_2, relative_path).groups()[1]
            phrase = PAVA_PHRASES[str(int(phrase_id))]

        return phrase


def get_templates(video_list, gt):
    templates = []
    for video in video_list:
        label = get_label(video, gt)
        if not label:
            # we only want PAVA phrases
            continue

        template = create_template(video)
        if not template:
            continue

        templates.append((label, template.blob))

    return templates


def construct_sessions(video_paths):
    sessions = {}
    for video_path in video_paths:
        video_path_basename = os.path.basename(video_path)
        session_id = re.match(VIDEO_REGEX_2, video_path_basename).groups()[2]

        session_video_paths = sessions.get(session_id, [])
        session_video_paths.append(video_path)
        sessions[session_id] = session_video_paths

    for session_id, session_video_paths in sessions.items():
        session_templates = get_templates(session_video_paths, None)
        sessions[session_id] = session_templates

    return sessions


def process(args):
    ref_videos = glob.glob(os.path.join(args.ref_videos_directory, '*.mp4'))
    test_videos = glob.glob(os.path.join(args.test_videos_directory, '*.mp4'))

    # get reference signals
    if os.path.exists(REF_SIGNALS):
        ref_signals = read_pickle_file(REF_SIGNALS)
    else:
        ref_templates = get_templates(ref_videos, args.ref_groundtruth_file)
        ref_signals = [(label, template) for label, template in ref_templates]
        write_pickle_file(ref_signals, REF_SIGNALS)

    if args.session_basis:
        user_sessions = construct_sessions(test_videos)
        for session_id, session_templates in user_sessions.items():
            print(f'Session: {session_id}, '
                  f'Num Templates: {len(session_templates)}')
            accuracies = get_accuracy(ref_signals, session_templates)
            plt.plot([1, 2, 3], accuracies,
                     label=f'S{session_id}', marker='o')
    else:
        test_templates = get_templates(test_videos, args.test_groundtruth_file)
        accuracies = get_accuracy(ref_signals, test_templates)[0]
        plt.plot([1, 2, 3], accuracies, marker='o')
        print(accuracies)

    plt.xlabel('Rank Accuracies')
    plt.ylabel('Accuracy %')
    plt.ylim((0, 101))
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.show()


def main(args):
    process(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_videos_directory')
    parser.add_argument('test_videos_directory')
    parser.add_argument('--ref_groundtruth_file')
    parser.add_argument('--test_groundtruth_file')
    parser.add_argument('--session_basis', type=bool, default=False)

    main(parser.parse_args())
