"""
Experiments to test unseen phrase sets vs the default "best" templates

The idea is that any user not in the default "best" templates is tested
against the default list to represent new users using the system
"""
import argparse
import ast
import io
import os
import re

import requests
from main import configuration
from main.models import Config, PAVAList, PAVAUser
from main.research.cmc import CMC
from main.research.confusion_matrix import ConfusionMatrix
from main.utils.db import find_phrase_mappings, invert_phrase_mappings
from main.utils.io import read_json_file
from main.utils.parsing import extract_template_info

TRANSCRIBE_URL = \
    'http://0.0.0.0:5000/pava/api/v1/lists/{list_id}/transcribe/video'
VIDEOS_DIRECTORY = '/home/domhnall/Documents/sravi_dataset/liopa/{}'

NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
ANALYSE_REGEX = r'(.+),(\[.+\]),(.+)'

user = PAVAUser.create(default_list=True)
lst = PAVAList.get(
    filter=(PAVAList.user_id == user.id),
    first=True
)
list_id = str(lst.id)


def analyse(**kwargs):
    results_path = kwargs['results_path']

    confusion_matrix = ConfusionMatrix()

    wrong_recordings = []

    with open(results_path, 'r') as f:
        for line in f.readlines():
            video_file, predictions, actual_label = re.match(ANALYSE_REGEX,
                                                             line).groups()
            predictions = ast.literal_eval(predictions)

            first_prediction = predictions[0]
            confusion_matrix.append(first_prediction, actual_label)

            if first_prediction != actual_label:
                wrong_recordings.append(video_file)

    phrase_sessions = {}
    session_count = {}
    for recording in wrong_recordings:
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, recording).groups()
        sessions = phrase_sessions.get(phrase_id, [])
        sessions.append(session_id)
        phrase_sessions[phrase_id] = sessions

        session_count[session_id] = session_count.get(session_id, 0) + 1

    print(phrase_sessions)
    confusion_matrix.plot()
    for phrase_id, sessions in phrase_sessions.items():
        print(f'Phrase: {phrase_id}, # Incorrect Predictions: {len(sessions)}')

    print(session_count)


def process_new_users(**kwargs):
    videos_directory = kwargs['videos_directory']
    output_filename = 'unseen_vs_default.csv'

    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    cmc = CMC(num_ranks=3)

    if os.path.exists(output_filename):
        os.remove(output_filename)

    total_num, num_failed = 0, 0
    # get results for entire combination of sessions
    for video_file in os.listdir(videos_directory):
        if not video_file.endswith('.mp4'):
            continue

        total_num += 1

        template_id = video_file.replace('.mp4', '')

        phrase_id = re.match(NEW_RECORDINGS_REGEX, template_id).groups()[1]

        video_path = os.path.join(videos_directory, video_file)
        with open(video_path, 'rb') as f:
            files = {
                'file': (video_file, io.BytesIO(f.read()))
            }
            response = requests.post(
                TRANSCRIBE_URL.format(list_id=list_id),
                files=files
            )

            if response.status_code == 200 and \
                    response.json()['status']['code'] == 200:
                predictions = response.json()['response']['predictions']
                actual_label = pava_phrases[phrase_id]

                prediction_labels = [prediction['label']
                                     for prediction in predictions[:-1]]

                # ignore NOTA phrase
                cmc.tally(prediction_labels, actual_label)

                with open(output_filename, 'a') as f:
                    f.write(f'{video_file},{prediction_labels},{actual_label}\n')
            else:
                print(f'Video failed: {video_path}, {response.json()}')
                num_failed += 1

    cmc.calculate_accuracies(total_num, count_check=False)
    cmc.plot()

    print(f'User: ', cmc.all_rank_accuracies[0])
    print(f'Num tests: {total_num - num_failed}/{total_num}')

    # # get results per session
    # sessions = {}
    # for video_file in os.listdir(videos_directory):
    #     if not video_file.endswith('.mp4'):
    #         continue
    #
    #     template_id = video_file.replace('.mp4', '')
    #     video_path = os.path.join(videos_directory, video_file)
    #
    #     user, phrase_id, session_id = \
    #         re.match(NEW_RECORDINGS_REGEX, template_id).groups()
    #
    #     session_recordings = sessions.get(session_id, [])
    #     session_recordings.append((phrase_id, video_path))
    #     sessions[session_id] = session_recordings
    #
    # cmc = CMC(num_ranks=3)
    # for session_id, recordings in sessions.items():
    #     num_tests = 0
    #     for phrase_id, recording in recordings:
    #         with open(recording, 'rb') as f:
    #             files = {
    #                 'file': (recording, io.BytesIO(f.read()))
    #             }
    #             response = requests.post(
    #                 TRANSCRIBE_URL.format(list_id=list_id),
    #                 files=files
    #             )
    #
    #             if response.status_code == 200 and \
    #                     response.json()['status']['code'] == 200:
    #                 predictions = response.json()['response']['predictions']
    #                 actual_label = pava_phrases[phrase_id]
    #
    #                 prediction_labels = [prediction['label']
    #                                      for prediction in predictions[:-1]]
    #
    #                 # ignore NOTA phrase
    #                 cmc.tally(prediction_labels, actual_label)
    #                 num_tests += 1
    #             else:
    #                 print(f'Video failed: {recording}, {response.json()}')
    #     cmc.calculate_accuracies(num_tests, count_check=False)
    #     cmc.labels.append(session_id)
    #     print(f'Session: {session_id}', cmc.all_rank_accuracies[0])
    #     print(f'Num tests: {num_tests}/{len(recordings)}')
    #
    # print(cmc.all_rank_accuracies)
    # cmc.plot()


def process_previous_users(**kwargs):
    default_directory = kwargs['videos_directory']
    users = kwargs['users']

    phrase_mappings = find_phrase_mappings('PAVA-DEFAULT')
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)
    all_phrases = read_json_file(configuration.PHRASES_PATH)

    output_filepath = 'previous_users_vs_default.csv'

    for user in users:
        cmc = CMC(num_ranks=3)
        num_tests = 0

        videos_directory = os.path.join(default_directory, user)

        for video_file in os.listdir(videos_directory):
            template_id = video_file.replace('.mp4', '')
            user_id, phrase_set, phrase_id, session_id = \
                extract_template_info(template_id)

            if phrase_set + phrase_id in inverse_phrase_mappings:
                video_path = os.path.join(videos_directory, video_file)
                with open(video_path, 'rb') as f:
                    files = {
                        'file': (video_file, io.BytesIO(f.read()))
                    }
                    response = requests.post(
                        TRANSCRIBE_URL.format(list_id=list_id),
                        files=files
                    )

                    if response.status_code == 200:
                        print()
                        print(response.json())
                        print()

                        response = response.json()['response']
                        actual_label = all_phrases[phrase_set][phrase_id]

                        if response == {}:
                            continue

                        predictions = response['predictions']
                        # remove NOTA phrase
                        del predictions[-1]

                        predictions = [prediction['label']
                                       for prediction in predictions]
                        cmc.tally(predictions, actual_label)

                        num_tests += 1

        if num_tests != 0:
            cmc.calculate_accuracies(num_tests, count_check=False)
            accuracies = cmc.all_rank_accuracies[0]
        else:
            accuracies = [0, 0, 0]

        with open(output_filepath, 'a') as f:
            f.write(f'{user},{accuracies},{num_tests},'
                    f'{len(os.listdir(videos_directory))}\n')


def process_manual(**kwargs):
    groundtruth_file = kwargs['groundtruth_file']
    directory = os.path.dirname(groundtruth_file)

    data = []
    with open(groundtruth_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            data.append(line.split(','))

    cmc = CMC(num_ranks=3)
    num_failed = 0
    for video in data:
        if len(video) == 2:
            video_path, actual_label = video
        else:
            video_path, actual_label = video[0], video[2]
        video_path = os.path.join(directory, video_path)

        with open(video_path, 'rb') as f:
            files = {
                'file': (video_path, io.BytesIO(f.read()))
            }
            response = requests.post(
                TRANSCRIBE_URL.format(list_id=list_id),
                files=files
            )

            if response.status_code == 200 and \
                    response.json()['status']['code'] == 200:
                predictions = response.json()['response']['predictions']

                # ignore NOTA phrase
                prediction_labels = [prediction['label']
                                     for prediction in predictions[:-1]]

                cmc.tally(prediction_labels, actual_label)

                with open('predictions.txt', 'a') as f:
                    results = [(prediction['label'], prediction['accuracy'])
                               for prediction in predictions[:-1]]
                    f.write(f'{results},{actual_label}\n')
            else:
                print(f'Video failed: {video_path}, {response.json()}')
                num_failed += 1

    num_tests = len(data)
    cmc.calculate_accuracies(num_tests, count_check=False)
    print(f'Rank accuracies: ', cmc.all_rank_accuracies[0])
    print(f'Num tests: {num_tests - num_failed}/{num_tests}')


def list_type(s):
    return [i for i in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='run_type')

    parser1 = sub_parser.add_parser('new_users')
    parser1.add_argument('videos_directory', type=str)

    parser2 = sub_parser.add_parser('previous_users')
    parser2.add_argument('users', type=list_type)
    parser2.add_argument('videos_directory', type=str)

    parser3 = sub_parser.add_parser('manual')
    parser3.add_argument('groundtruth_file', type=str)

    parser4 = sub_parser.add_parser('analyse')
    parser4.add_argument('results_path', type=str)

    f = {
        'new_users': process_new_users,
        'previous_users': process_previous_users,
        'manual': process_manual,
        'analyse': analyse
    }

    args = parser.parse_args()
    if args.run_type in list(f.keys()):
        f[args.run_type](**args.__dict__)
    else:
        parser.print_help()
