"""
Experiments to check if we need to update templates in bulk or individually

The idea is:

If we split domhnall's 100 recorded phrases into 2 phrase sets; 1 containing phrases 1-10 and 2 containing phrases 11-20
First test phrase set 2 against default phrase list and record performance
Then add phrase set 1 to the default phrase list
Retest phrase set 2 against the updated reference set

If the performance drops on the second test, it means that the predictions are being
made by how similar the videos are to others (visual) rather than by what's being spoken
If we were to update templates individually in this way, further predictions using
a different phrase might favour the previously added phrase rather than the actual correct one
This means it is better to update templates in bulk
"""
import argparse
import ast
import copy
import io
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
from main import configuration
from main.models import Config, PAVAList, PAVATemplate, PAVAUser
from main.utils.cfe import run_cfe
from main.utils.io import read_json_file, read_pickle_file, write_pickle_file
from main.utils.pre_process import pre_process_signals

TRANSCRIBE_ENDPOINT = \
    'http://0.0.0.0:5000/pava/api/v1/lists/{list_id}/transcribe/video'
NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
ANALYSE_FULL_UPDATE_REGEX = r'(\[.+\]),(\d),(\d+\.\d+),(\d+\.\d+)'
ANALYSE_FULL_UPDATE_REGEX_2 = r'(\[.+\]),(\[.+\]),(\d+\.\d+),(\d+\.\d+)'

user = PAVAUser.create(default_list=True, config=Config())
list_id = PAVAList.get(
    query=(PAVAList.id,),
    filter=(PAVAList.user_id == user.id),
    first=True
)[0]
list_id = str(list_id)


def test_phrase_set(d, videos_path, phrases):
    inverted_phrases = {v: k for k, v in phrases.items()}

    num_correct_predictions = 0
    num_tests = sum([len(lst) for lst in d.values()])
    actual_vs_prediction = []
    for phrase_id, videos in d.items():
        for video in videos:
            full_video_path = os.path.join(videos_path, video)

            with open(full_video_path, 'rb') as f:
                files = {
                    'file': (video, io.BytesIO(f.read()))
                }
                response = requests.post(
                    TRANSCRIBE_ENDPOINT.format(list_id=list_id),
                    files=files
                )
                if response.status_code == 200:
                    json_response = response.json()
                    predictions = json_response['response']['predictions']

                    expected_phrase = phrases[str(phrase_id)]
                    prediction = predictions[0]['label']

                    if prediction == expected_phrase:
                        num_correct_predictions += 1

                    predicted_phrase_id = inverted_phrases[prediction]
                    actual_vs_prediction.append((str(phrase_id),
                                                 str(predicted_phrase_id)))

    accuracy = num_correct_predictions / num_tests
    accuracy = round(accuracy * 100, 1)

    return accuracy, actual_vs_prediction


def partial_phrase_update(**kwargs):
    videos_path = kwargs['videos_path']
    phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    phrase_templates = {}
    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        templates = phrase_templates.get(int(phrase_id), [])
        templates.append(video)
        phrase_templates[int(phrase_id)] = templates

    phrase_templates = dict(sorted(phrase_templates.items()))
    half_templates = len(phrase_templates) // 2

    # split into 2 sets of phrases. 1-10 in set 1, 11-20 in set 2
    d1 = dict(list(phrase_templates.items())[:half_templates])
    d2 = dict(list(phrase_templates.items())[half_templates:])

    # first check the performance of second half of phrases vs default list
    accuracy_1, results_1 = test_phrase_set(d2, videos_path, phrases)
    print('Accuracy before adding: ', accuracy_1)

    dtw_params = Config().__dict__

    # now add the first half of the phrases to the reference set
    default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)
    copy_default_lst = copy.deepcopy(default_lst)
    for phrase_id, videos in d1.items():
        for video in videos:
            full_video_path = os.path.join(videos_path, video)
            with open(full_video_path, 'rb') as f:
                feature_matrix = run_cfe(f)
                pre_processed_matrix = \
                    pre_process_signals([feature_matrix], **dtw_params)[0]

                actual_phrase_content = phrases[str(phrase_id)]
                for i, phrase in enumerate(copy_default_lst.phrases):
                    if phrase.content == actual_phrase_content:
                        pava_template = PAVATemplate(blob=pre_processed_matrix)
                        phrase.templates.append(pava_template)
                        copy_default_lst.phrases[i] = phrase

    write_pickle_file(copy_default_lst, configuration.DEFAULT_LIST_PATH)

    # now retest second half of phrases against the default list
    accuracy_2, results_2 = test_phrase_set(d2, videos_path, phrases)
    print('Accuracy after adding: ', accuracy_2)

    results = []
    for result_1, result_2 in zip(results_1, results_2):
        actual = result_1[0]
        prediction_1, prediction_2 = result_1[1], result_2[1]

        # only choose where it predicted correctly the first time but predicted
        # incorrectly the second time
        if prediction_1 == actual and prediction_1 != prediction_2:
            results.append((result_1[0], prediction_1, prediction_2))

    print(results)

    # make sure to write the original lst back to file
    write_pickle_file(default_lst, configuration.DEFAULT_LIST_PATH)


def full_update_sessions(**kwargs):
    videos_path = kwargs['videos_path']
    phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']
    dtw_params = Config().__dict__

    # create session templates dictionary
    sessions = {}
    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        session_videos = sessions.get(int(session_id), [])
        session_videos.append(video)
        sessions[int(session_id)] = session_videos

    def sessions_to_phrases(d, sessions_to_use):
        # convert session templates dict to phrase templates dict
        phrases = {}
        for session_id, videos in d.items():
            if session_id in sessions_to_use:
                for video in videos:
                    phrase_id = \
                        re.match(NEW_RECORDINGS_REGEX, video).groups()[1]

                    phrase_videos = phrases.get(phrase_id, [])
                    phrase_videos.append(video)
                    phrases[phrase_id] = phrase_videos

        return phrases

    # test every combination of sessions to add vs others
    available_sessions = sorted(list(sessions.keys()))
    print(available_sessions)

    combo_index = kwargs['combo_index']
    uses_test_sessions = True if kwargs['test_sessions'] else False

    tested_sessions = {}

    if uses_test_sessions:
        test_sessions = [int(session) for session in kwargs['test_sessions']]

        available_sessions = list(set(available_sessions) - set(test_sessions))
        print(available_sessions)

        before_accuracy = None

        for i in range(combo_index, len(available_sessions)):
            for combo in itertools.combinations(available_sessions, i + 1):
                sessions_to_add = list(combo)

                print(f'Testing with sessions: {test_sessions}')
                print(f'Adding sessions: {sessions_to_add}')

                phrases_to_add = sessions_to_phrases(sessions, sessions_to_add)
                phrases_to_test = sessions_to_phrases(sessions, test_sessions)

                # do the first test if not done already
                if before_accuracy:
                    accuracy_1 = before_accuracy
                else:
                    accuracy_1, results_1 = test_phrase_set(phrases_to_test,
                                                            videos_path,
                                                            phrases)
                    before_accuracy = accuracy_1

                print(f'Before adding sessions {sessions_to_add}: {accuracy_1}')

                # add the sessions to add to the default list before testing again
                default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)
                copy_default_lst = copy.deepcopy(default_lst)
                for phrase_id, videos in phrases_to_add.items():
                    for video in videos:
                        full_video_path = os.path.join(videos_path, video)
                        with open(full_video_path, 'rb') as f:
                            feature_matrix = run_cfe(f)
                            pre_processed_matrix = \
                                pre_process_signals([feature_matrix],
                                                    **dtw_params)[0]

                            actual_phrase_content = phrases[str(phrase_id)]
                            for i, phrase in enumerate(
                                    copy_default_lst.phrases):
                                if phrase.content == actual_phrase_content:
                                    pava_template = PAVATemplate(
                                        blob=pre_processed_matrix
                                    )
                                    phrase.templates.append(pava_template)
                                    copy_default_lst.phrases[i] = phrase

                write_pickle_file(copy_default_lst,
                                  configuration.DEFAULT_LIST_PATH)

                # do the second test
                accuracy_2, results_2 = test_phrase_set(phrases_to_test,
                                                        videos_path,
                                                        phrases)
                print(f'After adding sessions {sessions_to_add}: {accuracy_2}\n')

                # make sure to write the original lst back to file
                write_pickle_file(default_lst,
                                  configuration.DEFAULT_LIST_PATH)

                # save the results
                with open('full_update_sessions.csv', 'a') as f:
                    f.write(f'{test_sessions},{sessions_to_add},'
                            f'{accuracy_1},{accuracy_2}\n')
    else:
        for i in range(combo_index, len(available_sessions) - 1):
            for combo in itertools.combinations(available_sessions, i + 1):
                test_sessions = list(combo)

                for session_to_add in list(set(available_sessions)
                                           - set(test_sessions)):
                    print(f'Testing with sessions: {test_sessions}')
                    print(f'Adding sessions: {session_to_add}')

                    phrases_to_add = sessions_to_phrases(sessions, [session_to_add])
                    phrases_to_test = sessions_to_phrases(sessions, test_sessions)

                    # do the first test if not done already
                    if tuple(test_sessions) in tested_sessions:
                        accuracy_1 = tested_sessions[tuple(test_sessions)]
                    else:
                        accuracy_1, results_1 = test_phrase_set(phrases_to_test,
                                                                videos_path,
                                                                phrases)
                        tested_sessions[tuple(test_sessions)] = accuracy_1

                    print(f'Before adding sessions {session_to_add}: {accuracy_1}')

                    # add the sessions to add to the default list before testing again
                    default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)
                    copy_default_lst = copy.deepcopy(default_lst)
                    for phrase_id, videos in phrases_to_add.items():
                        for video in videos:
                            full_video_path = os.path.join(videos_path, video)
                            with open(full_video_path, 'rb') as f:
                                feature_matrix = run_cfe(f)
                                pre_processed_matrix = \
                                    pre_process_signals([feature_matrix],
                                                        **dtw_params)[0]

                                actual_phrase_content = phrases[str(phrase_id)]
                                for i, phrase in enumerate(copy_default_lst.phrases):
                                    if phrase.content == actual_phrase_content:
                                        pava_template = PAVATemplate(
                                            blob=pre_processed_matrix
                                        )
                                        phrase.templates.append(pava_template)
                                        copy_default_lst.phrases[i] = phrase

                    write_pickle_file(copy_default_lst,
                                      configuration.DEFAULT_LIST_PATH)

                    # do the second test
                    accuracy_2, results_2 = test_phrase_set(phrases_to_test,
                                                            videos_path, phrases)
                    print(f'After adding sessions {session_to_add}: {accuracy_2}\n')

                    # make sure to write the original lst back to file
                    write_pickle_file(default_lst, configuration.DEFAULT_LIST_PATH)

                    # save the results
                    with open('full_update_sessions.csv', 'a') as f:
                        f.write(f'{test_sessions},{session_to_add},'
                                f'{accuracy_1},{accuracy_2}\n')


def analyse_full_update_sessions(**kwargs):
    file_path = kwargs['file_path']
    uses_test_sessions = kwargs['uses_test_sessions']

    if uses_test_sessions:
        data = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                test_sessions, added_sessions, accuracy_before, accuracy_after = \
                    re.match(ANALYSE_FULL_UPDATE_REGEX_2, line).groups()
                test_sessions = ast.literal_eval(test_sessions)
                added_sessions = ast.literal_eval(added_sessions)
                data.append({
                    'test_sessions': test_sessions,
                    'added_sessions': added_sessions,
                    'accuracy_before': float(accuracy_before),
                    'accuracy_after': float(accuracy_after)
                })
            df = pd.DataFrame(data)
            print(df)

            unique_added_sessions = \
                [int(s) for s in kwargs['unique_added_sessions']]

            session_performances = {}
            for s in unique_added_sessions:
                for index, row in df.iterrows():
                    added_sessions = row['added_sessions']
                    if s in added_sessions:
                        accuracy_before = row['accuracy_before']
                        accuracy_after = row['accuracy_after']
                        percentage_difference = accuracy_after - \
                            accuracy_before
                        current_performances = session_performances.get(s, [])
                        current_performances.append(percentage_difference)
                        session_performances[s] = current_performances

            for s, performances in session_performances.items():
                session_performances[s] = sum(performances) / len(performances)

            print(session_performances)

            session_performances = {}
            for index_1, row_1 in df.iterrows():
                added_sessions = row_1['added_sessions']
                previous_sessions = added_sessions[:-1]
                last_session = added_sessions[-1]
                accuracy_after_1 = row_1['accuracy_after']

                if not previous_sessions:
                    continue

                rows = []
                for index_2, row_2 in df.iterrows():
                    if row_2['added_sessions'] == previous_sessions:
                        rows.append(row_2)

                for row in rows:
                    accuracy_after_2 = row['accuracy_after']
                    percentage_difference = accuracy_after_1 - accuracy_after_2

                    current_performances = \
                        session_performances.get(last_session, [])
                    current_performances.append(percentage_difference)
                    session_performances[last_session] = current_performances

            for s, performances in session_performances.items():
                session_performances[s] = sum(performances) / len(performances)

            print(session_performances)
    else:
        data = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                test_sessions, session_to_add, accuracy_before, accuracy_after = \
                    re.match(ANALYSE_FULL_UPDATE_REGEX, line).groups()
                test_sessions = ast.literal_eval(test_sessions)
                data.append({
                    'test_sessions': test_sessions,
                    'added_session': int(session_to_add),
                    'accuracy_before': float(accuracy_before),
                    'accuracy_after': float(accuracy_after)
                })
            df = pd.DataFrame(data)

            unique_added_sessions = sorted(df['added_session'].unique())
            unique_added_sessions = np.asarray(unique_added_sessions)

            accuracies_before, accuracies_after = [], []
            for session in unique_added_sessions:
                rows = df[df['added_session'] == session]
                av_accuracy_before = rows['accuracy_before'].mean()
                av_accuracy_after = rows['accuracy_after'].mean()
                accuracies_before.append(av_accuracy_before)
                accuracies_after.append(av_accuracy_after)

            bar_width = 0.2
            plt.bar(unique_added_sessions, accuracies_before, width=bar_width,
                    label='Before Session Added')
            plt.bar(unique_added_sessions + bar_width, accuracies_after,
                    width=bar_width, label='After Session Added')
            plt.xticks(unique_added_sessions + bar_width, unique_added_sessions)
            plt.legend()
            plt.ylabel('Average acc. (%)')
            plt.xlabel('Session to be added')
            plt.title('Average accuracy before/after sessions added')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser1 = sub_parsers.add_parser('partial_update_phrases')
    parser1.add_argument('videos_path', type=str)

    parser2 = sub_parsers.add_parser('full_update_sessions')
    parser2.add_argument('videos_path', type=str)
    parser2.add_argument('--combo_index', default=0, type=int)
    parser2.add_argument('--test_sessions', type=list, default=None)

    parser3 = sub_parsers.add_parser('analyse_full_update_sessions')
    parser3.add_argument('file_path', type=str)
    parser3.add_argument('--uses_test_sessions', type=bool, default=False)
    parser3.add_argument('--unique_added_sessions', type=list, default=None)

    f = {
        'partial_update_phrases': partial_phrase_update,
        'full_update_sessions': full_update_sessions,
        'analyse_full_update_sessions': analyse_full_update_sessions
    }

    args = parser.parse_args()
    if args.run_type in list(f.keys()):
        f[args.run_type](**args.__dict__)
    else:
        parser.print_help()
