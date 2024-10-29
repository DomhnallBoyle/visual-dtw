import itertools
import os
import random
import re

import numpy as np
from main import configuration
from main.models import Config
from main.research.test_full_update import save_default_list_to_file, \
    save_default_as_sessions, get_number_of_sessions, \
    train_test_session_split, create_template
from main.services.transcribe import transcribe_signal
from main.utils.io import read_json_file, read_pickle_file, write_pickle_file
from main.utils.exceptions import InaccuratePredictionException

NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'


def get_training_sessions(videos_path):
    """Split videos into training and test"""
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    # split unseen user templates into sessions
    sessions_to_add = {}
    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        video = os.path.join(videos_path, video)

        phrase = pava_phrases[phrase_id]
        session_videos = sessions_to_add.get(int(session_id), [])
        session_videos.append((phrase, video))
        sessions_to_add[int(session_id)] = session_videos

    return sessions_to_add


def tests(ref_signals, test_templates, dtw_params):
    num_tests = 0
    accuracy = 0
    for actual_label, test_template in test_templates:
        try:
            predictions = transcribe_signal(ref_signals,
                                            test_template.blob,
                                            None, **dtw_params)
        except Exception as e:
            continue

        # consider only rank 1 accuracy
        top_prediction = predictions[0]
        prediction_label = top_prediction['label']
        if prediction_label == actual_label:
            accuracy += 1

        num_tests += 1

    accuracy /= num_tests
    accuracy *= 100

    return accuracy


def find_top_sessions(sessions, n=5):
    if len(sessions) == n:
        return sessions

    session_keys = [session[0] for session in sessions]
    dtw_params = Config().__dict__
    best_sessions = None
    best_accuracy = 0

    for training_sessions in list(itertools.combinations(sessions, n)):
        training_sessions = list(training_sessions)
        training_keys = [session[0] for session in training_sessions]
        testing_keys = list(set(session_keys) - set(training_keys))

        test_sessions = []
        for test_key in testing_keys:
            for session_key, session in sessions:
                if test_key == session_key:
                    test_sessions.append((session_key, session))

        # print(training_keys, testing_keys)
        # print(training_sessions)
        # print(test_sessions)
        # print()

        ref_signals = [(label, template.blob)
                       for session_key, session in training_sessions
                       for label, template in session]

        test_templates = [(label, template)
                          for session_key, session in test_sessions
                          for label, template in session]

        accuracy = tests(ref_signals, test_templates, dtw_params)

        # keep best sessions
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_sessions = training_sessions

    return best_sessions


def get_accuracy(sessions, test_videos):
    ref_signals = [(label, template.blob)
                   for session_key, session in sessions
                   for label, template in session]
    dtw_params = Config().__dict__

    return tests(ref_signals, test_videos, dtw_params)


def get_saved_sessions():
    return read_pickle_file('all_sessions.pkl')


def updating_default_list(videos_path):
    save_default_list_to_file()
    save_default_as_sessions()

    sessions_to_add_path = 'sessions_to_add.pkl'

    # the idea is to find the worst session using LOOCV for finding best 5
    # best sessions to use
    max_num_sessions = 5
    worst_session = 6
    leave_out_session = 7

    sessions_to_add, test_videos = train_test_session_split(videos_path,
                                                            worst_session)
    default_sessions = get_saved_sessions()
    leave_out_session = sessions_to_add.pop(leave_out_session)

    # shuffle the sessions to add
    keys = list(sessions_to_add.keys())
    random.shuffle(keys)
    sessions_to_add = [(key, sessions_to_add[key]) for key in keys]

    print('Loading user sessions')
    # to speed up the program
    if os.path.exists(sessions_to_add_path):
        sessions_to_add = read_pickle_file(sessions_to_add_path)
    else:
        # convert sessions to templates
        for session_key, session in sessions_to_add:
            for i, (label, video_path) in enumerate(session):
                template = create_template(video_path)
                session[i] = (label, template)

        # save them for later
        write_pickle_file(sessions_to_add, sessions_to_add_path)

    print('Loading test videos')
    for i, (actual_label, video_path) in enumerate(test_videos):
        template = create_template(video_path)
        test_videos[i] = (actual_label, template)

    num_sessions = get_number_of_sessions()
    if num_sessions < max_num_sessions:
        pass
    else:
        print('Finding top user sessions')
        best_5_user_sessions = find_top_sessions(sessions=sessions_to_add)

        user_accuracy = get_accuracy(sessions=best_5_user_sessions,
                                     test_videos=test_videos)
        default_accuracy = get_accuracy(sessions=default_sessions,
                                        test_videos=test_videos)
        print(user_accuracy, default_accuracy)

        if user_accuracy >= default_accuracy:
            # use best 5 from user's sessions
            sessions_to_use = 'best_5_user_sessions'
            best_5_max_match_sessions = None
        else:
            print('Finding top mix and match sessions')
            # combine user and default sessions
            mix_match_sessions = default_sessions + sessions_to_add
            best_5_max_match_sessions = find_top_sessions(mix_match_sessions)

            mix_match_accuracy = get_accuracy(best_5_max_match_sessions,
                                              test_videos=test_videos)
            print(mix_match_accuracy)
            if mix_match_accuracy >= default_accuracy:
                # use best 5 mix-and-match sessions
                sessions_to_use = 'mix_match_sessions'
            else:
                # revert to default sessions
                sessions_to_use = 'default_sessions'

        print('Testing against left out session')
        #  TODO: This can be removed after
        # now do final test with leave out session vs only user sessions, only
        # default sessions and mix-and-match sessions
        for i, (actual_label, video_path) in enumerate(leave_out_session):
            template = create_template(video_path)
            leave_out_session[i] = (actual_label, template)

        default_accuracy = get_accuracy(default_sessions, leave_out_session)
        user_accuracy = get_accuracy(best_5_user_sessions, leave_out_session)
        if not best_5_max_match_sessions:
            mix_match_sessions = default_sessions + sessions_to_add
            best_5_max_match_sessions = find_top_sessions(mix_match_sessions)
        mix_match_accuracy = get_accuracy(best_5_max_match_sessions,
                                          leave_out_session)

        print(sessions_to_use, [default_accuracy, user_accuracy,
                                mix_match_accuracy])

        # when testing with domhnall sessions
        # User accuracy = 94.73684210526315
        # Default accuracy = 83.33333333333334
        # didn't get to complete the mix match because took too long to find
        # best 5 of 20 sessions


def updating_new_list():
    pass


def main():
    videos_path = '/home/domhnall/Documents/sravi_dataset/liopa/12'
    updating_default_list(videos_path)


if __name__ == '__main__':
    main()
