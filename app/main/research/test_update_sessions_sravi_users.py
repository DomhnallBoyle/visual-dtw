import argparse
import os
import random

import cv2
import pandas as pd
from main import configuration
from main.models import Config
from main.services.transcribe import transcribe_signal
from main.research.cmc import CMC
from main.research.process_videos import get_video_rotation, fix_frame_rotation
from main.research.test_full_update import create_template
from main.research.test_update_list_3 import get_default_sessions
from main.research.test_update_list_6 import cross_validation as algorithm_1
from main.utils.io import read_json_file
from vidaug import augmentors as va


def test_data_vs_sessions(test_data, sessions):
    dtw_params = Config().__dict__

    cmc = CMC(num_ranks=3)

    test_signals = [(label, template.blob)
                    for label, template in test_data]

    ref_signals = [(label, template.blob)
                   for session_label, ref_session in sessions
                   for label, template in ref_session]

    for actual_label, test_signal in test_signals:
        try:
            predictions = transcribe_signal(ref_signals, test_signal, None,
                                            **dtw_params)
        except Exception as e:
            continue

        predictions = [p['label'] for p in predictions]
        cmc.tally(predictions, actual_label)

    cmc.calculate_accuracies(num_tests=len(test_data), count_check=False)

    return cmc.all_rank_accuracies[0]


def random_video_augmentation(video_path, debug):
    # randomly apply augmentation to video frames
    print('Augmenting video...')
    video_reader = cv2.VideoCapture(video_path)
    width, height = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                    int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    rotation = get_video_rotation(video_path)

    frames = []
    while True:
        success, frame = video_reader.read()
        if not success:
            break
        frame = fix_frame_rotation(frame, rotation)
        frames.append(frame)
    video_reader.release()

    sometimes = lambda aug: va.Sometimes(0.6, aug)
    augmentation_sequence = va.Sequential([
        sometimes(va.HorizontalFlip()),  # flip horizontally at random
        sometimes(va.RandomRotate(degrees=10)),  # rotate randomly between -10 and 10
        sometimes(va.Downsample(ratio=0.6)),  # downsample (delete frames) randomly (removes 40% frames)
        sometimes(va.Add(value=random.randint(-15, 15))),  # add pixel intensity to videos
        sometimes(va.GaussianBlur(sigma=random.random())),  # random gaussian blur
    ])
    augmented_frames = augmentation_sequence(frames)

    new_video_path = '/home/domhnall/Desktop/augmented_video.mp4'
    video_writer = cv2.VideoWriter(new_video_path,
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, (width, height))
    for frame in augmented_frames:
        video_writer.write(frame)

        if debug:
            cv2.imshow('Augmented Video', frame)
            cv2.waitKey(fps)

    video_writer.release()

    cv2.destroyAllWindows()

    return new_video_path


def construct_user_sessions(videos_directory, num_sessions,
                            corrupt_non_unique, num_phrases_to_use=None,
                            debug=False):
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']
    if not num_phrases_to_use:
        num_phrases_to_use = len(pava_phrases)

    groundtruth = pd.read_csv(
        os.path.join(videos_directory, 'validated_groundtruth.csv'),
        header=0
    )

    d_phrase_video_paths = {}
    for index, row in groundtruth.iterrows():
        video_path = os.path.join(videos_directory, row['Name'])

        try:
            assert row['Actual'] in pava_phrases.values()
        except AssertionError:
            continue

        phrase_video_paths = d_phrase_video_paths.get(row['Actual'], [])
        phrase_video_paths.append(video_path)
        d_phrase_video_paths[row['Actual']] = phrase_video_paths

    print('Num unique phrases:', len(d_phrase_video_paths))
    print({
        phrase: len(video_paths)
        for phrase, video_paths in d_phrase_video_paths.items()
    })

    assert len(d_phrase_video_paths) == num_phrases_to_use, \
        f'Only {len(d_phrase_video_paths)} phrases available...' \
        f'need {num_phrases_to_use}'

    num_sessions_to_construct = num_sessions
    user_sessions = []
    num_augmented_videos = 0
    for i in range(num_sessions_to_construct):
        session = []
        for phrase, video_paths in d_phrase_video_paths.items():
            augmented = False
            if len(video_paths) == 1:
                video_path = video_paths[0]
                if corrupt_non_unique:
                    video_path = random_video_augmentation(video_path, debug)
                    augmented = True
            else:
                video_path = video_paths.pop(0)

            template = create_template(video_path)
            if not template:
                if augmented:
                    while True:
                        video_path = random_video_augmentation(video_path,
                                                               debug)
                        template = create_template(video_path)
                        if template:
                            break
                else:
                    continue

            if augmented:
                num_augmented_videos += 1

            session.append((phrase, template))

        assert len(session) == num_phrases_to_use
        user_sessions.append((f'added_{i+1}', session))

    print('Session structure:')
    for s in user_sessions:
        print(s[0], len(s[1]))

    print('Num augmented videos:', num_augmented_videos)

    return user_sessions, num_augmented_videos


def experiment(args):
    user_id = args.videos_directory.split('/')[-1]
    if not user_id:
        user_id = args.videos_directory.split('/')[-2]

    default_sessions = get_default_sessions()

    for repeat in range(1, args.num_repeats + 1):
        user_sessions = construct_user_sessions(
            args.videos_directory,
            args.num_sessions,
            args.corrupt_non_unique
        )
        half_index = len(user_sessions) // 2
        print('Num user sessions:', len(user_sessions))
        print('Half index:', half_index)

        # extract sessions to add and train/test sessions
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
        print('Training/test data:', len(train_test_data))
        print('Training data:', len(training_data))
        print('Test data:', len(test_data))
        assert len(training_data) + len(test_data) == len(train_test_data)

        # get default accuracy
        default_accuracies = test_data_vs_sessions(test_data, default_sessions)
        print('Default Accuracy:', default_accuracies)

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
            mix_accuracies = test_data_vs_sessions(test_data, mix)
            print('Mix Accuracy:', mix_accuracies)
            print('Average time:', average_time)

            mix_labels = [session_label
                          for session_label, session_templates in mix]

            result.extend([mix_accuracies, average_time, mix_labels])
            results.append(result)

        with open('update_default_list_12.csv', 'a') as f:
            line = f'{user_id},{repeat},{args.k},' \
                   f'{len(training_data)},{len(test_data)},' \
                   f'{default_accuracies},{results}\n'
            f.write(line)


def testing(args):
    default_sessions = get_default_sessions()

    for repeat in range(1, args.num_repeats + 1):
        user_sessions, num_augmented_videos = construct_user_sessions(
            args.videos_directory,
            args.num_sessions,
            args.corrupt_non_unique,
            args.num_phrases_to_use,
            args.debug
        )
        half_index = len(user_sessions) // 2
        print('Num user sessions:', len(user_sessions))
        print('Half index:', half_index)

        # extract sessions to add and train/test sessions
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
        print('Training/test data:', len(train_test_data))
        print('Training data:', len(training_data))
        print('Test data:', len(test_data))
        assert len(training_data) + len(test_data) == len(train_test_data)

        # get default accuracy
        default_accuracies = test_data_vs_sessions(test_data, default_sessions)
        print('Default Accuracy:', default_accuracies)

        # cross validation with forward session selection
        mix, average_time = algorithm_1(training_data, sessions_to_add,
                                        default_sessions, k=args.k,
                                        multiple_max_test=True)

        mix_accuracies = test_data_vs_sessions(test_data, mix)
        print('Mix Accuracy:', mix_accuracies)
        print('Average time:', average_time)

        mix_labels = [session_label
                      for session_label, session_templates in mix]

        with open('update_default_list_13.csv', 'a') as f:
            line = f'{args.user_id},{repeat},{args.k},' \
                   f'{len(training_data)},{len(test_data)},{num_augmented_videos},' \
                   f'{default_accuracies},{mix_accuracies},{mix_labels}\n'
            f.write(line)


def main(args):
    method = {
        'experiment': experiment,
        'testing': testing
    }.get(args.run_type)

    if method:
        method(args)
    else:
        print('Incorrect args')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_type', choices=['experiment', 'testing'])
    parser.add_argument('videos_directory')
    parser.add_argument('--num_repeats', type=int, default=5)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--num_sessions', type=int, default=10)
    parser.add_argument('--corrupt_non_unique', type=bool, default=False)
    parser.add_argument('--num_phrases_to_use', type=int)
    parser.add_argument('--user_id', default='User')
    parser.add_argument('--debug', default=False)

    main(parser.parse_args())
