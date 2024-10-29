"""
Use video augmentation on single session - convert to multiple sessions
Use leave out test set
"""
import argparse
import glob
import os
import random
import re

import cv2
from main import configuration
from main.research.process_videos import get_video_rotation, fix_frame_rotation
from main.research.test_update_list_3 import get_user_sessions, \
    NEW_RECORDINGS_REGEX
from main.research.videos_vs_videos import create_template, get_accuracy, \
    get_templates
from main.utils.io import read_json_file
from vidaug import augmentors as va


def get_user_sessions(videos_path):
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    # split unseen user templates into sessions
    user_sessions = {}
    for video in os.listdir(videos_path):
        if not video.endswith('.mp4'):
            continue

        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        video = os.path.join(videos_path, video)
        template = create_template(video)
        if not template:
            continue

        phrase = pava_phrases[phrase_id]
        session_videos = user_sessions.get(int(session_id), [])
        session_videos.append((phrase, template, video))
        user_sessions[int(session_id)] = session_videos

    user_sessions = [(f'added_{k}', v) for k, v in user_sessions.items()]

    return user_sessions


def augment_video(video_path, debug=False):
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
        sometimes(va.RandomRotate(degrees=10)),
        # rotate randomly between -10 and 10
        sometimes(va.Downsample(ratio=0.6)),
        # downsample (delete frames) randomly (removes 40% frames)
        sometimes(va.Add(value=random.randint(-15, 15))),
        # add pixel intensity to videos
        sometimes(va.GaussianBlur(sigma=random.random())),
        # random gaussian blur
    ])
    augmented_frames = augmentation_sequence(frames)

    new_video_path = '/home/domhnall/Desktop/augmented_video.mp4'
    video_writer = cv2.VideoWriter(new_video_path,
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, (height, width))
    for frame in augmented_frames:
        video_writer.write(frame)

        if debug:
            cv2.imshow('Augmented Video', frame)
            cv2.waitKey(fps)

    video_writer.release()

    if debug:
        cv2.destroyAllWindows()

    return new_video_path


def augment_session(session, debug=False):
    original_session_length = len(session)

    while True:
        new_session = []
        for label, template, video in session:
            new_video_path = augment_video(video, debug=debug)
            template = create_template(new_video_path)
            if not template:
                break
            new_session.append((label, template, new_video_path))

        if len(new_session) == original_session_length:
            break

    return 'augmented_session', new_session


def main(args):
    user_sessions = get_user_sessions(args.ref_videos_directory)
    random.shuffle(user_sessions)

    test_video_paths = \
        glob.glob(os.path.join(args.test_videos_directory, '*.mp4'))
    test_templates = get_templates(test_video_paths, None)

    starting_session = user_sessions[0]
    sessions_to_add = user_sessions[1:]

    num_iterations = args.num_iterations
    if not num_iterations:
        num_iterations = len(sessions_to_add)
    assert num_iterations <= len(sessions_to_add)

    print('Num test templates:', len(test_templates))
    print('Num iterations:', num_iterations)

    normal_sessions, replicated_sessions, augmented_sessions = [], [], []
    normal_accuracies, replicated_accuracies, augmented_accuracies = [], [], []

    to_ref_signals = lambda sessions: [(label, template.blob)
                                       for session_label, session in sessions
                                       for label, template, video in session]

    for i in range(num_iterations):
        normal_sessions.append(sessions_to_add.pop(0))
        replicated_sessions.append(starting_session)
        augmented_sessions.append(augment_session(starting_session[1],
                                                  args.debug))

        normal_accuracy = get_accuracy(to_ref_signals(normal_sessions),
                                       test_templates)[0][0]
        replicated_accuracy = get_accuracy(to_ref_signals(replicated_sessions),
                                           test_templates)[0][0]
        augmented_accuracy = get_accuracy(to_ref_signals(augmented_sessions),
                                          test_templates)[0][0]

        normal_accuracies.append(normal_accuracy)
        replicated_accuracies.append(replicated_accuracy)
        augmented_accuracies.append(augmented_accuracy)

    print(normal_accuracies)
    print(replicated_accuracies)
    print(augmented_accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_videos_directory')
    parser.add_argument('test_videos_directory')
    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--debug', default=False)

    main(parser.parse_args())
