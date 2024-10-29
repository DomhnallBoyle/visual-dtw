"""
Get sessions from sravi dataset for experiments
"""
import os
import random
import re
import shutil

from main import configuration
from main.utils.db import find_phrase_mappings, invert_phrase_mappings
from main.utils.io import read_json_file

SRAVI_USERS = [1, 2, 3, 4, 5, 6, 7, 17]
SRAVI_DATASET_PATH = '/home/domhnall/Documents/sravi_dataset/liopa/'
PAVA_DATASET_PATH = os.path.join(SRAVI_DATASET_PATH, 'pava')
SRAVI_REGEX = r'(\d+)_(S\w?[A-Z])(\d+)_S(\d+)'
NUM_SESSIONS_TO_PICK = 10


def main():
    phrases = read_json_file(configuration.PHRASES_PATH)
    pava_phrases = phrases['PAVA-DEFAULT']

    phrase_mappings = find_phrase_mappings('PAVA-DEFAULT')
    phrase_mappings = {k.replace('PAVA-DEFAULT', ''): v
                       for k, v in phrase_mappings.items()}
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)

    for sravi_user in SRAVI_USERS:
        sravi_user_dataset_path = os.path.join(SRAVI_DATASET_PATH,
                                              str(sravi_user))

        user_videos = [v for v in os.listdir(sravi_user_dataset_path)
                       if v.endswith('.mp4')]

        user_phrases = {str(i): [] for i in range(1, len(pava_phrases) + 1)}

        for video in user_videos:
            user_id, phrase_set, phrase_id, session_id = \
                re.match(SRAVI_REGEX, video).groups()

            sravi_phrase_id = phrase_set + str(int(phrase_id))
            try:
                user_phrases[inverse_phrase_mappings[sravi_phrase_id]]\
                    .append(video)
            except KeyError:
                continue

        if any(len(videos) == 0 for videos in user_phrases.values()):
            continue

        phrase_lengths = {k: len(v) for k, v in user_phrases.items()}

        sessions = []
        while len(sessions) != NUM_SESSIONS_TO_PICK:
            session = {}
            for pava_phrase_id, videos in user_phrases.items():
                random.shuffle(videos)

                if phrase_lengths[pava_phrase_id] < NUM_SESSIONS_TO_PICK:
                    phrase_video = random.choice(videos)
                else:
                    phrase_video = videos.pop()

                session[pava_phrase_id] = phrase_video

            sessions.append(session)

        pava_user_dataset_path = os.path.join(PAVA_DATASET_PATH,
                                              str(sravi_user))
        if os.path.exists(pava_user_dataset_path):
            shutil.rmtree(pava_user_dataset_path)
        os.makedirs(pava_user_dataset_path)

        if not os.path.exists(pava_user_dataset_path):
            os.makedirs(pava_user_dataset_path)

        for i, session in enumerate(sessions):
            i = str(i + 1)
            for phrase_number, video in session.items():
                original_video_path = os.path.join(sravi_user_dataset_path,
                                                   video)

                user_id, phrase_set, phrase_id, session_id = \
                    re.match(SRAVI_REGEX, video).groups()
                copy_video_name = \
                    f'PV{user_id.zfill(3)}_' \
                    f'P1{phrase_number.zfill(4)}_' \
                    f'S{i.zfill(3)}.mp4'
                copy_video_path = os.path.join(pava_user_dataset_path,
                                               copy_video_name)

                shutil.copyfile(original_video_path, copy_video_path)


if __name__ == '__main__':
    main()
