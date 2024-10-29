"""Take the default templates and retrieve their respective videos"""

import os
import re
import shutil

from main import configuration

VIDEO_PATH = '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa'


def main():
    with open(configuration.DEFAULT_TEMPLATES_PATH, 'r') as f:
        default_templates = f.read().split(',')

    phrase_sets = configuration.SRAVI_PHRASE_SETS

    output_directory = os.path.join(VIDEO_PATH, 'default_videos')
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.mkdir(output_directory)

    # find video paths
    template_regex = r'AE_norm_2_(\d+)_(.+)_(\d+)'
    for default_template in default_templates:
        user_id, phrase_id, session_id = re.match(template_regex,
                                                  default_template).groups()

        _phrase_set = None
        _phrase_set_id = None
        for phrase_set in phrase_sets[::-1]:
            if phrase_set in phrase_id:
                _phrase_set = phrase_set
                _phrase_set_id = phrase_id.replace(_phrase_set, '')
                break

        relative_video_path = f'00{user_id}_' \
                              f'{_phrase_set + _phrase_set_id.zfill(4)}_' \
                              f'S{session_id.zfill(3)}.mp4'
        video_path = os.path.join(VIDEO_PATH, user_id, relative_video_path)

        if not os.path.exists(video_path):
            print('Failed to find video:')
            print(user_id, _phrase_set, _phrase_set_id, session_id)
            print(video_path)
            print()

        # copy video path to a different directory
        copy_video_path = os.path.join(output_directory, relative_video_path)
        shutil.copyfile(video_path, copy_video_path)


if __name__ == '__main__':
    main()
