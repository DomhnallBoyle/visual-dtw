import argparse
import glob
import os
import random
import re

from main import configuration
from main.models import PAVAPhrase, PAVATemplate
from main.research.test_update_list_2 import create_template
from main.research.test_update_list_3 import NEW_RECORDINGS_REGEX
from main.utils.db import db_session
from main.utils.io import read_json_file


def main(args):
    video_paths = glob.glob(os.path.join(args.videos_directory, '*.mp4'))
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    random.shuffle(video_paths)

    num_added = 0
    for video_path in video_paths:
        basename = os.path.basename(video_path)

        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, basename).groups()

        template = create_template(video_path)
        if not template:
            continue

        phrase = pava_phrases[phrase_id]

        with db_session() as s:
            phrase_id = PAVAPhrase.get(
                s, query=(PAVAPhrase.id,),
                filter=(
                    (PAVAPhrase.content == phrase)
                    & (PAVAPhrase.list_id == args.list_id)),
                first=True
            )[0]

        PAVATemplate.create(phrase_id=phrase_id, blob=template.blob)

        num_added += 1
        if num_added == args.num_templates_to_add:
            break

    print('List ID:', args.list_id)
    print('Num added templates:', num_added)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')
    parser.add_argument('list_id')
    parser.add_argument('--num_templates_to_add', type=int)

    main(parser.parse_args())
