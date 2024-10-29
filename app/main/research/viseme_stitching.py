import argparse
import glob
import io
import os
import re
import random
import requests
import shutil
import subprocess
from http import HTTPStatus

import textgrid
from main.research.phrase_list_composition import VISEME_TO_PHONEME, \
    PHONEME_TO_VISEME
from main.utils.cam import Cam
from main.utils.io import read_pickle_file, write_pickle_file

# TODO: Rather than extracting visemes from phrase videos
#  just get them to record the visemes individually

NUM_SESSIONS = 5
DIRECTORY = os.path.abspath('viseme_stitching_data')

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

WORDS_TO_VISEMES = {}
with open('cmudict-en-us.dict', 'r') as f:
    for line in f.readlines():
        line = line.split(' ')
        word = line[0]

        phones = list(map(lambda phone: phone.lower().strip(), line[1:]))
        visemes = list(map(lambda phone: PHONEME_TO_VISEME[phone], phones))

        WORDS_TO_VISEMES[word] = visemes


def clip_video(video_path, textgrid_path):
    """Clip video based on the phoneme timestamps in the textgrid"""
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    phonemes = tg[0]
    phoneme_paths = {}
    regex = r'([A-Z]+)\d*'
    for i, phoneme in enumerate(phonemes):
        # print(phoneme.minTime, phoneme.maxTime, phoneme.mark)
        if phoneme.mark in ['', 'sil']:
            continue

        phoneme.mark = re.match(regex, phoneme.mark).groups()[0]

        output_video_path = os.path.join(
            os.path.abspath(DIRECTORY),
            os.path.basename(video_path).replace('.mp4',
                                                 f'_{i+1}_{phoneme.mark}.mp4')
        )

        paths = phoneme_paths.get(phoneme.mark, [])
        paths.append(output_video_path)
        phoneme_paths[phoneme.mark] = paths

        if os.path.exists(output_video_path):
            continue

        command = [
            'ffmpeg', '-y',
            '-i', f'{video_path}',
            '-force_key_frames',
            f'{float(phoneme.minTime):.2f},{float(phoneme.maxTime):.2f}',
            'temp.mp4'
        ]
        subprocess.call(command)

        command = [
            'ffmpeg', '-y',
            '-ss', f'{float(phoneme.minTime):.2f}',
            '-i', 'temp.mp4',
            '-t',
            f'{float(phoneme.maxTime) - float(phoneme.minTime):.2f}',
            f'{output_video_path}'
        ]
        subprocess.call(command)  # this is blocking

    return phoneme_paths


def extract_audio_from_video(video_path):
    audio_path = os.path.basename(video_path).replace('mp4', 'wav')
    audio_path = os.path.join(os.path.dirname(video_path), audio_path)
    command = f'ffmpeg -y -i {video_path} -ab 160k -ac 1 -ar 44100 -vn ' \
              f'{audio_path}'

    subprocess.call(command, shell=True)

    return audio_path


def extract_viseme_clips(args):
    """Extract viseme clips from the phonetic alignment of audio and
    transcript"""
    pass


def stitch_video_paths(phrase, video_paths):
    new_video_path = f'/home/domhnall/Desktop/{phrase}_stitched.mp4'
    tmp_txt_path = '/home/domhnall/Desktop/stitch_paths.txt'
    with open(tmp_txt_path, 'w') as f:
        for video_path in video_paths:
            f.write(f'file {video_path}\n')

    command = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        f'-i', f'{tmp_txt_path}',
        '-c', 'copy',
        f'{new_video_path}'
    ]
    subprocess.call(command)

    # os.remove(tmp_txt_path)

    return new_video_path


def record_visemes(args):
    cam = Cam(debug=True, playback=True, countdown=True)
    for i in range(args.num_sessions):
        for viseme in VISEME_TO_PHONEME.keys():
            if viseme == 'sp': continue
            print('Recording viseme:', viseme)
            save_path = os.path.join(DIRECTORY, f'{viseme}_{i+1}.mp4')
            cam.record(save_path=save_path)


def test(args):
    viseme_recording_paths = glob.glob(os.path.join(DIRECTORY, '*.mp4'))
    viseme_path_d = {}
    for viseme_recording_path in viseme_recording_paths:
        viseme, session = \
            re.match(r'(.+)_(\d+).mp4',
                     os.path.basename(viseme_recording_path)).groups()
        viseme_path_d[viseme] = viseme_recording_path
    print(viseme_path_d)

    phrase = 'What\'s the plan?'
    viseme_video_paths_combo = []
    for word in phrase.replace('?', '').split(' '):
        word = word.lower()
        visemes = WORDS_TO_VISEMES.get(word)

        for viseme in visemes:
            viseme_video_paths_combo.append(viseme_path_d[viseme])

    stitch_video_paths(phrase, viseme_video_paths_combo)


def main(args):
    f = {
        'record_visemes': record_visemes,
        'test': test
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('record_visemes')
    parser_1.add_argument('--num_sessions', type=int, default=1)

    parser_2 = sub_parsers.add_parser('test')

    main(parser.parse_args())
