import argparse
import io
import os
import requests

from main import configuration
from main.models import Config, PAVAList, PAVAUser
from main.utils.timer import Timer

URL = 'http://{host}/pava/api/v1/lists/{list_id}/transcribe/video'
VIDEO_PATH = os.path.join(configuration.VIDEOS_PATH, '001_S2F0020_S004.mp4')

timer = Timer()


def time_transcribe_endpoint(host, list_id, num_hits):
    if not list_id:
        user = PAVAUser.create(default_list=True, config=Config())
        lst = PAVAList.get(filter=(PAVAList.user_id == user.id), first=True)
        list_id = lst.id

    while num_hits != 0:
        with open(VIDEO_PATH, 'rb') as f:
            files = {
                'file': ('001_S2F0020_S004.mp4', io.BytesIO(f.read()))
            }

        with timer.time('Transcribe'):
            requests.post(URL.format(host=host, list_id=list_id), files=files)

        num_hits -= 1

    timer.analyse()


def main(args):
    time_transcribe_endpoint(args.host, args.list_id, args.num_hits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0:5000')
    parser.add_argument('--list_id', default=None)
    parser.add_argument('--num_hits', default=100, type=int)

    main(parser.parse_args())
