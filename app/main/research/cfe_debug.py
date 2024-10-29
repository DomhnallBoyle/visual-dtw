"""
CFE debug a single video or directory of videos
"""
import argparse
import glob
import io
import os
import zipfile
from http import HTTPStatus

import cv2
import requests
from main import configuration


def main(args):
    if os.path.isdir(args.path):
        video_paths = glob.glob(os.path.join(args.path, '*.mp4'))
    else:
        video_paths = [args.path]

    for video_path in video_paths:
        with open(video_path, 'rb') as f:
            response = requests.post(configuration.CFE_URL, files={
                'video': io.BytesIO(f.read())
            })
            if response.status_code == HTTPStatus.OK:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    zip_file_names = zip_file.namelist()
                    debug_file = list(filter(lambda x: 'debug.avi' in x,
                                             zip_file_names))[0]
                    zip_file.extract(debug_file)

                    video_capture = cv2.VideoCapture(debug_file)
                    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

                    while True:
                        success, frame = video_capture.read()
                        if not success:
                            break

                        cv2.imshow('Frame', frame)
                        cv2.waitKey(fps)

                    video_capture.release()
                    os.remove(debug_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path')

    main(parser.parse_args())
