"""
Processing frames individually and saving
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import cv2

# enlighten gan path
sys.path.append('/home/domhnall/Repos/EnlightenGAN')


def naive(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img


def egan(image):
    pass


def clahe(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image_hsv[:, :, 2] = clahe.apply(image_hsv[:, :, 2])

    return cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)


def get_video_rotation(video_path, debug=False):
    cmd = f'ffmpeg -i {video_path}'

    p = subprocess.Popen(
        cmd.split(' '),
        stderr=subprocess.PIPE,
        close_fds=True
    )
    stdout, stderr = p.communicate()

    try:
        reo_rotation = re.compile('rotate\s+:\s(\d+)')
        match_rotation = reo_rotation.search(str(stderr))
        rotation = match_rotation.groups()[0]
    except AttributeError:
        if debug:
            print(f'Rotation not found: {video_path}')
        return 0

    return int(rotation)


def fix_frame_rotation(image, rotation):
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotation == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def process_video(videos_directory, output_directory_name, process_function,
                  debug=False, save=False):
    output_directory = os.path.join(videos_directory, output_directory_name)

    if save:
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.mkdir(output_directory)

    video_paths = [os.path.basename(x)
                   for x in glob.glob(os.path.join(videos_directory, '*.mp4'))]
    for video_path in video_paths:
        input_video_path = os.path.join(videos_directory, video_path)
        output_video_path = os.path.join(output_directory, video_path)

        video_reader = cv2.VideoCapture(input_video_path)
        width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_reader.get(cv2.CAP_PROP_FPS))
        rotation = get_video_rotation(input_video_path)

        if save:
            video_writer = cv2.VideoWriter(output_video_path,
                                           cv2.VideoWriter_fourcc(*'MJPG'),
                                           fps, (height, width))

        while True:
            success, frame = video_reader.read()
            if not success:
                break

            frame = fix_frame_rotation(frame, rotation)
            frame_copy = frame.copy()
            if process_function:
                frame = process_function(frame)

            if debug:
                cv2.imshow('Original frame', frame_copy)
                cv2.imshow('New frame', frame)
                cv2.waitKey(fps)

            if save:
                video_writer.write(frame)

        video_reader.release()

        if save:
            video_writer.release()


def main(args):
    process_functions = {
        'naive': naive,
        'egan': egan,
        'clahe': clahe
    }
    process_function = process_functions.get(args.process_function)

    process_video(args.videos_directory, args.output_directory_name,
                  process_function, args.debug, args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')
    parser.add_argument('output_directory_name')
    parser.add_argument('--process_function')
    parser.add_argument('--debug', default=False)
    parser.add_argument('--save', default=False)

    main(parser.parse_args())
