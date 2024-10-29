"""
Contrast Enhancement of Brightness-Distorted Images by Improved Adaptive
Gamma Correction
https://arxiv.org/pdf/1709.04427v1.pdf
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from main.research.video_quality_gmm import get_video_rotation, \
    fix_frame_rotation

THRESHOLD = 0.3


def image_agcwd(img, a=0.25, truncated_cdf=False):
    # h, w = img.shape[:2]
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # cdf = hist.cumsum()
    # cdf_normalized = cdf / cdf.max()
    prob_normalized = hist / hist.sum()

    unique_intensity = np.unique(img)
    # intensity_max = unique_intensity.max()
    # intensity_min = unique_intensity.min()
    prob_min = prob_normalized.min()
    prob_max = prob_normalized.max()

    pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
    pn_temp[pn_temp > 0] = prob_max * (pn_temp[pn_temp > 0] ** a)
    pn_temp[pn_temp < 0] = prob_max * (-((-pn_temp[pn_temp < 0]) ** a))
    prob_normalized_wd = pn_temp / pn_temp.sum()  # normalize to [0,1]
    cdf_prob_normalized_wd = prob_normalized_wd.cumsum()

    if truncated_cdf:
        inverse_cdf = np.maximum(0.5, 1 - cdf_prob_normalized_wd)
    else:
        inverse_cdf = 1 - cdf_prob_normalized_wd

    img_new = img.copy()
    for i in unique_intensity:
        img_new[img == i] = np.round(255 * (i / 255) ** inverse_cdf[i])

    return img_new


def process_dimmed(img):
    agcwd = image_agcwd(img, a=0.75, truncated_cdf=True)

    return agcwd


def process_bright(img):
    img_negative = 255 - img
    agcwd = image_agcwd(img_negative, a=0.25, truncated_cdf=False)
    reversed = 255 - agcwd

    return reversed


def process_frame(frame):
    # extract intensity component of the image
    YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y = YCrCb[:, :, 0]

    # determine whether image is bright or dimmed
    exp_in = 112  # Expected global average intensity
    M, N = frame.shape[:2]
    mean_in = np.sum(Y / (M * N))
    t = (mean_in - exp_in) / exp_in

    if t < -THRESHOLD:
        # dimmed image
        result = process_dimmed(Y)
        YCrCb[:, :, 0] = result
        frame = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)
    elif t > THRESHOLD:
        # bright image
        result = process_bright(Y)
        YCrCb[:, :, 0] = result
        frame = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2BGR)

    return frame


def process_video(video_path, output_video_path, debug=False):
    video_reader = cv2.VideoCapture(video_path)
    width, height = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)), \
        int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    video_rotation = get_video_rotation(video_path)

    video_writer = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, (height, width))

    while True:
        success, frame = video_reader.read()
        if not success:
            break

        frame = fix_frame_rotation(frame, video_rotation)

        original = frame.copy()
        frame = process_frame(frame)

        video_writer.write(frame)

        if debug:
            cv2.imshow('Original', original)
            cv2.imshow('Improved', frame)
            cv2.waitKey(fps)

    video_reader.release()
    video_writer.release()


def main(args):
    file_type = args.file_type
    file_path = args.file_path

    if file_type == 'image':
        image = cv2.imread(file_path, 1)
        image = process_frame(image)
        cv2.imshow('Image', image)
    elif file_type == 'video':
        process_video(file_path, args.debug)
    elif file_type == 'directory':
        video_paths = glob.glob(os.path.join(file_path, '*.mp4'))

        output_directory = os.path.join(file_path, 'iagcwd')
        if os.path.exists(output_directory):
            shutil.rmtree(output_directory)
        os.mkdir(output_directory)

        for video_path in video_paths:
            output_video_path = os.path.join(output_directory,
                                             os.path.basename(video_path))
            process_video(video_path, output_video_path, args.debug)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_type')
    parser.add_argument('file_path')
    parser.add_argument('--debug', default=False)

    main(parser.parse_args())
