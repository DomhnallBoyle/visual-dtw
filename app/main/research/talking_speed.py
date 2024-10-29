import argparse

import cv2
import numpy as np
import sys

sys.path.append('/home/domhnall/Repos/EnlightenGAN')

from extract_mouth_region import get_jaw_roi_dnn, \
    get_dnn_face_detector_and_facial_predictor
from main.research.process_videos import get_video_rotation, fix_frame_rotation

_detector, _predictor = get_dnn_face_detector_and_facial_predictor()


def dense_optical_flow(previous_frame, next_frame):
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    # HSV - hue, saturation, value
    hsv = np.zeros_like(previous_frame)

    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(next_frame, cv2.COLOR_BGR2HSV)[:, :, 1]

    optical_flow = \
        cv2.calcOpticalFlowFarneback(prev=previous_frame_gray,
                                     next=next_frame_gray, flow=None,
                                     pyr_scale=0.5, levels=1, winsize=15,
                                     iterations=2, poly_n=5, poly_sigma=1.3,
                                     flags=0)

    # convert cartesian to polar
    magnitude, angle = cv2.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

    # hue corresponds to direction
    hsv[:, :, 0] = angle * (180 / np.pi / 2)

    # value corresponds to magnitude - normalise between 0 and 255
    hsv[:, :, 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.float32)

    # convert back to RGB for the network
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def process_video(args):
    video_capture = cv2.VideoCapture(args.video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    rotation = get_video_rotation(args.video_path)

    previous_frame = video_capture.read()[1]
    previous_frame = fix_frame_rotation(previous_frame, rotation)

    while True:
        success, next_frame = video_capture.read()
        if not success:
            break

        next_frame = fix_frame_rotation(next_frame, rotation)

        optical_flow_frame = dense_optical_flow(
            previous_frame=previous_frame,
            next_frame=next_frame
        )

        roi = get_jaw_roi_dnn(next_frame, 'mouth', _detector, _predictor)
        if not roi:
            continue
        roi, roi_x1, roi_y1, roi_x2, roi_y2 = roi
        optical_flow_roi = optical_flow_frame[roi_y1:roi_y2, roi_x1:roi_x2]

        cv2.imshow('Original', next_frame)
        cv2.imshow('Original ROI', roi)
        cv2.imshow('Optical Flow ROI', optical_flow_roi)
        if cv2.waitKey(fps) & 0xFF == ord('q'):
            break

        previous_frame = next_frame

    video_capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('process_video')
    parser_1.add_argument('video_path')
    parser_1.add_argument('--debug', default=False, type=bool)

    functionality = {
        'process_video': process_video
    }

    args = parser.parse_args()

    if args.run_type not in functionality:
        parser.print_usage()
        exit(1)

    functionality[args.run_type](args)
