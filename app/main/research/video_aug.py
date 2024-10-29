import cv2
import random

import numpy as np
from main.research.process_videos import get_video_rotation, fix_frame_rotation


class VideoAug:

    def __init__(self, rotation_max=10, enlighten_max=20,
                 output_path='/tmp/video_aug.mp4',
                 fps=25,
                 threshold=0.5):
        self.rotation_max = rotation_max
        self.enlighten_max = enlighten_max
        self.output_path = output_path
        self.fps = fps
        self.threshold = threshold

    def process(self, video_path):
        to_flip = True if random.random() < self.threshold else False
        to_rotate = 0 if random.random() < self.threshold \
            else random.randint(-self.rotation_max, self.rotation_max)
        to_enlighten = 0 if random.random() < self.threshold \
            else random.randint(-self.enlighten_max, self.enlighten_max)
        to_down_sample = True if random.random() < self.threshold \
            else False
        to_up_sample = True \
            if not to_down_sample and random.random() < self.threshold \
            else False

        video_reader = cv2.VideoCapture(video_path)
        rotation = get_video_rotation(video_path)

        # gather frames and fix rotation
        frames = []
        while True:
            success, frame = video_reader.read()
            if not success:
                break
            frame = fix_frame_rotation(frame, rotation)
            height, width = frame.shape[:2]
            frames.append(frame)

        video_writer = cv2.VideoWriter(self.output_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.fps, (int(width), int(height)))

        # run the augmentation
        for i, frame in enumerate(frames):
            if to_flip:
                frame = self.horizontal_flip(frame)
            if to_rotate:
                frame = self.rotate(frame, to_rotate)
            if to_enlighten:
                frame = self.enlighten(frame, to_enlighten)

            if to_down_sample and i % random.randint(1, 4) == 0:
                continue
            elif to_up_sample and random.random() < self.threshold:
                video_writer.write(frame)

            video_writer.write(frame)

        video_reader.release()
        video_writer.release()

        return self.output_path

    def horizontal_flip(self, frame):
        # flip on the y-axis
        return cv2.flip(frame, 1)

    def enlighten(self, frame, value):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if value < 0:
            lim = 255
        else:
            lim = 255 - value

        v[v > lim] = 255
        v[v <= lim] = v[v <= lim] + value

        final_hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return frame

    def rotate(self, frame, value):
        image_center = tuple(np.array(frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, value, 1.0)
        frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1],
                               flags=cv2.INTER_LINEAR)

        return frame
