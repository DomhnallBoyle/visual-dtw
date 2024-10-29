"""Code to interact with a web-cam to record videos"""
import time

import cv2


class Cam:

    def __init__(self, width=640, height=480, codec='mp4v', debug=False,
                 playback=False, countdown=False):
        self.width = width
        self.height = height
        self.record_path = '/tmp/recording.mp4'
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.debug = debug
        self.playback = playback
        self.countdown = countdown

    def record(self, save_path=None):
        if not save_path:
            save_path = self.record_path

        vc = cv2.VideoCapture(0)
        fps = int(vc.get(cv2.CAP_PROP_FPS))
        output = cv2.VideoWriter(save_path, self.codec, 25,
                                 (self.width, self.height))

        if self.countdown:
            print('Recording in', end=' ', flush=True)
            for i in range(3, 0, -1):
                print(f'{i}...', end='', flush=True)
                time.sleep(1)

        while True:
            success, frame = vc.read()
            if not success:
                break

            if self.debug:
                cv2.imshow('Recording', frame)

            output.write(frame)

            if cv2.waitKey(int((1 / fps) * 1000)) & 0XFF == ord('q'):
                break

        vc.release()
        output.release()

        if self.playback:
            vc = cv2.VideoCapture(save_path)
            fps = int(vc.get(cv2.CAP_PROP_FPS))

            while True:
                success, frame = vc.read()
                if not success:
                    break

                cv2.imshow('Playback', frame)
                cv2.waitKey(fps)

            vc.release()

        cv2.destroyAllWindows()

        return save_path

    def record_loop(self):
        while True:
            time.sleep(2)
            yield self.record()
