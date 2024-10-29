import argparse
import glob
import os

import cv2
import pandas as pd
from main.models import PAVAList
from main.research.process_videos import get_video_rotation, fix_frame_rotation
from main.utils.db import db_session


def main(args):
    video_paths = glob.glob(os.path.join(args.videos_directory, '*.mp4'))

    with db_session() as s:
        lst = PAVAList.get(s, filter=(PAVAList.id == args.list_id), first=True)
        phrases = {
            str(i): phrase.content for i, phrase in enumerate(lst.phrases)
            if not phrase.is_nota
        }

    groundtruth_results = []
    for video_path in video_paths:
        while True:
            print('\n', phrases)

            video_player = cv2.VideoCapture(video_path)
            fps = int(video_player.get(cv2.CAP_PROP_FPS))
            video_rotation = get_video_rotation(video_path)
            if fps < 25:
                fps = 30

            while True:
                success, frame = video_player.read()
                if not success:
                    break
                frame = fix_frame_rotation(frame, video_rotation)

                cv2.imshow(os.path.basename(video_path), frame)
                cv2.waitKey(fps)

            video_player.release()

            entry = input('What phrase was uttered? (r = repeat): ')
            if entry != 'r':
                phrase = phrases.get(entry)
                if not phrase:
                    print('Incorrect entry, try again')
                    continue

                groundtruth_results.append(
                    [os.path.basename(video_path), phrase]
                )
                break

        cv2.destroyAllWindows()

    df = pd.DataFrame(groundtruth_results,
                      columns=['video_path', 'groundtruth_phrase'])
    df.to_csv(os.path.join(args.videos_directory, 'groundtruth.csv'),
              index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_directory')
    parser.add_argument('list_id')

    main(parser.parse_args())
