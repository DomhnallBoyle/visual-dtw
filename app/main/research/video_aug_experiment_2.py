import argparse
import os
import random

import matplotlib.pyplot as plt
from main.research.research_utils import create_template, get_accuracy, \
    sessions_to_templates, get_default_sessions
from main.research.video_aug import VideoAug
from main.utils.io import read_csv_file, read_pickle_file
from scripts.session_selection import \
    session_selection_with_cross_validation_fast


def main(args):
    csv_path = os.path.join(args.videos_directory, 'data.csv')

    df = read_csv_file(csv_path)
    df = df[df['Groundtruth'] != 'None']  # use only groundtruthed
    print(df)

    default_sessions = get_default_sessions()
    default_templates = sessions_to_templates(default_sessions)

    all_round_accuracies = []

    for repeat in range(args.num_repeats):
        try:
            print(f'\nRepeat {repeat+1}')

            # get phrase videos
            phrases_d = {}
            for index, row in df.iterrows():
                phrase = row['Groundtruth']
                video_path = os.path.join(args.videos_directory,
                                          f'{row["Sample ID"]}.mp4')

                video_paths = phrases_d.get(phrase, [])
                video_paths.append(video_path)
                phrases_d[phrase] = video_paths

            # get phrase counts
            phrase_count_d = {phrase: len(video_paths)
                              for phrase, video_paths in phrases_d.items()}
            print(len(phrase_count_d),
                  min(phrase_count_d, key=phrase_count_d.get),
                  min(phrase_count_d.values()))

            # create random session as initial session
            for phrase, video_paths in phrases_d.items():
                random.shuffle(video_paths)
            initial_session_video_paths = \
                [(phrase, video_paths.pop(0))
                 for phrase, video_paths in phrases_d.items()]

            # get training and test templates
            remaining_templates = []
            for phrase, video_paths in phrases_d.items():
                for video_path in video_paths:
                    template = create_template(video_path)
                    if not template:
                        continue
                    remaining_templates.append((phrase, template.blob))
            random.shuffle(remaining_templates)
            split = int(len(remaining_templates) * 0.6)
            training_templates = remaining_templates[:split]
            test_templates = remaining_templates[split:]
            print('Training templates:', len(training_templates))
            print('Test templates:', len(test_templates))

            # apply video augmentation to create new sessions from 1
            video_aug = VideoAug()
            augmented_sessions = []
            sessions = []
            for i in range(args.num_augmented_sessions+1):
                session = []
                for phrase, video_path in initial_session_video_paths:
                    if i == 0:
                        template = create_template(video_path)
                        if template:
                            session.append((phrase, template.blob))
                    else:
                        while True:
                            aug_video_path = video_aug.process(video_path)
                            template = create_template(aug_video_path)
                            if template:
                                session.append((phrase, template.blob))
                                break
                if i == 0:
                    sessions.append((f'initial_{i+1}', session))
                else:
                    augmented_sessions.append((f'augmented_{i+1}', session))
            print('Num augmented sessions:', len(augmented_sessions))
            # for label, session in augmented_sessions:
            #     print(label, len(session))

            # get default accuracies
            default_accuracies = get_accuracy(
                default_templates,
                test_templates,
                debug=True
            )[0]

            # get model accuracies from augmented sessions
            selected_ids = session_selection_with_cross_validation_fast(
                augmented_sessions, training_templates
            )
            selected_sessions = \
                [s for s in augmented_sessions if s[0] in selected_ids]
            num_selected = len(selected_sessions)
            print(f'Selected {num_selected}')
            selected_sessions_templates = sessions_to_templates(selected_sessions)
            augmented_accuracies = get_accuracy(
                selected_sessions_templates,
                test_templates,
                debug=True
            )[0]

            # replicate the sessions to 5
            replicated_sessions = [sessions[0]] * 5
            replicated_sessions_templates = \
                sessions_to_templates(replicated_sessions)
            replicated_accuracies = get_accuracy(
                replicated_sessions_templates,
                test_templates,
                debug=True
            )[0]

            print('Selected sessions:', len(selected_sessions))
            print('Replicated sessions:', len(replicated_sessions))

            round_accuracies = [default_accuracies,
                                augmented_accuracies,
                                replicated_accuracies]

            # running test templates vs external model supplied as arg
            if args.model_path:
                model = read_pickle_file(args.model_path)[0]
                model_sessions = model[1]
                model_sessions_templates = sessions_to_templates(model_sessions)
                model_accuracies = get_accuracy(
                    model_sessions_templates,
                    test_templates,
                    debug=True
                )[0]
                round_accuracies.append(model_accuracies)

            all_round_accuracies.append(round_accuracies)
        except KeyboardInterrupt:
            break

    x = [1, 2, 3]
    num_rows, num_columns = 2, 3
    row, column = 0, 0
    fig, axs = plt.subplots(num_rows, num_columns)
    for i, round_accuracies in enumerate(all_round_accuracies):
        xs = [x] * len(round_accuracies)
        labels = ['Default', 'Augmented', 'Replicated']
        if len(xs) > 3:
            labels.append('Model')
        for x, y, label in zip(xs, round_accuracies, labels):
            axs[row, column].plot(x, y, label=label)
        axs[row, column].set_xticks(x, x)
        axs[row, column].set_title(f'Repeat {i+1}')
        axs[row, column].legend()
        axs[row, column].set_ylabel('Accuracy %')
        axs[row, column].set_xlabel('Rank')
        axs[row, column].set_ylim((0, 101))

        if column == num_columns - 1:
            row += 1
            column = 0
        else:
            column += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')
    parser.add_argument('--num_repeats', type=int, default=5)
    parser.add_argument('--num_augmented_sessions', type=int, default=10)
    parser.add_argument('--model_path')

    main(parser.parse_args())
