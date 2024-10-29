import argparse

import matplotlib.pyplot as plt
from main.research.session_updating_utils import *


def main(args):
    user_sessions = get_user_sessions(args.videos_directory)
    default_sessions = get_default_sessions()

    half_index = len(user_sessions) // 2

    for repeat in range(args.num_repeats):
        print('Repeat:', repeat+1, '\n')

        # extract sessions to add and training/test sessions
        while True:
            random.shuffle(user_sessions)
            sessions_to_add = user_sessions[:half_index]
            train_test_sessions = user_sessions[half_index:]

            # make sure sessions to add are complete
            if all(len(s[1]) == 20 for s in sessions_to_add):
                break

        # extract training/test data
        train_test_data = [(label, template)
                           for session_label, session in train_test_sessions
                           for label, template in session]
        random.shuffle(train_test_data)
        train_split = int(len(train_test_data) * 0.6)
        training_data = train_test_data[:train_split]
        test_data = train_test_data[train_split:]

        print('Training:', len(training_data))
        print('Testing:', len(test_data))

        all_sessions_to_add = sessions_to_add + default_sessions

        model_sessions = session_selection_algorithm(all_sessions_to_add,
                                                     training_data)
        model_session_ids = [s[0] for s in model_sessions]
        print('Model Session IDs:', model_session_ids)

        original_accuracy = get_accuracy(model_sessions, test_data)
        print('Original accuracy:', original_accuracy)

        caps, capped_accuracies = [], []
        for session_cap in range(len(sessions_to_add), len(model_sessions)+1):
            caps.append(session_cap)
            capped_sessions = model_sessions[:session_cap]
            print(f'Session cap {session_cap}: {len(capped_sessions)} sessions')

            capped_sessions_accuracy = get_accuracy(capped_sessions, test_data)
            capped_accuracies.append(capped_sessions_accuracy)

        # last accuracy should be same as original
        assert capped_accuracies[-1] == original_accuracy

        print('Caps:', caps)
        print('Capped Accuracies:', capped_accuracies)

        # plot results
        x, y = caps, capped_accuracies
        plt.plot(x, y, label='Capped Accuracy', marker='x')
        plt.axhline(y=original_accuracy, label='Original Accuracy',
                    color='black', linestyle='--')
        plt.xlabel('Cap')
        plt.ylabel('Accuracy')
        plt.ylim((0, 100))
        plt.xticks(x)
        plt.legend()
        plt.title(f"{args.user} - Repeat {repeat+1}\n"
                  f"{', '.join([s[0] + '-' + s.split('_')[1] for s in model_session_ids])}")
        plt.savefig(f'{args.output_directory}/{args.user}_{repeat+1}.png')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')
    parser.add_argument('output_directory')
    parser.add_argument('user')
    parser.add_argument('--num_repeats', type=int, default=1)

    main(parser.parse_args())
