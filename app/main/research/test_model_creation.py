"""
Create models from video directories using SSA
Works with any phrase list
"""
import argparse
import random

from main.research.research_utils import get_accuracy, \
    create_sessions, sessions_to_templates
from scripts.session_selection import \
    session_selection_with_cross_validation, \
    session_selection_with_cross_validation_fast


def main(args):
    phrase_lookup = {
        i+1: phrase
        for i, phrase in enumerate(args.phrase_list)
    }
    user_sessions = create_sessions(
        args.videos_directory,
        [args.regex],
        phrase_lookup=phrase_lookup,
        save=True,
        debug=True
    )
    half_index = len(user_sessions) // 2

    average_rank_accuracies = [0, 0, 0]
    for repeat in range(args.num_repeats):
        random.shuffle(user_sessions)
        sessions_to_add = user_sessions[:half_index]
        train_test_sessions = user_sessions[half_index:]
        print('Num sessions to add:', len(sessions_to_add))
        print('Num train/test sessions:', len(train_test_sessions))

        # now extract training/test data
        # in production, this will be all user transcriptions
        # using sessions for now
        train_test_data = sessions_to_templates(train_test_sessions)
        random.shuffle(train_test_data)
        training_test_split = int(len(train_test_data)
                                  * args.training_test_split)
        training_data = train_test_data[:training_test_split]
        test_data = train_test_data[training_test_split:]
        print('Training/test data:', len(train_test_data))
        print('Training data:', len(training_data))
        print('Test data:', len(test_data))
        assert len(training_data) + len(test_data) == len(train_test_data)

        if args.fast:
            selected_session_ids = session_selection_with_cross_validation_fast(
                _sessions=sessions_to_add,
                _training_templates=training_data,
            )
        else:
            selected_session_ids = session_selection_with_cross_validation(
                _sessions=sessions_to_add,
                _training_templates=training_data,
                initial_max=args.initial_max
            )
        print(f'Selected {len(selected_session_ids)} sessions',
              selected_session_ids)

        selected_sessions = [s for s in sessions_to_add
                             if s[0] in selected_session_ids]
        assert len(selected_sessions) == len(selected_session_ids)

        selected_sessions_templates = sessions_to_templates(selected_sessions)

        print(f'Testing model vs {len(test_data)} external templates')
        accuracies, confusion_matrix = get_accuracy(
            selected_sessions_templates,
            test_data,
            debug=True
        )
        print('Accuracies:', accuracies, '\n')

        for i, accuracy in enumerate(accuracies):
            average_rank_accuracies[i] += accuracy

    average_rank_accuracies = [accuracy / args.num_repeats
                               for accuracy in average_rank_accuracies]
    print('Average Rank Accuracies:', average_rank_accuracies)


def file_list(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')
    parser.add_argument('phrase_list', type=file_list)
    parser.add_argument('regex')
    parser.add_argument('--num_repeats', type=int, default=5)
    parser.add_argument('--training_test_split', type=float, default=0.6)
    parser.add_argument('--initial_max', type=int, default=2)
    parser.add_argument('--fast', action='store_true')

    main(parser.parse_args())
