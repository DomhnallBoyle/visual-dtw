import argparse
import ast
import glob
import os
import random
import re

import matplotlib.pyplot as plt
from main import configuration
from main.models import Config
from main.research.test_full_update import create_template
from main.research.test_update_list_3 import get_default_sessions, \
    get_user_sessions, transcribe_signal
from main.utils.io import read_json_file

RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'


def get_sessions(videos_directory):
    video_paths = glob.glob(os.path.join(videos_directory, '*.mp4'))

    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    sessions = {}
    for video_path in video_paths:
        base_path = os.path.basename(video_path)

        user_id, phrase_id, session_id = \
            re.match(RECORDINGS_REGEX, base_path).groups()

        template = create_template(video_path)
        if not template:
            continue

        session_id = int(session_id)

        label = pava_phrases[phrase_id]

        session_templates = sessions.get(session_id, [])
        session_templates.append((label, template))
        sessions[session_id] = session_templates

    return sessions


def create_session_mixes(videos_directory, session_mixes):
    sessions = get_sessions(videos_directory)

    # # remove non full sessions
    # sessions = {
    #     session_id: session_templates
    #     for session_id, session_templates in sessions.items()
    #     if len(session_templates) == len(pava_phrases)
    # }

    # create session mixes
    new_sessions = {}
    new_session_mixes = []
    for session_1, session_2 in session_mixes:
        if session_1 not in sessions or session_2 not in sessions:
            continue

        new_session_mixes.append([session_1, session_2])

        session_1_templates = sessions[session_1]
        session_2_templates = sessions[session_2]

        halfway_point_1 = len(session_1_templates) // 2
        halfway_point_2 = len(session_2_templates) // 2

        print(session_1, len(session_1_templates))
        print(session_2, len(session_2_templates))

        mix_1 = session_1_templates[:halfway_point_1] + \
                session_2_templates[halfway_point_2:]
        mix_2 = session_2_templates[:halfway_point_2] + \
                session_1_templates[halfway_point_1:]

        new_sessions[session_1] = mix_1
        new_sessions[session_2] = mix_2

    return sessions, new_sessions, new_session_mixes


def get_accuracy(ref_sessions, test_sessions):
    dtw_params = Config().__dict__

    ref_signals = [(label, template.blob)
                   for session_label, ref_session in ref_sessions
                   for label, template in ref_session]

    test_signals = [(label, template.blob)
                    for session_label, test_session in test_sessions
                    for label, template in test_session]

    num_correct = 0
    for actual_label, test_signal in test_signals:
        try:
            predictions = transcribe_signal(ref_signals, test_signal, None,
                                            **dtw_params)
        except Exception as e:
            continue

        if actual_label == predictions[0]['label']:
            num_correct += 1

    accuracy = (num_correct / len(test_signals)) * 100

    return accuracy


def draw_graph(ys, title, point_labels=None, save_path=None):
    x = [i + 1 for i in range(len(ys[0]))]
    labels = ['Original', 'Mix']
    markers = ['o', 'x']
    text_positions = [10, -10]
    for i, y in enumerate(ys):
        plt.plot(x, y, label=labels[i], marker=markers[i])
        if point_labels:
            for j, (_x, _y) in enumerate(zip(x, y)):
                plt.annotate(point_labels[i][j], (_x, _y),
                             textcoords='offset points',
                             xytext=(0, text_positions[i]), ha='center')
    plt.xlabel('Num Added Sessions')
    plt.ylabel('Accuracy')
    plt.ylim((0, 100))
    plt.xticks(x)
    plt.legend(loc='lower right')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def experiment_1(args):
    """Adding mix and same originals at a time
    Every 2nd add should be the same accuracies
    """
    original_sessions, mixed_sessions, session_mixes = \
        create_session_mixes(args.ref_videos_directory, args.session_mixes)

    test_sessions = get_user_sessions(args.test_videos_directory)
    default_sessions = get_default_sessions()

    added_originals, added_mixed = [], []
    original_accuracies, mixed_accuracies = [], []
    for j, (session_1, session_2) in enumerate(session_mixes):
        original_1 = original_sessions[session_1]
        original_2 = original_sessions[session_2]

        mix_1 = mixed_sessions[session_1]
        mix_2 = mixed_sessions[session_2]

        added_originals.append((f'session_{session_1}', original_1))
        added_mixed.append((f'mix_{session_1}_{session_2}', mix_1))

        print(f'Mix {j+1}:')
        for i in range(len(added_originals)):
            print(i+1)
            print(added_originals[i][0], len(added_originals[i][1]))
            print(added_mixed[i][0], len(added_mixed[i][1]))

        # test
        original_accuracy = get_accuracy(default_sessions + added_originals,
                                         test_sessions)
        mix_accuracy = get_accuracy(default_sessions + added_mixed,
                                    test_sessions)
        original_accuracies.append(original_accuracy)
        mixed_accuracies.append(mix_accuracy)

        print(original_accuracy, mix_accuracy, '\n')

        added_originals.append((f'session_{session_2}', original_2))
        added_mixed.append((f'mix_{session_1}_{session_2}', mix_2))

        # test
        original_accuracy = get_accuracy(default_sessions + added_originals,
                                         test_sessions)
        mix_accuracy = get_accuracy(default_sessions + added_mixed,
                                    test_sessions)
        original_accuracies.append(original_accuracy)
        mixed_accuracies.append(mix_accuracy)

        assert original_accuracy == mix_accuracy

    print(original_accuracies, mixed_accuracies)

    ys = [original_accuracies, mixed_accuracies]
    draw_graph(ys, 'Experiment 1: Original vs Mixed')


def experiment_2(args):
    """No consistent accuracies until the final sessions have been added"""
    original_sessions, mixed_sessions, session_mixes = \
        create_session_mixes(args.ref_videos_directory, args.session_mixes)

    test_sessions = get_user_sessions(args.test_videos_directory)
    default_sessions = get_default_sessions()

    original_sessions = [
        (f'original_{k}', v) for k, v in original_sessions.items()
        if k in mixed_sessions
    ]
    mixed_sessions = [
        (f'mixed_{k}', v) for k, v in mixed_sessions.items()
    ]

    assert len(original_sessions) == len(mixed_sessions)

    random.shuffle(original_sessions)
    random.shuffle(mixed_sessions)

    added_originals, added_mixed = [], []
    original_accuracies, mixed_accuracies = [], []
    for i, (original_session, mixed_session) in enumerate(
            zip(original_sessions, mixed_sessions)):

        added_originals.append(original_session)
        added_mixed.append(mixed_session)

        original_accuracy = get_accuracy(default_sessions + added_originals,
                                         test_sessions)
        mix_accuracy = get_accuracy(default_sessions + added_mixed,
                                    test_sessions)

        original_accuracies.append(original_accuracy)
        mixed_accuracies.append(mix_accuracy)

        # check last accuracies
        if i == len(original_sessions) - 1:
            assert original_accuracy == mix_accuracy

    ys = [original_accuracies, mixed_accuracies]
    draw_graph(ys, 'Experiment 2: Original vs Mixed')


def create_mix(session_1, session_2, name):
    while True:
        split = int(len(session_1[1]) * random.random())
        mixed_templates = session_1[1][:split] + session_2[1][split:]

        assert len(mixed_templates) == len(session_1[1]) == len(session_2[1])

        phrase_names = [t[0] for t in mixed_templates]
        if len(set(phrase_names)) == 20:
            print(split, phrase_names)
            break

    return name, mixed_templates


def experiment_3(args):
    """Create mixes using different splits,
    test with different sets, add different original sessions and repeat"""
    user_sessions = get_sessions(args.ref_videos_directory)
    user_sessions = [
        (f'session_{k}', v)
        for k, v in user_sessions.items()
        if len(v) == 20  # make sure we've completed sessions
    ]

    test_sessions = get_user_sessions(args.test_videos_directory)
    default_sessions = get_default_sessions()

    num_repeats = 5
    for i in range(num_repeats):

        # grab the user sessions this time
        random.shuffle(user_sessions)
        sessions_this_round = user_sessions[:8]

        # different test sessions each time
        random.shuffle(test_sessions)
        test_sessions_this_round = test_sessions[:5]

        added_originals, added_mixed = [], []
        original_accuracies, mixed_accuracies = [], []
        original_ticks, mix_ticks = [], []
        for j in range(len(sessions_this_round) - 1):
            session_1, session_2 = sessions_this_round[j], \
                                   sessions_this_round[j+1]

            mixed_session = create_mix(session_1, session_2,
                                       name=f'mixed_{j+1}')
            added_mixed.append(mixed_session)

            # start original from the end
            original_session = sessions_this_round[-(j+1)]
            added_originals.append(original_session)

            original_accuracy = get_accuracy(
                default_sessions + added_originals,
                test_sessions_this_round)
            mix_accuracy = get_accuracy(default_sessions + added_mixed,
                                        test_sessions_this_round)

            original_accuracies.append(original_accuracy)
            mixed_accuracies.append(mix_accuracy)

            original_ticks.append(original_session[0])
            mix_ticks.append(mixed_session[0])

        draw_graph([original_accuracies, mixed_accuracies], f'Round {i+1}',
                   point_labels=[original_ticks, mix_ticks],
                   save_path=f'round_{i+1}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref_videos_directory')
    parser.add_argument('test_videos_directory')

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('experiment_1')
    parser_1.add_argument('session_mixes', type=ast.literal_eval)

    parser_2 = sub_parsers.add_parser('experiment_2')
    parser_2.add_argument('session_mixes', type=ast.literal_eval)

    parser_3 = sub_parsers.add_parser('experiment_3')

    args = parser.parse_args()
    run_type = args.run_type

    f = {
        'experiment_1': experiment_1,
        'experiment_2': experiment_2,
        'experiment_3': experiment_3
    }
    f[run_type](args)
