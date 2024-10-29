import argparse
import ast
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config
from main.research.cmc import CMC
from main.services.transcribe import transcribe_signal
from main.utils.cfe import run_cfe
from main.utils.io import read_json_file, read_pickle_file
from main.utils.pre_process import pre_process_signals

VIDEOS_DIRECTORY = '/home/domhnall/Documents/sravi_dataset/liopa/'
VIDEO_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
ANALYSIS_REGEX = r'(\d+),(\d+),(\[.+\]),(\d+),(\d+)'


def experiment():
    """Vary different values for the threshold on Richard, Fabian and Domhnall
    data. From None - 70
    """
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']
    dtw_params = Config().__dict__
    user_templates = {}

    for user in [9, 11, 12]:
        user_templates[user] = []
        user_directory = os.path.join(VIDEOS_DIRECTORY, str(user))
        for video in os.listdir(user_directory):
            if not video.endswith('.mp4'):
                continue

            with open(os.path.join(user_directory, video), 'rb') as f:
                matrix = run_cfe(f)

            test_signal = \
                pre_process_signals(signals=[matrix], **dtw_params)[0]

            phrase_id = re.match(VIDEO_REGEX, video).groups()[1]
            actual_label = pava_phrases[phrase_id]

            user_templates[user].append((actual_label, test_signal))

    default_list = read_pickle_file(configuration.DEFAULT_LIST_PATH)
    ref_signals = [(phrase.content, template.blob)
                   for phrase in default_list.phrases
                   for template in phrase.templates
                   if not phrase.is_nota]

    for dtw_threshold in range(1, 71):
        dtw_params['threshold'] = dtw_threshold

        for user, test_signals in user_templates.items():
            print(f'{dtw_threshold} - {user}')
            cmc = CMC(num_ranks=3)
            num_tests = 0
            for actual_label, test_signal in test_signals:
                try:
                    predictions = transcribe_signal(ref_signals, test_signal,
                                                    **dtw_params)
                    predictions = [prediction['label']
                                   for prediction in predictions]
                    cmc.tally(predictions, actual_label)
                    num_tests += 1
                except Exception as e:
                    print(e)

            cmc.calculate_accuracies(num_tests, count_check=False)

            with open('dtw_threshold.csv', 'a') as f:
                f.write(f'{dtw_threshold},{user},'
                        f'{cmc.all_rank_accuracies[0]},{num_tests},'
                        f'{len(test_signals)}\n')


def analyse(file_path):
    data, columns = [], ['DTW Threshold', 'User ID', 'Ranks', 'Num Tests',
                         'Total Tests']

    with open(file_path, 'r') as f:
        for line in f.readlines():
            dtw_threshold, user_id, ranks, num_tests, total_tests = \
                re.match(ANALYSIS_REGEX, line).groups()
            ranks = ast.literal_eval(ranks)
            data.append([int(dtw_threshold), int(user_id), ranks,
                         int(num_tests), int(total_tests)])

    df = pd.DataFrame(columns=columns, data=data)

    users = df['User ID'].unique()

    for user in users:
        sub_df = df[df['User ID'] == user]

        rank_1, rank_2, rank_3 = [], [], []
        for index, row in sub_df.iterrows():
            ranks = row['Ranks']
            rank_1.append(ranks[0])
            rank_2.append(ranks[1])
            rank_3.append(ranks[2])

        num_tests = list(sub_df['Num Tests'])
        total_tests = list(sub_df['Total Tests'])[0]
        x = list(sub_df['DTW Threshold'])

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(x, rank_1, label='Rank 1')
        ax1.plot(x, rank_2, label='Rank 2')
        ax1.plot(x, rank_3, label='Rank 3')
        ax1.set_ylim(80, 101)
        ax1.set_ylabel('% Accuracy')
        ax1.set_xlabel('DTW Threshold')
        ax1.legend()

        ax2.plot(x, num_tests, color='red')
        ax2.set_ylabel(f'Num Tests (Total = {total_tests})')

        plt.title(f'% Accuracy vs Num Tests vs DTW Threshold - User {user}')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser1 = sub_parsers.add_parser('experiment')

    parser2 = sub_parsers.add_parser('analyse')
    parser2.add_argument('file_path')

    args = parser.parse_args()
    run_type = args.run_type

    if run_type == 'experiment':
        experiment()
    elif run_type == 'analyse':
        analyse(args.file_path)
    else:
        parser.print_help()
