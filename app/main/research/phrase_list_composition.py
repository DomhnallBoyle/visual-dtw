"""
Testing accuracy and recognition speed as more phrases are added
Looking to find the optimum number of phrases per list to have
Experiment 3 & 5 are the best experiments from the confluence page
"""
import argparse
import ast
import glob
import math
import multiprocessing
import os
import random
import re
import time
from collections import Counter

import abydos.phonetic as ap
import jellyfish
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from main.models import Config
from main.research.extract_similar_phrase import find_similar_phrases
from main.research.research_utils import create_template
from main.services.transcribe import transcribe_signal
from main.utils.cmc import CMC
from main.utils.io import read_pickle_file, write_pickle_file
from tqdm import tqdm

CURRENT_DIRECTORY = os.path.dirname(__file__)
PHRASES_LOOKUP = {
    i+1: p.strip() for i, p in enumerate(
        open(os.path.join(CURRENT_DIRECTORY, 'phrases.txt'), 'r').readlines()
    )
}
NUM_PHRASES = len(PHRASES_LOOKUP)
VIDEO_NAME_REGEX = r'SRAVIExtended-(.+)-P(\d+)-S(\d+).mp4'
NUM_SESSIONS = 5
PICKLE_FILE = 'phrase_list_composition_{}_data.pkl'
DTW_PARAMS = Config().__dict__
CONTRACTIONS = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}
WORD = re.compile(r"\w+")
VISEME_TO_PHONEME = {  # lee and york 2002
    'p': ['p', 'b', 'm', 'em'],
    'f': ['f', 'v'],
    't': ['t', 'd', 's', 'z', 'th', 'dh', 'dx'],
    'w': ['w', 'wh', 'r'],
    'ch': ['ch', 'jh', 'sh', 'zh'],
    'ey': ['eh', 'ey', 'ae', 'aw'],
    'k': ['k', 'g', 'n', 'l', 'nx', 'hh', 'y', 'el', 'en', 'ng'],
    'iy': ['iy', 'ih'],
    'aa': ['aa'],
    'ah': ['ah', 'ax', 'ay'],
    'er': ['er'],
    'ao': ['ao', 'oy', 'ix', 'ow'],
    'uh': ['uh', 'uw'],
    'sp': ['sil', 'sp']
}
PHONEME_TO_VISEME = {
    phoneme: viseme
    for viseme, phonemes in VISEME_TO_PHONEME.items()
    for phoneme in phonemes
}
WORDS_TO_VISEMES = {}

# TODO: Extend cmudict with new words e.g. SRAVI
#  use https://github.com/cmusphinx/g2p-seq2seq

# construct visemes from word phoneme dictionary
# inspiration from https://github.com/steinbro/hyperviseme
CMU_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIRECTORY))), 'cmudict-en-us.dict')
with open(CMU_DICT_PATH, 'r') as f:
    for line in f.readlines():
        line = line.split(' ')
        word = line[0]

        phones = list(map(lambda phone: phone.lower().strip(), line[1:]))
        visemes = ''.join(
            list(map(lambda phone: PHONEME_TO_VISEME[phone], phones)))

        WORDS_TO_VISEMES[word] = visemes


def append_to_dict(d, phrase_id, template):
    phrase = PHRASES_LOOKUP[phrase_id]

    phrase_videos = d.get(phrase, [])
    phrase_videos.append(template)
    d[phrase] = phrase_videos

    return d


def get_data(videos_directory, user_id):
    print('Getting data')
    pickle_path = PICKLE_FILE.format(user_id)

    if os.path.exists(pickle_path):
        data = read_pickle_file(pickle_path)
    else:
        data = {}

        video_paths = glob.glob(os.path.join(videos_directory, '*.mp4'))
        for video_path in tqdm(video_paths):
            base_name = os.path.basename(video_path)
            user_id, phrase_id, session_id = \
                re.match(VIDEO_NAME_REGEX, base_name).groups()

            template = create_template(video_path)
            if not template:
                continue

            data = append_to_dict(data, int(phrase_id), template)

        # ensure num phrase templates == num sessions
        # get unused phrases
        unused_phrases = []
        for phrase_id, phrase in PHRASES_LOOKUP.items():
            if phrase not in data:
                unused_phrases.append(phrase)
            elif len(data[phrase]) != NUM_SESSIONS:
                del data[phrase]
                unused_phrases.append(phrase)

        write_pickle_file(data, pickle_path)
        print('Unused phrases:', unused_phrases)

    # shuffle templates in dictionary
    for k, v in data.items():
        random.shuffle(v)

    return data


def get_synthetic_data(videos_directory, user_id):
    print('Getting synthetic data')
    # get data
    pickle_file = f'phrase_list_composition_{user_id}_synthetic_data.pkl'
    if os.path.exists(pickle_file):
        phrases_d = read_pickle_file(pickle_file)
    else:
        phrases_d = {}
        total_num_sessions = 0
        df = pd.read_csv(os.path.join(videos_directory, 'groundtruth.csv'),
                         names=['video_name', 'said', 'actual'])
        max_num_phrases = len(df['actual'].unique())
        for index, row in tqdm(df.iterrows()):
            video_name = row['video_name']
            video_path = os.path.join(videos_directory, video_name)
            template = create_template(video_path)
            if not template:
                continue
            video_name = video_name.replace('.mp4', '')
            session_id = int(video_name.split('_')[-1])
            phrase = row['actual']
            phrase_templates = phrases_d.get(phrase, [])
            phrase_templates.append(template)
            phrases_d[phrase] = phrase_templates
            if session_id > total_num_sessions:
                total_num_sessions = session_id
        phrases_d = {
            phrase: templates
            for phrase, templates in phrases_d.items()
            if len(templates) == total_num_sessions
        }
        write_pickle_file(phrases_d, pickle_file)
    for k, v in phrases_d.items():
        random.shuffle(v)
    return phrases_d


def test(training_templates, test_templates, max_num_vectors=None):
    if len(training_templates) == 0 or len(test_templates) == 0:
        return 0, 0

    if max_num_vectors:
        DTW_PARAMS['max_num_vectors'] = max_num_vectors

    training_signals = [(label, template.blob)
                        for label, template in training_templates]

    num_unique_labels = len(set([s[0] for s in training_signals]))
    num_ranks = 3 if num_unique_labels >= 3 else num_unique_labels

    cmc = CMC(num_ranks=num_ranks)

    average_time_taken = 0
    for groundtruth_label, test_template in test_templates:
        start_time = time.time()
        predictions = transcribe_signal(training_signals, test_template.blob,
                                        None, **DTW_PARAMS)
        end_time = time.time()
        time_taken = end_time - start_time

        prediction_labels = [p['label'] for p in predictions]
        cmc.tally(prediction_labels, groundtruth_label)

        average_time_taken += time_taken

    average_time_taken /= len(test_templates)

    cmc.calculate_accuracies(num_tests=len(test_templates), count_check=False)
    accuracies = [round(a / 100, 2) for a in cmc.all_rank_accuracies[0]]

    return accuracies[0], round(average_time_taken, 2), accuracies


def experiment_1(args):
    """
    get accuracy and time taken as more phrases are added
    repeat a number of times
    """
    data = get_data(args.videos_directory, args.user_id)
    phrases_to_add = list(data.keys())
    random.shuffle(phrases_to_add)

    for repeat in range(NUM_SESSIONS):
        print('\nRepeat:', repeat+1)

        training_data = {
            k: v[:repeat] + v[repeat+1:] for k, v in data.items()
        }
        test_data = {
            k: v[repeat] for k, v in data.items()
        }

        training_templates, test_templates = [], []
        for number_added_phrases, phrase_to_add in enumerate(phrases_to_add):
            training_templates.extend(
                [(phrase_to_add, template)
                 for template in training_data[phrase_to_add]]
            )
            test_templates.append((phrase_to_add, test_data[phrase_to_add]))

            accuracy, average_time_taken, accuracies = test(training_templates,
                                                            test_templates)

            with open(f'phrase_list_composition_1_{args.user_id}_results.csv', 'a') as f:
                f.write(f'{repeat+1},{number_added_phrases+1},{accuracies},'
                        f'{average_time_taken},{len(training_templates)},'
                        f'{len(test_templates)}\n')


def analysis_1_a(args):
    df = pd.read_csv(args.results_file,
                     names=['Num Added Phrases', 'Accuracy',
                            'Average Time', 'Num Training Templates',
                            'Num Test Templates'])

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    ax.plot(df['Num Added Phrases'], df['Accuracy'], label='Accuracy')
    ax.set_xlabel('Num Added Phrases')
    ax.set_ylabel('Accuracy')
    ax.set_title(args.title)
    ax.set_ylim([0.0, 1])
    ax.grid(True)

    # add second y-axis plot
    ax2 = ax.twinx()
    ax2.plot(df['Num Added Phrases'], df['Average Time'], color='green',
             label='Average Time')
    ax2.set_ylabel('Average Time Per Test (s)')
    ax2.set_ylim([0.0, 0.5])

    ax.legend(loc='upper center')
    ax2.legend(loc='lower center')

    plt.show()


def analysis_1_b(args):
    data, columns = [], ['Repeat', 'Num Added Phrases', 'Accuracies',
                         'Time Taken', 'Num Training Templates',
                         'Num Test Templates']
    regex = r'(\d+),(\d+),(\[.+\]),(\d+.\d+),(\d+),(\d+)'

    with open(args.results_file, 'r') as f:
        for line in f.readlines():
            repeat, num_added_phrases, accuracies, time_taken, \
                num_training_templates, num_test_templates = \
                re.match(regex, line).groups()
            data.append([
                int(repeat), int(num_added_phrases),
                ast.literal_eval(accuracies), float(time_taken),
                int(num_training_templates), int(num_test_templates)
            ])
    df = pd.DataFrame(data=data, columns=columns)

    for repeat in df['Repeat'].unique():
        sub_df = df[df['Repeat'] == repeat]
        xs = [[], [], []]
        ys = [[], [], []]
        time_taken = sub_df['Time Taken']
        for index, row in sub_df.iterrows():
            accuracies = row['Accuracies']
            for i in range(len(accuracies)):
                xs[i].append(row['Num Added Phrases'])
                ys[i].append(accuracies[i])

        # create figure and axis objects with subplots()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax1.plot(x, y, label=f'R{i + 1}')
        ax1.set_xlabel('Num Added Phrases')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy')
        ax1.set_ylim([0.0, 1.01])
        ax1.grid(True)
        ax1.legend()

        # add second y-axis plot
        ax2.plot(sub_df['Num Added Phrases'], time_taken, color='green',
                 label='Average Time')
        ax2.set_title('Time')
        ax2.set_ylabel('Average Time Per Test (s)')
        ax2.set_xlabel('Num Added Phrases')
        ax2.grid(True)

        fig.suptitle(f'{args.title} - Repeat {repeat}')
        plt.show()


def experiment_2_process_fold(user_id, phrases_to_add, repeat, training_data,
                              test_data):
    training_templates, test_templates = [], []
    num_times_new_phrase_predicted_correctly = 0
    for number_added_phrases, phrase_to_add in enumerate(phrases_to_add):
        test_templates_without_new_phrase = test_templates.copy()

        # 2 get accuracy before new phrase added
        before_accuracy = test(training_templates, test_templates)[0]

        # add a phrase to training and test templates
        training_templates.extend([
            (phrase_to_add, template)
            for template in training_data[phrase_to_add]
        ])
        test_templates.append((phrase_to_add, test_data[phrase_to_add]))

        # get original accuracy and time taken
        accuracy, average_time_taken = test(training_templates,
                                            test_templates)

        # 1 is new phrase correct when we add it?
        training_signals = [(label, template.blob)
                            for label, template in training_templates]
        test_signal = test_data[phrase_to_add].blob
        predictions = transcribe_signal(training_signals, test_signal,
                                        None, **DTW_PARAMS)
        if phrase_to_add == predictions[0]['label']:
            num_times_new_phrase_predicted_correctly += 1

        # 2 get accuracy after new phrase is added
        after_accuracy = test(training_templates,
                              test_templates_without_new_phrase)[0]

        with open(f'phrase_list_composition_2_{user_id}_results.csv', 'a') as f:
            f.write(f'{repeat+1},{number_added_phrases+1},{accuracy},'
                    f'{average_time_taken},{len(training_templates)},'
                    f'{len(test_templates)},{before_accuracy},'
                    f'{after_accuracy},'
                    f'{num_times_new_phrase_predicted_correctly}\n')


def experiment_2(args):
    """
    How is accuracy of a phrase affected by adding more? e.g.
    #  1) The new phrase samples are not being recognised correctly
    #  2) Previous phrase samples are now confused
    Repeat and use multiprocessing
    """

    # 1) everytime we add a phrase, we check the predictions of the
    # new phrase specifically

    # 2) do a before and after accuracy comparison of all other phrases
    # before and after a new phrase is added

    data = get_data(args.videos_directory, args.user_id)
    phrases_to_add = list(data.keys())
    random.shuffle(phrases_to_add)

    process_tasks = []
    for repeat in range(NUM_SESSIONS):
        training_data = {
            k: v[:repeat] + v[repeat + 1:] for k, v in data.items()
        }
        test_data = {
            k: v[repeat] for k, v in data.items()
        }

        process_tasks.append([args.user_id, phrases_to_add, repeat,
                              training_data, test_data])

    num_processes = NUM_SESSIONS
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(experiment_2_process_fold, process_tasks)


def analysis_2(args):
    df = pd.read_csv(args.results_file,
                     names=['Repeat', 'Num Added Phrases', 'Accuracy',
                            'Average Time', 'Num Training Templates',
                            'Num Test Templates', 'Before Accuracy',
                            'After Accuracy', 'Num correct new phrases'])

    # loop over the repeats
    num_repeats = df['Repeat'].max()
    for i in range(1, num_repeats+1):
        sub_df = df[df['Repeat'] == i]

        fig, axs = plt.subplots(3)
        fig.suptitle(f'{args.user_id}: Repeat {i}')
        fig.tight_layout()

        min_accuracy_scale = sub_df['Accuracy'].min() - 0.02
        max_accuracy_scale = sub_df['Accuracy'].max() + 0.02

        axs[0].plot(sub_df['Num Added Phrases'], sub_df['Accuracy'])
        axs[0].set_title('Accuracy')
        axs[0].set_xlabel('Num Added Phrases')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_ylim([min_accuracy_scale, max_accuracy_scale])
        axs[0].grid()

        # # add second y-axis plot
        # ax2 = axs[0].twinx()
        # ax2.plot(sub_df['Num Added Phrases'], sub_df['Average Time'],
        #          color='green', label='Average Time')
        # ax2.set_ylabel('Average Time Per Test (s)')
        # axs[0].legend(loc='upper center')
        # ax2.legend(loc='lower center')

        axs[1].plot(sub_df['Num Added Phrases'], sub_df['Before Accuracy'],
                    label='Before')
        axs[1].plot(sub_df['Num Added Phrases'], sub_df['After Accuracy'],
                    label='After')
        axs[1].set_title('Before and After new phrase is added to ref set')
        axs[1].set_xlabel('Num Added Phrases')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_ylim([min_accuracy_scale, max_accuracy_scale])
        axs[1].legend()
        axs[1].grid()

        markers_on = []
        num_correct_new_phrases = sub_df['Num correct new phrases'].values
        for i in range(len(num_correct_new_phrases)-1):
            if num_correct_new_phrases[i] == num_correct_new_phrases[i+1]:
                markers_on.extend([i, i+1])

        axs[2].plot(sub_df['Num Added Phrases'],
                    sub_df['Num correct new phrases'], marker='o',
                    markevery=markers_on)
        axs[2].set_title('Num correct rank 1 predictions of newly added phrases')
        axs[2].set_xlabel('Num Added Phrases')
        axs[2].set_ylabel('Num correct')
        axs[2].grid()

        plt.subplots_adjust(top=0.9)
        plt.show()

    # show average of all graphs
    num_added_phrases = df['Num Added Phrases'].max()
    average_accuracies, average_before, average_after, \
        average_num_correct, average_times = [], [], [], [], []
    num_phrases_range = list(range(1, num_added_phrases+1))
    for num_phrases_so_far in num_phrases_range:
        sub_df = df[df['Num Added Phrases'] == num_phrases_so_far]
        average_accuracies.append(sub_df['Accuracy'].mean())
        average_before.append(sub_df['Before Accuracy'].mean())
        average_after.append(sub_df['After Accuracy'].mean())
        average_num_correct.append(sub_df['Num correct new phrases'].mean())
        average_times.append(sub_df['Average Time'].mean())

    min_accuracy_scale = min(average_accuracies) - 0.02
    max_accuracy_scale = max(average_accuracies) + 0.02

    # fig, axs = plt.subplots(3)
    # fig.suptitle('Average')
    # fig.tight_layout()
    #
    # axs[0].plot(num_phrases_range, average_accuracies)
    # axs[0].set_title('Accuracy')
    # axs[0].set_xlabel('Num Added Phrases')
    # axs[0].set_ylabel('Accuracy')
    # axs[0].set_ylim([min_accuracy_scale, max_accuracy_scale])
    # axs[0].grid()
    #
    # axs[1].plot(num_phrases_range, average_before, label='Before')
    # axs[1].plot(num_phrases_range, average_after, label='After')
    # axs[1].set_title('Before and After new phrase is added to ref set')
    # axs[1].set_xlabel('Num Added Phrases')
    # axs[1].set_ylabel('Accuracy')
    # axs[1].set_ylim([min_accuracy_scale, max_accuracy_scale])
    # axs[1].legend()
    # axs[1].grid()
    #
    # axs[2].plot(num_phrases_range, average_num_correct)
    # axs[2].set_title('Num correct rank 1 predictions of newly added phrases')
    # axs[2].set_xlabel('Num Added Phrases')
    # axs[2].set_ylabel('Num correct')
    # axs[2].grid()

    plt.plot(num_phrases_range, average_accuracies)
    plt.title(f'{args.user_id}: Average Accuracy over 5 folds')
    plt.xlabel('Num Added Phrases')
    plt.ylabel('Accuracy')
    plt.ylim([min_accuracy_scale, max_accuracy_scale])
    plt.grid()

    # # create figure and axis objects with subplots()
    # fig, ax = plt.subplots()
    # ax.plot(num_phrases_range, average_accuracies, label='Accuracy')
    # ax.set_xlabel('Num Added Phrases')
    # ax.set_ylabel('Accuracy')
    #
    # # add second y-axis plot
    # ax2 = ax.twinx()
    # ax2.plot(num_phrases_range, average_times, color='green',
    #          label='Average Time')
    # ax2.set_ylabel('Average Time Per Test (s)')
    # ax.legend(loc='upper center')
    # ax2.legend(loc='lower center')

    plt.show()


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def cosine_distance(t1, t2):
    vec1 = text_to_vector(t1)
    vec2 = text_to_vector(t2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    distance = float(numerator) / denominator if denominator else 0.0

    return distance


def mra_levinshtein_distance(t1, t2):
    phoneme_1 = jellyfish.match_rating_codex(t1)
    phoneme_2 = jellyfish.match_rating_codex(t2)
    distance = jellyfish.levenshtein_distance(phoneme_1, phoneme_2)

    return distance


def metaphone_levhinshtein_distance(t1, t2):
    return jellyfish.levenshtein_distance(
        jellyfish.metaphone(t1),
        jellyfish.metaphone(t2)
    )


def beider_morse_levinshtein_distance(t1, t2):
    bmpm = ap.BeiderMorse(language_arg='english')

    phonemes_1 = ' '.join([bmpm.encode(w).split(' ')[0] for w in t1.split(' ')])
    phonemes_2 = ' '.join([bmpm.encode(w).split(' ')[0] for w in t2.split(' ')])

    return jellyfish.levenshtein_distance(phonemes_1, phonemes_2)


def extract_visemes(t):
    all_visemes = []

    if '_' in t:
        split_by = '_'
        join_by = ''
    else:
        split_by = ' '
        join_by = split_by

    for word in t.split(split_by):
        word = word.lower().strip()  # lowercase and strip word
        word = re.sub(r'[\?\!]', '', word)  # replace any characters

        try:
            all_visemes.append(WORDS_TO_VISEMES[word])
        except KeyError:
            return None

    return join_by.join(all_visemes)


def viseme_levinstein_distance(t1, t2):
    v1 = extract_visemes(t1)
    v2 = extract_visemes(t2)

    if not v1 or not v2:
        return None

    return jellyfish.levenshtein_distance(v1, v2)


def is_cosine_similar(t1, t2, threshold):
    return cosine_distance(t1, t2) >= threshold


def is_mra_levinstein_similar(t1, t2, threshold):
    return mra_levinshtein_distance(t1, t2) <= threshold


def is_beider_morse_levinstein_similar(t1, t2, threshold):
    return beider_morse_levinshtein_distance(t1, t2) <= threshold


def is_metaphone_levinstein_similar(t1, t2, threshold):
    return metaphone_levhinshtein_distance(t1, t2) <= threshold


def is_viseme_levinstein_similar(t1, t2, threshold):
    distance = viseme_levinstein_distance(t1, t2)
    if distance is None:
        return None
    else:
        return distance <= threshold


def experiment_3_process_fold(user_id, phrases_to_add, repeat, training_data,
                              test_data, similarity_function, threshold):
    #  https://stackabuse.com/phonetic-similarity-of-words-a-vectorized-approach-in-python/
    print(repeat,
          len(phrases_to_add),
          sum([len(v) for v in training_data.values()]),
          len(test_data),
          similarity_function,
          threshold)

    random.shuffle(phrases_to_add)

    training_templates, test_templates = [], []
    included_training_templates, included_test_templates = [], []
    phrases_so_far = []
    previous_included_accuracies = None
    for number_added_phrases, phrase_to_add in enumerate(phrases_to_add):
        # record phrase ordering
        with open(f'phrase_list_composition_3_{user_id}_phrase_order.csv', 'a') as f:
            f.write(f'{repeat+1},{number_added_phrases+1},{phrase_to_add}\n')

        # add a phrase to training and test templates
        training_templates.extend([
            (phrase_to_add, template)
            for template in training_data[phrase_to_add]
        ])
        test_templates.append((phrase_to_add, test_data[phrase_to_add]))

        # get original accuracy
        original_accuracies = test(training_templates, test_templates)[2]

        # exclude similar phrase
        if any([similarity_function(phrase_to_add, phrase_so_far, threshold)
                for phrase_so_far in phrases_so_far]):
            # use previous accuracy
            included_accuracies = previous_included_accuracies

            with open(f'phrase_list_composition_3_{user_id}_excluded_phrases.csv',
                      'a') as f:
                f.write(f'{repeat+1},{number_added_phrases+1},{phrase_to_add}\n')
        else:
            # add a phrase to included training and test templates
            included_training_templates.extend([
                (phrase_to_add, template)
                for template in training_data[phrase_to_add]
            ])
            included_test_templates.append((phrase_to_add,
                                            test_data[phrase_to_add]))

            included_accuracies = test(included_training_templates,
                                       included_test_templates)[2]
            previous_included_accuracies = included_accuracies

            phrases_so_far.append(phrase_to_add)

        with open(
                f'phrase_list_composition_3_{user_id}_results.csv',
                'a') as f:
            f.write(f'{repeat+1},{number_added_phrases+1},{original_accuracies},'
                    f'{included_accuracies}\n')


def experiment_3(args):
    """Affect of distance metrics on accuracy
    Exclude phrases that are similar - record accuracy
    """
    if args.synthetic:
        data = get_synthetic_data(args.videos_directory, args.user_id)
    else:
        data = get_data(args.videos_directory, args.user_id)
    phrases_to_add = list(data.keys())
    random.shuffle(phrases_to_add)

    similarity_functions = {
        'cosine': is_cosine_similar,
        'mral': is_mra_levinstein_similar,
        'bml': is_beider_morse_levinstein_similar,
        'ml': is_metaphone_levinstein_similar,
        'vl': is_viseme_levinstein_similar,
    }
    similarity_function = similarity_functions[args.similarity_function]

    process_tasks = []
    for repeat in range(NUM_SESSIONS):
        training_data = {
            k: v[:repeat] + v[repeat + 1:] for k, v in data.items()
        }
        test_data = {
            k: v[repeat] for k, v in data.items()
        }

        process_tasks.append([args.user_id, phrases_to_add, repeat,
                              training_data, test_data, similarity_function,
                              args.threshold])

    num_processes = NUM_SESSIONS
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(experiment_3_process_fold, process_tasks)


def analysis_3(args):
    data = []
    regex = r'(\d+),(\d+),(\[.+\]),(\[.+\])'
    with open(args.results_file, 'r') as f:
        for line in f.readlines():
            repeat, num_added_phrases, original_accuracies, excluded_accuracies \
                = re.match(regex, line).groups()
            data.append([
                int(repeat), int(num_added_phrases),
                ast.literal_eval(original_accuracies),
                ast.literal_eval(excluded_accuracies)
            ])

    df = pd.DataFrame(data=data,
                      columns=['Repeat', 'Num Added Phrases',
                               'Original Accuracies', 'Excluded Accuracies'])

    base_path = os.path.dirname(args.results_file)

    excluded_phrases_df = pd.read_csv(
        os.path.join(base_path, f'phrase_list_composition_3_{args.user_id}_excluded_phrases.csv'),
        names=['Repeat', 'Position', 'Phrase']
    )

    phrase_order_df = pd.read_csv(
        os.path.join(base_path, f'phrase_list_composition_3_{args.user_id}_phrase_order.csv'),
        names=['Repeat', 'Order Num', 'Phrase']
    )

    repeats = df['Repeat'].unique()
    for repeat in repeats:
        sub_df = df[df['Repeat'] == repeat]

        original_ranks = [[], [], []]
        excluded_ranks = [[], [], []]
        x1 = sub_df['Num Added Phrases']
        x2, x3 = [], []
        for index, row in sub_df.iterrows():
            original_accuracies = row['Original Accuracies']
            excluded_accuracies = row['Excluded Accuracies']
            assert len(original_accuracies) == len(excluded_accuracies)

            num_ranks = len(original_accuracies)
            if num_ranks == 2:
                x2.append(row['Num Added Phrases'])
            elif num_ranks == 3:
                x2.append(row['Num Added Phrases'])
                x3.append(row['Num Added Phrases'])

            for i in range(num_ranks):
                original_ranks[i].append(original_accuracies[i])
                excluded_ranks[i].append(excluded_accuracies[i])

        xs = [x1, x2, x3]
        i = 1
        for x, original_rank, excluded_rank in zip(xs, original_ranks,
                                                   excluded_ranks):
            plt.plot(x, original_rank, label=f'Original R{i}')
            plt.plot(x, excluded_rank, label=f'Included R{i}')
            i += 1

        # add exclusion lines
        for index, row in excluded_phrases_df[excluded_phrases_df['Repeat']
                                              == repeat].iterrows():
            # plt.axvline(row['Position'], color='black')
            plt.plot(row['Position'], original_ranks[0][row['Position']-1],
                     '|', color='black', markersize='10')

        # print phrase ordering
        for index, row in phrase_order_df[phrase_order_df['Repeat'] == repeat].iterrows():
            print(row['Phrase'])
        print()

        # calculate rank 1 accuracy difference
        added_original_accuracy, added_excluded_accuracy = 0, 0
        overall_change = 0
        for index, row in sub_df.iterrows():
            original_accuracy, excluded_accuracy = \
                row['Original Accuracies'][0], row['Excluded Accuracies'][0]
            added_original_accuracy += original_accuracy
            added_excluded_accuracy += excluded_accuracy

            overall_change += (excluded_accuracy - original_accuracy)

        percentage_increase = ((added_excluded_accuracy - added_original_accuracy) / added_original_accuracy) * 100
        print('Percentage increase:', percentage_increase)
        print('Average change:', (overall_change / len(sub_df)) * 100)

        plt.title(f'{args.user_id}: Excluding similar phrases - Repeat '
                  f'{repeat}')
        plt.ylim(0, 1.01)
        plt.legend()
        plt.xlabel('Num Added Phrases')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()


def show_confusion_matrix(args):
    m = []
    phrases = list(PHRASES_LOOKUP.values())

    similarity_functions = {
        'cosine': cosine_distance,
        'mral': mra_levinshtein_distance,
        'bml': beider_morse_levinshtein_distance,
        'ml': metaphone_levhinshtein_distance
    }
    similarity_function = similarity_functions[args.similarity_function]

    for i, phrase in enumerate(list(PHRASES_LOOKUP.values())):
        print(i+1, phrase)

    for p1 in phrases:
        r = []
        for p2 in phrases:
            distance = similarity_function(p1, p2)
            r.append(distance)
        m.append(r)

    # plot confusion matrix
    df_cm = pd.DataFrame(m, index=list(PHRASES_LOOKUP.keys()),
                         columns=list(PHRASES_LOOKUP.keys()))
    plt.figure()
    sn.heatmap(df_cm)
    plt.tight_layout()
    plt.title(args.title)
    plt.show()


def show_confused_phrases(args):
    similarity_functions = {
        'cosine': is_cosine_similar,
        'mral': is_mra_levinstein_similar,
        'bml': is_beider_morse_levinstein_similar,
        'ml': is_metaphone_levinstein_similar,
        'vl': is_viseme_levinstein_similar
    }
    similarity_function = similarity_functions[args.similarity_function]

    phrases = list(PHRASES_LOOKUP.values())
    num_confused_phrases = 0
    for p1 in phrases:
        confused_with = []
        for p2 in phrases:
            if p1 != p2:
                similar = similarity_function(p1, p2, args.threshold)
                if similar is None:
                    continue
                if similar:
                    confused_with.append(p2)
        if confused_with:
            print(p1, confused_with)
            num_confused_phrases += 1
    print(f'Num confused: {num_confused_phrases}/{len(phrases)}')


def experiment_4_process_fold(user_id, repeat, phrases_to_add, training_data,
                              test_data):
    training_templates = []
    # using a constant test set
    test_templates = [
        (phrase, template) for phrase, template in test_data.items()
    ]

    for num_added_phrases, phrase_to_add in enumerate(phrases_to_add):
        # adding the specific phrase to the training set
        training_templates.extend([
            (phrase_to_add, template)
            for template in training_data[phrase_to_add]
        ])

        accuracy, time_taken = test(training_templates, test_templates)

        with open(f'phrase_list_composition_4_{user_id}_results.csv', 'a') as f:
            f.write(f'{repeat+1},{num_added_phrases+1},{accuracy},{time_taken}\n')


def experiment_4(args):
    """Use a constant test set instead of adding to it everytime"""
    data = get_data(args.videos_directory, args.user_id)
    phrases_to_add = list(data.keys())
    random.shuffle(phrases_to_add)

    process_tasks = []
    for repeat in range(NUM_SESSIONS):
        training_data = {
            k: v[:repeat] + v[repeat + 1:] for k, v in data.items()
        }
        test_data = {
            k: v[repeat] for k, v in data.items()
        }

        process_tasks.append([
            args.user_id, repeat, phrases_to_add, training_data, test_data
        ])

    num_processes = NUM_SESSIONS
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(experiment_4_process_fold, process_tasks)


def analysis_4(args):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)

    def animate(i):
        """live graph animation"""
        df = pd.read_csv(args.results_file,
                         names=['Repeat', 'Num Added Phrases', 'Accuracy',
                                'Time'])
        max_added_phrases = df['Num Added Phrases'].max()

        y = []
        for num_added_phrases in range(1, max_added_phrases+1):
            sub_df = df[df['Num Added Phrases'] == num_added_phrases]
            mean_accuracy = sub_df['Accuracy'].mean()
            y.append(mean_accuracy)

        x = df['Num Added Phrases'].unique()
        axis.clear()
        axis.plot(x, y, color='blue', label='actual')
        axis.set_xlabel('Number of added phrases')
        axis.set_ylabel('Accuracy %')
        axis.set_title(args.user_id)
        axis.set_ylim([0, 1])
        axis.set_xlim([1, max_added_phrases])
        axis.grid(True)
        axis.plot([1, max_added_phrases], [0.01, 1], color='green',
                  label='expected')  # expected accuracy line
        plt.legend()

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


def show_prediction_confused_phrases(args):
    data = get_data(args.videos_directory, args.user_id)
    all_phrases = list(data.keys())

    confused_d = {
        k1: {k2: 0 for k2 in all_phrases}
        for k1 in all_phrases
    }
    confused_matrix = []
    for i, phrase in enumerate(all_phrases):
        print(i+1, phrase)
        phrase_templates = data[phrase]
        training_templates = [
            (k, template.blob)
            for k, templates in data.items()
            for template in templates
            if k != phrase
        ]
        for test_template in phrase_templates:
            predictions = transcribe_signal(training_templates,
                                            test_template.blob,
                                            None, **DTW_PARAMS)
            first_prediction_label = predictions[0]['label']
            confused_d[phrase][first_prediction_label] += 1

        confused_matrix.append(list(confused_d[phrase].values()))

    indices = list(range(1, len(all_phrases)+1))

    # plot confusion matrix
    df_cm = pd.DataFrame(confused_matrix, index=indices, columns=indices)
    plt.figure()
    sn.heatmap(df_cm)
    plt.tight_layout()
    plt.show()


def experiment_5_process_fold(user_id, repeat, phrases_to_add, training_data,
                              test_data, is_similar, threshold):

    with open(f'phrase_list_composition_5_{user_id}_phrase_order.csv', 'a') as f:
        for i, phrase in enumerate(phrases_to_add):
            f.write(f'{repeat+1},{i+1},{phrase}\n')

    def run_original():
        original_training_templates = []
        # using a constant test set
        test_templates = [
            (phrase, template) for phrase, template in test_data.items()
        ]

        for num_added_phrases, phrase_to_add in enumerate(phrases_to_add):
            # adding the specific phrase to the training set
            original_training_templates.extend([
                (phrase_to_add, template)
                for template in training_data[phrase_to_add]
            ])
            original_accuracy, original_time_taken, original_accuracies = \
                test(original_training_templates, test_templates)

            # record original
            with open(f'phrase_list_composition_5_{user_id}_original_results.csv', 'a') as f:
                f.write(f'{repeat+1},{num_added_phrases+1},{len(phrases_to_add)},'
                        f'{phrase_to_add},{original_accuracies},'
                        f'{original_time_taken}\n')

    import threading
    original_thread = threading.Thread(target=run_original)
    original_thread.start()

    # get included phrases
    included_phrases = []
    included_test_templates = []
    excluded_phrase_indices = []
    for i, phrase_to_add in enumerate(phrases_to_add):
        any_similar = False
        for previous_phrase in included_phrases:
            similar = is_similar(phrase_to_add, previous_phrase, threshold)
            if similar is None or similar:
                any_similar = True
                # print(phrase_to_add, previous_phrase, 'are similar')
                break
        if not any_similar:
            included_phrases.append(phrase_to_add)
            included_test_templates.append(
                (phrase_to_add, test_data[phrase_to_add])
            )
        else:
            excluded_phrase_indices.append((phrase_to_add, previous_phrase, i+1))

            with open(f'phrase_list_composition_5_{user_id}_excluded_phrases.csv', 'a') as f:
                f.write(f'{repeat+1},{i+1},{phrase_to_add},{previous_phrase}\n')

    print(f'Repeat {repeat} using {len(included_phrases)}/{len(phrases_to_add)}')
    print(f'Repeat {repeat} excluded indices:', excluded_phrase_indices)

    included_training_templates = []
    for num_added_phrases, phrase_to_add in enumerate(included_phrases):
        # adding the specific phrase to the training set
        included_training_templates.extend([
            (phrase_to_add, template)
            for template in training_data[phrase_to_add]
        ])
        included_accuracy, included_time_taken, included_accuracies = \
            test(included_training_templates, included_test_templates)

        with open(f'phrase_list_composition_5_{user_id}_included_results.csv', 'a') as f:
            f.write(f'{repeat+1},{num_added_phrases+1},'
                    f'{len(included_phrases)},{phrase_to_add},'
                    f'{included_accuracies},{included_time_taken}\n')

    # main thread wait here
    original_thread.join()


def experiment_5(args):
    """Like experiment 4 only using different phrase order for each repeat
    Use a constant test set instead of adding to it everytime
    """
    if args.synthetic:
        data = get_synthetic_data(args.videos_directory, args.user_id)
    else:
        data = get_data(args.videos_directory, args.user_id)
    phrases_to_add = list(data.keys())

    similarity_functions = {
        'cosine': is_cosine_similar,
        'mral': is_mra_levinstein_similar,
        'bml': is_beider_morse_levinstein_similar,
        'ml': is_metaphone_levinstein_similar,
        'vl': is_viseme_levinstein_similar,
    }
    similarity_function = similarity_functions[args.similarity_function]

    if args.similarity_function == 'vl':
        # hack to remove phrase - "SRAVI" not in cmudict, excluding it
        phrase_to_remove = 'SRAVI is great!'
        if phrase_to_remove in data:
            del data[phrase_to_remove]
            phrases_to_add.remove(phrase_to_remove)

    process_tasks = []
    for repeat in range(NUM_SESSIONS):
        phrases_to_add_copy = phrases_to_add.copy()

        # shuffle for each repeat
        random.shuffle(phrases_to_add_copy)

        # different training and test data each time
        training_data = {
            k: v[:repeat] + v[repeat + 1:] for k, v in data.items()
        }
        test_data = {
            k: v[repeat] for k, v in data.items()
        }

        process_tasks.append([
            args.user_id, repeat, phrases_to_add_copy, training_data, test_data,
            similarity_function, args.threshold
        ])

    num_processes = NUM_SESSIONS
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(experiment_5_process_fold, process_tasks)


def analysis_5_read_csv(csv_path):
    regex = r'(\d+),(\d+),(\d+),(.+),(\[.+\]),(\d+.\d+)'

    data = []
    with open(csv_path, 'r') as f:
        for line in f.readlines():
            repeat, num_added_phrases, num_phrases, added_phrase, \
            accuracies, time_taken = re.match(regex, line).groups()
            data.append([
                int(repeat), int(num_added_phrases), int(num_phrases),
                added_phrase, ast.literal_eval(accuracies),
                float(time_taken)
            ])

    return pd.DataFrame(data=data,
                      columns=['Repeat', 'Num Added Phrases',
                               'Num Phrases', 'Added Phrase', 'Accuracies',
                               'Time Taken'])


def analysis_5(args):
    directory = os.path.dirname(args.results_file)
    user_id = os.path.basename(args.results_file).split('_')[4]
    experiment_id = os.path.basename(args.results_file).split('_')[3]

    num_rows, num_columns = 2, 3
    fig, axs = plt.subplots(num_rows, num_columns)
    fig.suptitle(args.title)

    def animate(i):
        """live graph animation"""
        df = analysis_5_read_csv(args.results_file)

        has_excluded_phrases = True
        try:
            excluded_phrases_df = pd.read_csv(
                os.path.join(directory, f'phrase_list_composition_{experiment_id}_{user_id}_excluded_phrases.csv'),
                names=['Repeat', 'Position', 'Phrase To Add', 'Similar Phrase']
            )
        except FileNotFoundError:
            has_excluded_phrases = False

        repeats = df['Repeat'].unique()
        graph_row, graph_column = 0, 0

        for repeat in repeats:
            sub_df = df[df['Repeat'] == repeat]
            max_added_phrases = sub_df['Num Added Phrases'].max()

            num_phrases = sub_df['Num Phrases'].unique()[0]

            x1 = sub_df['Num Added Phrases']
            xs = [x1, [], []]
            ys = [[], [], []]

            for i, accs in enumerate(sub_df['Accuracies']):
                num_ranks = len(accs)
                if num_ranks == 2:
                    xs[1].append(i + 1)
                elif num_ranks == 3:
                    xs[1].append(i + 1)
                    xs[2].append(i + 1)

                for j in range(len(accs)):
                    ys[j].append(accs[j])

            y4 = [j / num_phrases for j in range(1, max_added_phrases+1)]

            # absolute accuracy diff between R1 and optimal
            accuracy_diff = [
                abs(y4[i] - ys[0][i]) for i in range(max_added_phrases)
            ]
            av_accuracy_dff = \
                round((sum(accuracy_diff) / max_added_phrases) * 100, 2)

            # plot subgraph
            axs[graph_row, graph_column].clear()
            for _x, y, label, color in zip(xs, ys,
                                           ['R1', 'R2', 'R3'],
                                           ['blue', 'brown', 'black']):
                rank_line = axs[graph_row, graph_column].plot(_x, y, color=color, label=f'actual {label}')
            axs[graph_row, graph_column].set_xlabel('Number of added phrases')
            axs[graph_row, graph_column].set_ylabel('Accuracy %')
            axs[graph_row, graph_column].set_title(f'Repeat {repeat} - Av. Diff {av_accuracy_dff}%')
            axs[graph_row, graph_column].set_ylim([0, 1])
            axs[graph_row, graph_column].set_xlim([1, max_added_phrases])
            axs[graph_row, graph_column].grid(True)
            axs[graph_row, graph_column].plot(x1, y4, color='green', label='optimal')
            axs[graph_row, graph_column].plot(x1, accuracy_diff, color='red', label='R1 accuracy drop')

            # show markers for excluded phrases
            if has_excluded_phrases:
                excluded_phrases_sub_df = excluded_phrases_df[excluded_phrases_df['Repeat'] == repeat]
                x = [p for p in excluded_phrases_sub_df['Position'].values if p <= max_added_phrases]
                y = [
                    accuracy_diff[position-1]
                    for position in x # find rank 1 accuracy by position
                ]
                axs[graph_row, graph_column].plot(x, y, color='red', linestyle='', marker='|', label='excluded phrase markers', markersize=15)

            # proportional_y = [round(i/100, 2) for i in range(1, max_added_phrases+1)]
            # proportional_diff = [
            #     abs(proportional_y[i] - ys[0][i]) for i in range(max_added_phrases)
            # ]
            # axs[graph_row, graph_column].plot(x1, proportional_diff, color='purple',
            #                                   label='proportional diff')

            axs[graph_row, graph_column].legend()

            if graph_column == num_columns - 1:
                graph_row += 1
                graph_column = 0
            else:
                graph_column += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()


def analysis_5_part_2(args):
    """Show absolute accuracy diff between original and included
    Diff between R1 and optimal"""
    user_id = os.path.basename(args.original_results).split('_')[4]

    original_df = analysis_5_read_csv(args.original_results)
    included_df = analysis_5_read_csv(args.included_results)

    num_rows, num_columns = 2, 3
    fig, axs = plt.subplots(num_rows, num_columns)
    fig.suptitle(user_id)
    graph_row, graph_column = 0, 0

    for repeat in original_df['Repeat'].unique():
        original_sub_df = original_df[original_df['Repeat'] == repeat]
        included_sub_df = included_df[included_df['Repeat'] == repeat]

        xs = [
            original_sub_df['Num Added Phrases'],
            included_sub_df['Num Added Phrases']
        ]

        ys = []
        for df in [original_sub_df, included_sub_df]:
            num_phrases = df['Num Added Phrases'].max()
            optimal_y = [(j+1) / num_phrases for j in range(num_phrases)]
            rank1_y = [acc[0] for acc in df['Accuracies']]
            accuracy_diff = [
                abs(optimal_y[i] - rank1_y[i]) for i in range(num_phrases)
            ]

            ys.append(accuracy_diff)

        for x, y, label in zip(xs, ys, ['Original', 'Included Only']):
            axs[graph_row, graph_column].plot(x, y, label=label)
        axs[graph_row, graph_column].legend()
        axs[graph_row, graph_column].set_ylim(0, 1.01)
        axs[graph_row, graph_column].set_xlim(1, original_sub_df['Num Added Phrases'].max())
        axs[graph_row, graph_column].set_xlabel('Num Added Phrases')
        axs[graph_row, graph_column].set_ylabel('Accuracy Drop %')
        axs[graph_row, graph_column].set_title(f'Repeat {repeat}')
        axs[graph_row, graph_column].grid(True)

        if graph_column == num_columns - 1:
            graph_row += 1
            graph_column = 0
        else:
            graph_column += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def analysis_5_part_3(args):
    directory = os.path.dirname(args.original_results_file)
    user_id = os.path.basename(args.original_results_file).split('_')[4]
    experiment_id = os.path.basename(args.original_results_file).split('_')[3]

    original_df = analysis_5_read_csv(args.original_results_file)
    included_df = analysis_5_read_csv(args.included_results_file)
    excluded_df = pd.read_csv(
        os.path.join(directory,
                     f'phrase_list_composition_{experiment_id}_{user_id}_excluded_phrases.csv'),
        names=['Repeat', 'Position', 'Phrase To Add', 'Similar Phrase']
    )

    def plot_accuracy_graph(ax, _df, _title, _excluded_df):
        xs = [[], [], []]
        ys = [[], [], []]
        max_added_phrases = _df['Num Added Phrases'].max()
        num_phrases = _df['Num Phrases'].unique()[0]
        for index, row in _df.iterrows():
            accuracies = row['Accuracies']
            num_ranks = len(accuracies)
            for i in range(num_ranks):
                xs[i].append(row['Num Added Phrases'])
                ys[i].append(accuracies[i])
        for _x, y, label, color in zip(xs, ys,
                                       ['R1', 'R2', 'R3'],
                                       ['blue', 'brown', 'black']):
            ax.plot(_x, y, color=color, label=f'actual {label}')

        y4 = [j / num_phrases for j in range(1, max_added_phrases + 1)]

        # absolute accuracy diff between R1 and optimal
        accuracy_diff = [
            abs(y4[i] - ys[0][i]) for i in range(max_added_phrases)
        ]
        av_accuracy_dff = \
            round((sum(accuracy_diff) / max_added_phrases) * 100, 2)

        # add markers for excluded phrases
        x = [p for p in _excluded_df['Position'].values if
             p <= max_added_phrases]
        y = [
            accuracy_diff[position - 1]
            for position in x  # find rank 1 accuracy by position
        ]
        ax.plot(x, y, color='red', linestyle='', marker='|',
                label='excluded phrase markers', markersize=15)
        ax.plot(xs[0], y4, color='green', label='optimal')
        ax.plot(xs[0], accuracy_diff, color='red', label='R1 accuracy drop')
        ax.set_xlabel('Number of added phrases')
        ax.set_ylabel('Accuracy %')
        ax.set_ylim([0, 1])
        ax.set_title(_title)
        ax.set_xlim([1, max_added_phrases])
        ax.grid(True)
        ax.legend()

    def plot_time_graph(ax, original_df, included_df):
        xs = [original_df['Num Added Phrases'], included_df['Num Added Phrases']]
        ys = [original_df['Time Taken'], included_df['Time Taken']]
        labels = ['Original', 'Included Only']
        for x, y, label in zip(xs, ys, labels):
            ax.plot(x, y, label=label)
        ax.legend()
        ax.set_xlabel('Number of added phrases')
        ax.set_ylabel('Time Taken (s)')
        ax.grid(True)
        ax.set_title('Time Taken')

    for repeat in original_df['Repeat'].unique():
        original_sub_df = original_df[original_df['Repeat'] == repeat]
        included_sub_df = included_df[included_df['Repeat'] == repeat]
        excluded_sub_df = excluded_df[excluded_df['Repeat'] == repeat]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        plot_accuracy_graph(ax1, original_sub_df, 'Original Accuracies', excluded_sub_df)
        plot_accuracy_graph(ax2, included_sub_df, 'Included Only Accuracies', excluded_sub_df)
        plot_time_graph(ax3, original_sub_df, included_sub_df)
        fig.suptitle(f'{user_id} - Repeat {repeat}')
        plt.show()


def experiment_6(args):
    """similar to experiment 1 - incremental num phrases"""
    data = get_data(args.videos_directory, args.user_id)
    all_phrases = list(data.keys())
    random.shuffle(all_phrases)

    num_phrases = len(all_phrases)
    for i in range(5, num_phrases+1, 5):
        phrases_this_round = all_phrases[:i]
        assert len(phrases_this_round) == i

        subset_data = {k: v for k, v in data.items()
                       if k in phrases_this_round}

        num_training_templates_this_round = i * (NUM_SESSIONS - 1)
        num_test_templates_this_round = i

        average_accuracy = 0
        for repeat in range(NUM_SESSIONS):
            training_data = {
                k: v[:repeat] + v[repeat + 1:] for k, v in subset_data.items()
            }
            test_data = {
                k: v[repeat] for k, v in subset_data.items()
            }

            training_templates = [
                (phrase, template)
                for phrase, templates in training_data.items()
                for template in templates
            ]
            test_templates = [
                (phrase, template) for phrase, template in test_data.items()
            ]

            assert len(training_templates) == num_training_templates_this_round
            assert len(test_templates) == num_test_templates_this_round

            accuracy = test(training_templates, test_templates)[0]
            average_accuracy += accuracy

        average_accuracy /= NUM_SESSIONS
        average_accuracy = round(average_accuracy, 2)

        print('Round:', i, num_training_templates_this_round,
              num_test_templates_this_round, average_accuracy)

        with open(f'phrase_list_composition_6_{args.user_id}_results.csv', 'a') as f:
            f.write(f'{i},{num_training_templates_this_round},'
                    f'{num_test_templates_this_round},'
                    f'{average_accuracy}\n')


def analysis_6(args):
    user_id = os.path.basename(args.results_file).split('_')[4]

    df = pd.read_csv(args.results_file,
                     names=['Num Added Phrases', 'Num Training Templates',
                            'Num Test Templates', 'Accuracy'])

    plt.plot(df['Num Added Phrases'], df['Accuracy'])
    plt.ylim((0, 1.01))
    plt.xlim((0, 100))
    plt.xlabel('Num Added Phrases')
    plt.ylabel('Accuracy %')
    plt.title(user_id)
    plt.show()


def experiment_7(args):
    """comparing 1-3 rank accuracies with adding/removing similar phrases"""
    data = get_data(args.videos_directory, args.user_id)
    all_phrases = list(data.keys())

    # hack to remove phrase - "SRAVI" not in cmudict, excluding it
    phrase_to_remove = 'SRAVI is great!'
    del data[phrase_to_remove]
    all_phrases.remove(phrase_to_remove)

    is_similar = {
        'vl': is_viseme_levinstein_similar
    }[args.similarity_function]

    def get_accuracies(_training_phrases, _test_phrases, _training_data,
                       _test_data):
        training_templates = [
            (p, template)
            for p, templates in _training_data.items()
            for template in templates
            if p in _training_phrases
        ]
        test_templates = [
            (p, template)
            for p, template in _test_data.items()
            if p in _test_phrases
        ]

        rank_accuracies = test(training_templates, test_templates)[2]

        return rank_accuracies

    for repeat in range(NUM_SESSIONS):
        random.shuffle(all_phrases)

        training_data = {
            k: v[:repeat] + v[repeat + 1:] for k, v in data.items()
        }
        test_data = {
            k: v[repeat] for k, v in data.items()
        }

        added_phrases = []
        for i, phrase_to_add in enumerate(all_phrases):
            any_similar = False
            for previous_phrase in added_phrases:
                if is_similar(phrase_to_add, previous_phrase, args.threshold):
                    any_similar = True
                    break
            if any_similar:
                before_phrases = added_phrases.copy()
                after_phrases = before_phrases + [phrase_to_add]

                before_accuracies = get_accuracies(
                    _training_phrases=before_phrases,
                    _test_phrases=before_phrases,
                    _training_data=training_data,
                    _test_data=test_data
                )
                after_accuracies = get_accuracies(
                    _training_phrases=after_phrases,
                    _test_phrases=before_phrases,
                    _training_data=training_data,
                    _test_data=test_data
                )

                print(f'Repeat {repeat+1} - "{phrase_to_add}" similar to "{previous_phrase}" at {i+1}. {before_accuracies}, {after_accuracies}')
                with open(f'phrase_list_composition_7_{args.user_id}_results.csv', 'a') as f:
                    f.write(f'{repeat+1},{i+1},{phrase_to_add},{previous_phrase},{before_accuracies},{after_accuracies}\n')

            added_phrases.append(phrase_to_add)


def analysis_7(args):
    user_id = os.path.basename(args.results_file).split('_')[4]

    regex = r'(\d+),(\d+),(.+),(.+),(\[.+\]),(\[.+\])'
    columns = ['Repeat', 'Phrase Num', 'Phrase to Add', 'Similar Phrase',
               'Before Acc', 'After Acc']
    data = []
    with open(args.results_file, 'r') as f:
        for line in f.readlines():
            repeat, phrase_num, phrase_to_add, similar_phrase, before_acc, \
            after_acc = re.match(regex, line).groups()
            data.append([int(repeat), int(phrase_num), phrase_to_add,
                         similar_phrase, ast.literal_eval(before_acc),
                         ast.literal_eval(after_acc)])

    df = pd.DataFrame(data=data, columns=columns)

    num_repeats = df['Repeat'].unique()
    for repeat in num_repeats:
        sub_df = df[df['Repeat'] == repeat]

        before_ranks, after_ranks = [[], [], []], [[], [], []]
        tick_labels = []
        for index, row in sub_df.iterrows():
            before_acc, after_acc = row['Before Acc'], row['After Acc']

            for i in range(3):
                before_ranks[i].append(before_acc[i])
                after_ranks[i].append(after_acc[i])

            tick_labels.append(f'Position {row["Phrase Num"]}\n'
                               f'Phrase to add: {row["Phrase to Add"]}\n'
                               f'Similar: {row["Similar Phrase"]}')

        x = np.arange(1, len(sub_df)+1)
        colors = ['red', 'green', 'blue']
        for i, (before_rank, after_rank) in enumerate(zip(before_ranks,
                                                          after_ranks)):
            plt.plot(x, before_rank, linestyle='', marker='o', color=colors[i])
            plt.plot(x, after_rank, linestyle='', marker='x', color=colors[i])

        plt.title(f'Repeat {repeat} - {user_id}')
        plt.legend(['Before', 'After'])
        plt.ylim(0, 101)
        plt.xticks(x, tick_labels)
        plt.show()


def extract_similar_phrases(args):
    """extract similar phrases to create more synthetic with Wav2Lip"""
    all_phrases = []
    for num, phrase in PHRASES_LOOKUP.items():
        similar_phrases = find_similar_phrases(phrase,
                                               args.num_alternative_phrases)
        if not len(similar_phrases):
            continue
        all_phrases.extend([phrase] + similar_phrases)

    with open('phrase_list_composition_extract_similar_phrases.csv', 'a') as f:
        for phrase in all_phrases:
            f.write(f'{phrase}\n')


def get_phrases_by(args):
    """generate random phrases for phrase length experiments
    generate by number of words, syllables or phonemes
    """
    count_functions = {
        'words': lambda phrase: len(phrase.split(' ')),
        'syllables': lambda phrase: get_num_syllables(phrase),
        'phonemes': lambda phrase: get_num_phonemes(phrase)
    }
    count_function = count_functions[args.count_function]

    random_words = []
    with open('cmudict-en-us.dict', 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            word = line[0]
            if '(' in word: continue
            random_words.append(word)

    for count in range(1, args.max_length+1):
        print(f'Finding {args.phrases_per_length} phrases with {count} {args.count_function}...')
        phrases = []
        while len(phrases) != args.phrases_per_length:
            random_phrase = ' '.join(random.sample(random_words,
                                                   random.randint(1, 100)))
            if count_function(random_phrase) == count:
                phrases.append(random_phrase)

        with open(f'phrase_list_composition_phrases_by_{args.count_function}_{args.max_length}_{args.phrases_per_length}.csv', 'a') as f:
            for phrase in phrases:
                f.write(f'{phrase}\n')


def get_syllable_count(word):
    """get syllables of word"""
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    num_syllables = 0

    word = word.lower()
    if word[0] in vowels:
        num_syllables += 1
        start_at = 1
    else:
        start_at = 0

    for i in range(start_at, len(word)-1):
        if word[i] not in vowels and word[i+1] in vowels:
            num_syllables += 1
    if word[-1] == 'e':
        num_syllables -= 1
    if word[-2:] == 'le' or num_syllables == 0:
        num_syllables += 1

    return num_syllables


def get_phrases_by_num_words(phrases, num_words):
    return [phrase for phrase in phrases if len(phrase.split(' ')) == num_words]


def get_num_syllables(phrase):
    return sum([get_syllable_count(word) for word in phrase.split(' ')])


def get_num_phonemes(phrase):
    return len(jellyfish.metaphone(phrase))


def experiment_8_process_fold(repeat, phrases, training_data, test_data, user_id):
    random.shuffle(phrases)
    test_templates = [
        (label, template) for label, template in test_data.items()
    ]

    ref_templates = []
    total_num_syllables, total_num_words = 0, 0
    for i, phrase_to_add in enumerate(phrases):
        ref_templates.extend(
            [(phrase_to_add, template) for template in training_data[phrase_to_add]]
        )

        total_num_syllables += get_num_syllables(phrase_to_add)
        total_num_words += len(phrase_to_add.split(' '))
        num_ref_templates = len(ref_templates)

        rank_1, time_taken, accuracies = test(ref_templates, test_templates)

        with open(f'phrase_list_composition_8_{user_id}_accuracy_results.csv', 'a') as f:
            f.write(f'{repeat},{i+1},{phrase_to_add},{total_num_syllables},{total_num_words},{num_ref_templates},{accuracies},{time_taken}\n')


def experiment_8(args):
    """phrase length vs time taken and accuracy
    also how does phrase length affect CFE?
    phrase length = # words
    """

    # get histogram of previous phrase word counts
    if not args.phrase_list:
        args.phrase_list = list(PHRASES_LOOKUP.values())
    phrase_lengths = []
    d = {}
    for phrase in args.phrase_list:
        phrase_length = len(phrase.split(' '))
        phrase_lengths.append(phrase_length)
        d[phrase_length] = d.get(phrase_length, 0) + 1
    x = np.arange(1, len(d)+2) - 0.5
    plt.hist(phrase_lengths, bins=x, ec='k')
    plt.title('Word count of original 100 phrases')
    plt.xlabel('# Words')
    plt.ylabel('Count')
    plt.xticks([i for i in range(1, len(x)+1)])
    plt.show()

    if args.synthetic:
        data = get_synthetic_data(args.videos_directory, args.user_id)
    else:
        data = get_data(args.videos_directory, args.user_id)
    print(set(args.phrase_list) - set(data.keys()))

    for num_words in range(1, len(d)+1):
        phrases = get_phrases_by_num_words(args.phrase_list, num_words)

        for phrase in phrases:
            phrase_data = \
                data.get(phrase, data.get('_'.join(phrase.lower().split(' '))))
            if not phrase_data: continue
            average_time = 0
            for i in range(len(phrase_data)):
                test_template = phrase_data[i]
                ref_templates = phrase_data[:i] + phrase_data[i+1:]
                assert 1 + len(ref_templates) == len(phrase_data)
                time_taken = test(ref_templates, [test_template])[1]
                average_time += time_taken
            average_time /= len(phrase_data)
            num_syllables = get_num_syllables(phrase)
            with open(f'phrase_list_composition_8_{args.user_id}_time_results.csv', 'a') as f:
                f.write(f'{phrase},{num_words},{num_syllables},{average_time}\n')

    # tasks = []
    # for repeat in range(NUM_SESSIONS):
    #     # different training and test data each time
    #     training_data = {
    #         k: v[:repeat] + v[repeat + 1:] for k, v in data.items()
    #     }
    #     test_data = {
    #         k: v[repeat] for k, v in data.items()
    #     }
    #
    #     tasks.append([repeat+1, phrases, training_data, test_data, args.user_id])
    #
    # num_processes = NUM_SESSIONS
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     pool.starmap(experiment_8_process_fold, tasks)


def analysis_8(args):
    # regex = r'(\d+),(\d+),(.+),(\d+),(\d+),(\d+),(\[.+\]),(\d+.\d+)'
    # data = []
    # columns = ['Repeat', 'Phrase Number', 'Phrase', 'Num Syllables',
    #            'Num Words', 'Num Ref Templates', 'Accuracies', 'Time Taken']
    # with open(args.results_file, 'r') as f:
    #     for line in f.readlines():
    #         repeat, phrase_number, phrase, total_num_syllables, \
    #             total_num_words, num_ref_templates, accuracies, time_taken = \
    #             re.match(regex, line).groups()
    #         data.append([
    #             int(repeat), int(phrase_number), phrase,
    #             int(total_num_syllables), int(total_num_words),
    #             int(num_ref_templates), ast.literal_eval(accuracies),
    #             float(time_taken)
    #         ])
    #
    # df = pd.DataFrame(data=data, columns=columns)
    # num_repeats = df['Repeat'].unique()
    #
    # def two_scales(ax1, x, data1, data2, c1, c2, ylabel):
    #     ax2 = ax1.twinx()
    #     ax1.plot(x, data1, color=c1)
    #     ax1.set_ylim((0, 300))
    #     ax1.set_xlabel('Num Added Phrases')
    #     ax1.set_ylabel(ylabel)
    #     ax2.plot(x, data2, color=c2)
    #     ax2.set_ylabel('Time Taken')
    #     return ax1, ax2
    #
    # for repeat in num_repeats:
    #     sub_df = df[df['Repeat'] == repeat]
    #
    #     r1_accuracy = [row['Accuracies'][0] for index, row in sub_df.iterrows()]
    #     phrase_number = sub_df['Phrase Number']
    #     num_ref_templates = sub_df['Num Ref Templates']
    #     num_words = sub_df['Num Words']
    #     num_syllables = sub_df['Num Syllables']
    #     time_taken = sub_df['Time Taken']
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(phrase_number, num_ref_templates, label='Ref Templates')
    #     ax.plot(phrase_number, num_words, label='Words')
    #     ax.plot(phrase_number, num_syllables, label='Syllables')
    #     ax.set_xlabel('Num Phrases')
    #     ax.set_ylabel('Count')
    #     ax2 = ax.twinx()
    #     ax2.plot(phrase_number, time_taken, color='red')
    #     ax2.set_ylabel('Time Taken (s)', color='red')
    #     ax.legend()
    #     plt.show()
    #
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #     ax1.plot(num_ref_templates, time_taken)
    #     ax2.plot(num_syllables, time_taken)
    #     ax3.plot(num_words, time_taken)
    #     for ax, xlabel in zip([ax1, ax2, ax3], ['# Ref Templates', '# Syllables', '# Words']):
    #         ax.set_ylabel('Time Taken (s)')
    #     plt.show()

    df = pd.read_csv(args.results_file,
                     names=['Phrase', 'Num Words', 'Num Syllables',
                            'Time Taken'])

    unique_num_words = sorted(df['Num Words'].unique())
    average_times = []
    for num_words in unique_num_words:
        average_times.append(df[df['Num Words'] == num_words]['Time Taken'].mean())

    plt.plot(unique_num_words, average_times)
    plt.xlabel('Num Words')
    plt.ylabel('Average Time Taken (s)')
    plt.show()

    unique_num_syllables = sorted(df['Num Syllables'].unique())
    average_times = []
    for num_syllables in unique_num_syllables:
        average_times.append(df[df['Num Syllables'] == num_syllables]['Time Taken'].mean())

    plt.plot(unique_num_syllables, average_times)
    plt.xlabel('Num Syllables')
    plt.ylabel('Average Time Taken (s)')
    plt.show()


def plot_histogram(d, counts, x_label):
    x = np.arange(1, len(d)+2) - 0.5
    plt.hist(counts, bins=x, ec='k')
    plt.ylabel('count')
    plt.xlabel(x_label)
    plt.xticks([i for i in range(1, len(x)+1)])
    plt.show()


def experiment_9(args):
    if not args.phrase_list:
        args.phrase_list = list(PHRASES_LOOKUP.values())

    # get real or synthetic data
    if args.synthetic:
        data = get_synthetic_data(args.videos_directory, args.user_id)
    else:
        data = get_data(args.videos_directory, args.user_id)

    # counting frequency using different count functions
    d_count = {}
    counts = []
    count_functions = {
        'words': lambda phrase: len(phrase.split(' ')),
        'syllables': lambda phrase: get_num_syllables(phrase),
        'phonemes': lambda phrase: get_num_phonemes(phrase)
    }
    count_function = count_functions[args.count_function]
    for phrase in args.phrase_list:
        count_result = count_function(phrase)
        d_count[count_result] = d_count.get(count_result, 0) + 1
        counts.append(count_result)

    plot_histogram(d_count, counts, f'# {args.count_function}')

    to_filename = lambda phrase: \
        re.sub(r'[\?\!]', '', phrase.lower().strip().replace(' ', '_'))
    for count in d_count.keys():
        phrases_at_length = [
            phrase for phrase in args.phrase_list
            if count_function(phrase) == count
        ]
        num_phrases_at_length = len(phrases_at_length)

        d_templates = {}
        for phrase in phrases_at_length:
            phrase_data = data.get(phrase, data.get(to_filename(phrase)))
            if not phrase_data or len(phrase_data) < NUM_SESSIONS: continue
            d_templates[phrase] = phrase_data

        if len(d_templates) != num_phrases_at_length:
            print(f'There are only {len(d_templates)} phrases with {count} {args.count_function} instead of {num_phrases_at_length}...skipping')
            continue
        else:
            print(f'Running {count} {args.count_function}...')
            for repeat in range(NUM_SESSIONS):
                training_data = {
                    phrase: templates[:repeat] + templates[repeat+1:]
                    for phrase, templates in d_templates.items()
                }
                test_data = {
                    phrase: templates[repeat]
                    for phrase, templates in d_templates.items()
                }
                training_templates = [(phrase, template)
                                      for phrase, templates in training_data.items()
                                      for template in templates]
                test_templates = [(phrase, template)
                                  for phrase, template in test_data.items()]
                accuracy, time_taken, accuracies = test(training_templates,
                                                        test_templates)
                with open(f'phrase_list_composition_9_{args.user_id}_{args.count_function}_results.csv', 'a') as f:
                    f.write(f'{count},{num_phrases_at_length},{repeat+1},{len(training_templates)},{len(test_templates)},{time_taken},{accuracies}\n')


def experiment_9_by_count(args):
    if not args.phrase_list:
        args.phrase_list = list(PHRASES_LOOKUP.values())

    count_functions = {
        'words': lambda phrase: len(phrase.split(' ')),
        'syllables': lambda phrase: get_num_syllables(phrase),
        'phonemes': lambda phrase: get_num_phonemes(phrase)
    }

    d_count = {}
    counts = []
    count_function = count_functions[args.count_function]
    for phrase in args.phrase_list:
        count_result = count_function(phrase)
        d_count[count_result] = d_count.get(count_result, 0) + 1
        counts.append(count_result)

    plot_histogram(d_count, counts, f'# {args.count_function}')

    def append_template(count_d, phrase, template):
        templates = count_d.get(phrase, [])
        templates.append(template)
        count_d[phrase] = templates

    def is_phrase_similar(phrases_so_far, phrase, threshold):
        for phrase_so_far in phrases_so_far:
            if is_viseme_levinstein_similar(phrase_so_far, phrase, threshold):
                return True
        return False

    def _get_data(_count_function, _videos_directory, _count_this_round):
        count_d = {}
        if args.synthetic:
            df = pd.read_csv(os.path.join(_videos_directory, 'groundtruth.csv'),
                             names=['video_name', 'said', 'actual'])
            for index, row in tqdm(df.iterrows()):
                phrase = row['actual']
                count = _count_function(phrase)
                if count == _count_this_round:
                    video_path = os.path.join(_videos_directory, row['video_name'])
                    template = create_template(video_path)
                    if not template: continue
                    append_template(count_d, phrase, template)
        else:
            video_paths = glob.glob(os.path.join(_videos_directory, '*.mp4'))
            for video_path in tqdm(video_paths):
                base_name = os.path.basename(video_path)
                user_id, phrase_id, session_id = \
                    re.match(VIDEO_NAME_REGEX, base_name).groups()
                phrase = PHRASES_LOOKUP.get(int(phrase_id))
                count = _count_function(phrase)
                if count == _count_this_round:
                    template = create_template(video_path)
                    if not template: continue
                    append_template(count_d, phrase, template)
        return {
            phrase: templates for phrase, templates in count_d.items()
            if len(templates) == NUM_SESSIONS
        }

    for count in range(args.start_from_count, args.max_length+1):
        count_data = _get_data(count_functions[args.count_function],
                               args.videos_directory,
                               count)
        num_phrases_at_length = len(count_data)
        if num_phrases_at_length < args.phrases_per_length \
                or num_phrases_at_length == 0:
            continue
        if args.similarity_threshold:
            included_phrases = []
            for phrase in count_data.keys():
                if not is_phrase_similar(included_phrases, phrase, args.similarity_threshold):
                    included_phrases.append(phrase)
        else:
            included_phrases = list(count_data.keys())
        num_phrases_at_length = len(included_phrases)
        print(f'Count {count}, Num Phrases, {num_phrases_at_length}')
        for repeat in range(NUM_SESSIONS):
            training_data = {
                phrase: templates[:repeat] + templates[repeat+1:]
                for phrase, templates in count_data.items()
                if phrase in included_phrases
            }
            test_data = {
                phrase: templates[repeat]
                for phrase, templates in count_data.items()
                if phrase in included_phrases
            }
            training_templates = [(phrase, template)
                                  for phrase, templates in training_data.items()
                                  for template in templates]
            test_templates = [(phrase, template)
                              for phrase, template in test_data.items()]
            accuracy, time_taken, accuracies = test(training_templates,
                                                    test_templates,
                                                    args.max_num_vectors)
            with open(f'phrase_list_composition_9_{args.user_id}_{args.count_function}_results.csv', 'a') as f:
                f.write(f'{count},{num_phrases_at_length},{repeat+1},{len(training_templates)},{len(test_templates)},{time_taken},{accuracies}\n')


def analysis_9(args):
    count_function = os.path.basename(args.results_file).split('_')[5]
    regex = r'(\d+),(\d+),(\d+),(\d+),(\d+),(\d+.\d+),(\[.+\])'
    data = []
    columns = ['Count', 'Num Phrases At Count', 'Repeat', 'Num Training',
               'Num Test', 'Time Taken', 'Accuracies']
    with open(args.results_file, 'r') as f:
        for line in f.readlines():
            count, num_phrases_at_count, repeat, num_training, num_test, \
                time_taken, accuracies = re.match(regex, line).groups()
            data.append([
                int(count), int(num_phrases_at_count), int(repeat),
                int(num_training), int(num_test),
                float(time_taken), ast.literal_eval(accuracies),
            ])

    df = pd.DataFrame(data=data, columns=columns)

    x = df['Count'].unique()
    num_phrases_at_count = []
    xs = [[], [], []]
    ys = [[], [], []]
    av_time_taken = []
    for count in df['Count'].unique():
        sub_df = df[df['Count'] == count]
        num_ranks = len(sub_df.iloc[0]['Accuracies'])
        num_phrases_at_count.append(sub_df.iloc[0]['Num Phrases At Count'])
        av_ranks = [0] * num_ranks
        # average rank accuracies over every repeat
        for index, row in sub_df.iterrows():
            for i, acc in enumerate(row['Accuracies']):
                av_ranks[i] += acc
        av_ranks = [r / len(sub_df) for r in av_ranks]
        for i in range(num_ranks):
            xs[i].append(count)
            ys[i].append(av_ranks[i])
        times = [row['Time Taken'] for index, row in sub_df.iterrows()]
        av_time = sum(times) / len(times)
        av_time_taken.append(av_time)

    # plot rank accuracies
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.plot(x, y, label=f'R{i+1}', marker='x')
    plt.ylim((0, 1.01))
    plt.legend()
    plt.ylabel('Accuracy %')
    plt.xlabel(f'# {count_function} / # phrases per count')
    plt.title(f'Increasing # {count_function}')
    plt.xticks(xs[0], [f'{count}\n{num_phrases}' for count, num_phrases in zip(xs[0], num_phrases_at_count)])
    plt.grid()
    plt.tight_layout()
    plt.show()

    # plot time taken
    plt.plot(xs[0], av_time_taken, marker='x')
    plt.legend()
    plt.ylabel('Time Taken (s)')
    plt.xlabel(f'# {count_function} / # phrases per count')
    plt.title(f'Increasing # {count_function}')
    plt.xticks(xs[0], [f'{count}\n{num_phrases}' for count, num_phrases in zip(xs[0], num_phrases_at_count)])
    plt.grid()
    plt.tight_layout()
    plt.show()


def main(args):
    {
        'experiment_1': experiment_1,
        'analysis_1_a': analysis_1_a,
        'analysis_1_b': analysis_1_b,
        'experiment_2': experiment_2,
        'analysis_2': analysis_2,
        'experiment_3': experiment_3,
        'analysis_3': analysis_3,
        'show_confusion_matrix': show_confusion_matrix,
        'show_confused_phrases': show_confused_phrases,
        'experiment_4': experiment_4,
        'analysis_4': analysis_4,
        'show_prediction_confused_phrases': show_prediction_confused_phrases,
        'experiment_5': experiment_5,
        'analysis_5': analysis_5,
        'analysis_5_part_2': analysis_5_part_2,
        'analysis_5_part_3': analysis_5_part_3,
        'experiment_6': experiment_6,
        'analysis_6': analysis_6,
        'experiment_7': experiment_7,
        'analysis_7': analysis_7,
        'extract_similar_phrases': extract_similar_phrases,
        'get_phrases_by': get_phrases_by,
        'experiment_8': experiment_8,
        'analysis_8': analysis_8,
        'experiment_9': experiment_9,
        'experiment_9_by_count': experiment_9_by_count,
        'analysis_9': analysis_9
    }[args.run_type](args)


def file_list(s):
    with open(s, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    similarity_functions = ['cosine', 'mral', 'bml', 'ml', 'vl']
    count_functions = ['words', 'syllables', 'phonemes']

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('experiment_1')
    parser_1.add_argument('videos_directory')
    parser_1.add_argument('user_id')

    parser_2 = sub_parsers.add_parser('analysis_1_a')
    parser_2.add_argument('results_file')
    parser_2.add_argument('title')

    parser_3 = sub_parsers.add_parser('experiment_2')
    parser_3.add_argument('videos_directory')
    parser_3.add_argument('user_id')

    parser_4 = sub_parsers.add_parser('analysis_2')
    parser_4.add_argument('results_file')
    parser_4.add_argument('user_id')

    parser_5 = sub_parsers.add_parser('experiment_3')
    parser_5.add_argument('videos_directory')
    parser_5.add_argument('similarity_function', choices=similarity_functions)
    parser_5.add_argument('threshold', type=float)
    parser_5.add_argument('user_id')
    parser_5.add_argument('--synthetic', type=bool, default=False)

    parser_6 = sub_parsers.add_parser('analysis_3')
    parser_6.add_argument('results_file')
    parser_6.add_argument('user_id')

    parser_7 = sub_parsers.add_parser('show_phonetic_confusion_matrix')
    parser_7.add_argument('similarity_function', choices=similarity_functions)
    parser_7.add_argument('title')

    parser_8 = sub_parsers.add_parser('show_confused_phrases')
    parser_8.add_argument('similarity_function', choices=similarity_functions)
    parser_8.add_argument('threshold', type=float)

    parser_9 = sub_parsers.add_parser('experiment_4')
    parser_9.add_argument('videos_directory')
    parser_9.add_argument('user_id')

    parser_10 = sub_parsers.add_parser('analysis_4')
    parser_10.add_argument('results_file')
    parser_10.add_argument('user_id')

    parser_11 = sub_parsers.add_parser('show_prediction_confused_phrases')
    parser_11.add_argument('videos_directory')
    parser_11.add_argument('user_id')

    parser_12 = sub_parsers.add_parser('experiment_5')
    parser_12.add_argument('videos_directory')
    parser_12.add_argument('similarity_function', choices=similarity_functions)
    parser_12.add_argument('threshold', type=float)
    parser_12.add_argument('user_id')
    parser_12.add_argument('--synthetic', type=bool, default=False)

    parser_13 = sub_parsers.add_parser('analysis_5')
    parser_13.add_argument('results_file')
    parser_13.add_argument('title')

    parser_14 = sub_parsers.add_parser('experiment_6')
    parser_14.add_argument('videos_directory')
    parser_14.add_argument('user_id')

    parser_15 = sub_parsers.add_parser('analysis_6')
    parser_15.add_argument('results_file')

    parser_16 = sub_parsers.add_parser('experiment_7')
    parser_16.add_argument('videos_directory')
    parser_16.add_argument('user_id')
    parser_16.add_argument('similarity_function', choices=similarity_functions)
    parser_16.add_argument('threshold', type=float)

    parser_17 = sub_parsers.add_parser('analysis_7')
    parser_17.add_argument('results_file')

    parser_18 = sub_parsers.add_parser('analysis_5_part_2')
    parser_18.add_argument('original_results')
    parser_18.add_argument('included_results')

    parser_20 = sub_parsers.add_parser('extract_similar_phrases')
    parser_20.add_argument('--num_alternative_phrases', type=int, default=5)

    parser_21 = sub_parsers.add_parser('get_phrases_by')
    parser_21.add_argument('count_function', choices=count_functions)
    parser_21.add_argument('--max_length', type=int, default=10)
    parser_21.add_argument('--phrases_per_length', type=int, default=20)

    parser_22 = sub_parsers.add_parser('experiment_8')
    parser_22.add_argument('videos_directory')
    parser_22.add_argument('user_id')
    parser_22.add_argument('--phrase_list', type=file_list, default=None)
    parser_22.add_argument('--synthetic', type=bool, default=False)

    parser_23 = sub_parsers.add_parser('analysis_8')
    parser_23.add_argument('results_file')

    parser_24 = sub_parsers.add_parser('experiment_9')
    parser_24.add_argument('videos_directory')
    parser_24.add_argument('user_id')
    parser_24.add_argument('--phrase_list', type=file_list, default=None)
    parser_24.add_argument('--synthetic', type=bool, default=False)
    parser_24.add_argument('--count_function', choices=count_functions, default='words')

    parser_25 = sub_parsers.add_parser('experiment_9_by_count')
    parser_25.add_argument('videos_directory')
    parser_25.add_argument('user_id')
    parser_25.add_argument('max_length', type=int)
    parser_25.add_argument('phrases_per_length', type=int)
    parser_25.add_argument('--count_function', choices=count_functions,
                           default='words')
    parser_25.add_argument('--max_num_vectors', type=int, default=None)
    parser_25.add_argument('--start_from_count', type=int, default=1)
    parser_25.add_argument('--synthetic', action='store_true')
    parser_25.add_argument('--similarity_threshold', type=int)
    parser_25.add_argument('--phrase_list', type=file_list, default=None)

    parser_26 = sub_parsers.add_parser('analysis_9')
    parser_26.add_argument('results_file')

    parser_27 = sub_parsers.add_parser('analysis_1_b')
    parser_27.add_argument('results_file')
    parser_27.add_argument('title')

    parser_28 = sub_parsers.add_parser('analysis_5_part_3')
    parser_28.add_argument('original_results_file')
    parser_28.add_argument('included_results_file')

    main(parser.parse_args())
