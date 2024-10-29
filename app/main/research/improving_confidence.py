import os
import re
from difflib import SequenceMatcher

import jellyfish
import matplotlib.pyplot as plt
from main import configuration
from main.research.phrase_list_composition import WORDS_TO_VISEMES
from main.utils.io import read_json_file

CURRENT_DIRECTORY = os.path.dirname(__file__)
DEFAULT_PHRASES = list(
    read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT'].values()
)
NON_LIST_PHRASES = list(set(open(os.path.join(CURRENT_DIRECTORY, 'phrases.txt')).read().splitlines()) - set(DEFAULT_PHRASES))

"""
Algorithm 1 (checking if phrase in list): 
- Get visemes from video (in this case Adrian transcripts)
- Get visemes from phrase list
- Get distances of video visemes to phrase list visemes
- Get highest score
- If highest score < threshold: uttered phrase not in list

Algorithm 2: 
- If algorithm 1 succeeds, we have a possible match to a phrase in the phrase list
- Run DTW/KNN predictions as normal, get predictions
- Compare match with predictions e.g. 
    - Is match in predictions?
    - Is match and R1 the same? etc...
"""


def extract_visemes(phrase):
    visemes = ''
    for word in phrase.split(' '):
        word = word.lower().strip()  # lowercase and strip word
        word = re.sub(r'[\?\!]', '', word)  # replace any characters

        word_visemes = WORDS_TO_VISEMES.get(word, '')
        visemes += word_visemes

    return visemes


def get_phrase_list_distances(phrase, phrases, distance_metric):
    phrase_1_visemes = extract_visemes(phrase)

    phrase_list_distances = {}
    for phrase in phrases:
        phrase_2_visemes = extract_visemes(phrase)
        # print(phrase_1_visemes, phrase_2_visemes)
        distance = distance_metric(phrase_1_visemes, phrase_2_visemes)
        phrases_at_distance = phrase_list_distances.get(distance, [])
        phrases_at_distance.append(phrase)
        phrase_list_distances[distance] = phrases_at_distance

    return {
        k: phrase_list_distances[k] for k in sorted(phrase_list_distances)
    }


def levinshtein(s1, s2):
    return jellyfish.levenshtein_distance(s1, s2)


def sequence_matcher(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()


def main():
    distance_metrics = {
        'levinshtein': levinshtein,
        'sequence_matcher': sequence_matcher
    }

    best_scores = []
    for phrase in NON_LIST_PHRASES:
        list_distances = get_phrase_list_distances(phrase,
                                                   DEFAULT_PHRASES,
                                                   distance_metrics['sequence_matcher'])
        best_score = max(list_distances.keys())
        if best_score > 0.8:
            print(phrase, list_distances[best_score], best_score)
        best_scores.append(best_score)

    # show counts histogram
    plt.hist(best_scores, density=False, bins=10)
    plt.axis([0, 1.0, 0, 50])
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.show()


if __name__ == '__main__':
    main()
