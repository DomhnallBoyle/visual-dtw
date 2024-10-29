"""taking inspiration from https://github.com/steinbro/hyperviseme"""
import argparse
import re
from itertools import chain

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


def make_cmudict_viseme_map():
    d = {}  # str: str dict
    with open('cmudict-en-us.dict', 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            word = line[0]

            # skip words with alternatives
            if '(' in word:
                continue

            phones = list(map(lambda phone: phone.lower().strip(), line[1:]))
            visemes = list(map(lambda phone: PHONEME_TO_VISEME[phone], phones))

            d[word] = visemes

    return d


def make_inverse_cmudict_viseme_map(d):
    new_d = {}  # str: str dict
    for word, visemes in d.items():
        visemes_key = tuple(visemes)
        similar_words = new_d.get(visemes_key, [])
        similar_words.append(word)
        new_d[visemes_key] = similar_words

    return new_d


def find_shortest_prefix(visemes, visemes_2_words_map, min_length=0):
    buffer = tuple(visemes[0:min_length+1])
    while buffer not in visemes_2_words_map:
        if len(buffer) < len(visemes):
            buffer = tuple(visemes[0:len(buffer)+1])
        else:
            return None

    return len(buffer)


def find_possible_chunks(phrase_visemes, visemes_2_words_map, current=[]):
    successes = []
    n = 0
    while n < len(phrase_visemes):
        n = find_shortest_prefix(phrase_visemes, visemes_2_words_map, n)
        if not n:
            break

        if n == len(phrase_visemes):
            return successes + [current + [phrase_visemes[0:n]]]

        successes += find_possible_chunks(
            phrase_visemes[n:len(phrase_visemes)],
            visemes_2_words_map,
            current + [phrase_visemes[0:n]]
        )

    return successes


def find_similar_phrases(input_phrase, num_phrases=5):
    word_2_visemes_map = make_cmudict_viseme_map()
    visemes_2_words_map = make_inverse_cmudict_viseme_map(word_2_visemes_map)

    # input phrase validation
    input_phrase = input_phrase.lower().strip()
    input_phrase = re.sub(r'[\?\!]', '', input_phrase)

    try:
        phrase_visemes = [
            word_2_visemes_map[word]
            for word in input_phrase.split(' ')
        ]
        phrase_visemes = list(chain.from_iterable(phrase_visemes))  # flatten
    except KeyError as e:
        print(f'Word {e} does not exist')
        return []

    chunks = find_possible_chunks(phrase_visemes, visemes_2_words_map)[::-1]

    alternatives = [
        visemes_2_words_map[tuple(visemes)] for visemes in chunks[0]
    ]

    similar_phrases = []
    num_words = len(chunks[0])
    least_num_alts = min([len(alts) for alts in alternatives])

    if least_num_alts < num_phrases:
        num_phrases = least_num_alts

    for j in range(num_phrases):
        similar_phrase = ' '.join([
            alternatives[i][j]
            for i in range(num_words)
        ])
        similar_phrases.append(similar_phrase)

    return [p for p in similar_phrases if p != input_phrase]


def main(args):
    similar_phrases = find_similar_phrases(args.phrase)
    print(args.phrase, similar_phrases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('phrase')

    main(parser.parse_args())
