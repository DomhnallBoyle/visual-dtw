import io
import os
import random
import re

import requests
from main import configuration
from main.models import PAVAList, PAVAUser
from main.utils.io import read_json_file
from main.research.cmc import CMC

NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
TRANSCRIBE_ENDPOINT = \
    'http://0.0.0.0:5000/pava/api/v1/lists/{}/transcribe/video'

user = PAVAUser.create(default_list=True)
list_id = PAVAList.get(filter=(PAVAList.user_id == user.id), first=True).id


def get_templates(videos_path):
    templates = {}
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    for video in os.listdir(videos_path):
        if not video.endswith('.mp4'):
            continue

        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        pava_phrase = pava_phrases[phrase_id]
        phrase_templates = templates.get(pava_phrase, [])
        phrase_templates.append((pava_phrase,
                                 os.path.join(videos_path, video)))
        templates[pava_phrase] = phrase_templates

    return templates


def split(videos_path, chosen_phrases, n=20):
    user_templates = get_templates(videos_path)

    chosen_templates, other_templates = [], []
    for k, v in user_templates.items():
        if k in chosen_phrases:
            chosen_templates.extend(list(v))
        else:
            other_templates.extend(list(v))
    chosen_templates = random.sample(chosen_templates, n)

    random.shuffle(chosen_templates)
    random.shuffle(other_templates)

    return chosen_templates, other_templates


def get_accuracies(templates, phrase_probs):
    cmc = CMC(num_ranks=3)

    for actual_label, template in templates:
        with open(template, 'rb') as f:
            files = {'file': (template, io.BytesIO(f.read()))}
            response = requests.post(TRANSCRIBE_ENDPOINT.format(list_id),
                                     files=files)

            if response.status_code == 200:
                response = response.json()['response']
                if response != {}:
                    predictions = response['predictions']
                    del predictions[-1]

                    for p in predictions:
                        del p['id']
                        label = p['label']
                        if phrase_probs.get(label, None):
                            accuracy = p['accuracy']
                            new_accuracy = \
                                ((1 - accuracy) * phrase_probs[label]) + accuracy
                            p['accuracy'] = new_accuracy

                    accuracy_sum = sum([p['accuracy'] for p in predictions])
                    for p in predictions:
                        p['accuracy'] /= accuracy_sum
                        p['accuracy'] = round(p['accuracy'], 2)

                    predictions = sorted(predictions,
                                         key=lambda p: p['accuracy'],
                                         reverse=True)

                    prediction_labels = [p['label'] for p in predictions]
                    cmc.tally(prediction_labels, actual_label)

    cmc.calculate_accuracies(count_check=False)

    return cmc.all_rank_accuracies[0]


def main():
    # pick 5 worst phrases

    # randomly choose templates to make predictions on from phrases,
    # repeat this 20 times
    # find their true label and associate probability out of 20

    # each time template predicted of the 20, compare predictions with the
    # probability
    # e.g. Doctor = 3/5 = 0.6%, if Doctor comes up again in predictions,
    # 1: 0.8. 1 - 0.8 = 0.2. 0.2 * 0.6 = 0.12. 0.8 + 0.12 = 0.92
    # 2: 0.2. 1 - 0.2 = 0.8. 0.8 * 0.6 = 0.48. 0.2 + 0.48 = 0.68
    # 3: 0.45. 1 - 0.45 = 0.55. 0.55 * 0.6 = 0.33. 0.45 + 0.33 = 0.78

    # renormalise the new predictions to sum 1

    # get rank 3 accuracy each time

    videos_path = '/home/domhnall/Documents/sravi_dataset/liopa/9'
    worst_phrases = ['Thank you', 'I am scared', "I'm hungry", 'Doctor',
                     'Move me']

    chosen_templates, test_templates = split(videos_path, worst_phrases)

    def get_phrase_probs(templates):
        phrase_probs = {}
        num = len(templates)

        for label, template in templates:
            phrase_probs[label] = phrase_probs.get(label, 0) + 1

        return {k: v/num for k, v in phrase_probs.items()}

    accuracies = get_accuracies(test_templates, {})
    print(accuracies)

    templates_so_far = []
    for label, template in chosen_templates:
        templates_so_far.append((label, template))

    phrase_probs = get_phrase_probs(templates_so_far)
    accuracies = get_accuracies(test_templates, phrase_probs)
    print(accuracies)


if __name__ == '__main__':
    main()
