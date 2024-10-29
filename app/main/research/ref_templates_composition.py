import argparse
import ast
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config, SRAVITemplate
from main.services.transcribe import transcribe_signal
from main.utils.io import read_pickle_file
from main.utils.db import find_phrase_mappings, invert_phrase_mappings, \
    setup as setup_db
from main.utils.pre_process import pre_process_signals
from tqdm import tqdm

TEMPLATE_REGEX_FROM_DEFAULT_LIST = r'AE_norm_2_(\d+)_(S\w?[A-Z])(\d+)_(\d+)'
BEST_TEMPLATES_REGEX = r'\[2, 3, 7\],(\d+),(\d+.\d+),(\[.+\])'


def analyse(**kwargs):
    template_counts = {}
    default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)

    for phrase in default_lst.phrases:
        if phrase.content != 'None of the above':
            template_counts[phrase.content] = len(phrase.templates)

    # show template counts in graph
    plt.bar(template_counts.keys(), template_counts.values())
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.title('Template counts per phrase for the default list')
    plt.ylabel('Template count')
    plt.show()

    # show distribution per user
    phrase_mappings = find_phrase_mappings('PAVA-DEFAULT')
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)
    default_templates = np.genfromtxt(configuration.DEFAULT_TEMPLATES_PATH,
                                      delimiter=',', dtype='str')

    user_templates = {}
    for template in default_templates:
        user, phrase_set, phrase_id, session = \
            re.match(TEMPLATE_REGEX_FROM_DEFAULT_LIST, template).groups()

        templates = user_templates.get(user, [])
        templates.append(template)
        user_templates[user] = templates

    print('Templates per user:')
    for user, templates in user_templates.items():
        print(user, len(templates))

    print('\nTotal num templates: ',
          sum(len(templates) for templates in user_templates.values()))

    print('Sessions per user: ')
    user_sessions = {}
    for user, templates in user_templates.items():
        for template in templates:
            user, phrase_set, phrase_id, session = \
                re.match(TEMPLATE_REGEX_FROM_DEFAULT_LIST, template).groups()

            sessions = user_sessions.get(user, {})
            session_templates = sessions.get(session, {})
            session_phrase_set_templates = session_templates.get(phrase_set, [])

            session_phrase_set_templates.append(template)
            session_templates[phrase_set] = session_phrase_set_templates

            sessions[session] = session_templates
            user_sessions[user] = sessions

    print(user_sessions)

    total_count = 0
    for user, sessions in user_sessions.items():
        print(user)
        for session, phrase_templates in sessions.items():
            print(session)
            for phrase_set, templates in phrase_templates.items():
                print(phrase_set, len(templates))
                total_count += len(templates)
        print()

    print(total_count)


def get_templates(phrase_mappings, users):
    templates = []
    for phrases in phrase_mappings.values():
        templates.extend(SRAVITemplate.get(
            filter=(
                (SRAVITemplate.user_id.in_(users)) &
                (SRAVITemplate.phrase_id.in_(phrases)) &
                (SRAVITemplate.feature_type == 'AE_norm_2')
            )
        ))

    return templates


def get_training_templates(num_sessions, phrase_mappings, training_users):
    training_templates = []
    num_phrases = len(phrase_mappings.keys())

    stop = False
    attempts = 0
    max_attempts = 10
    user_sessions = {}

    while not stop and attempts != max_attempts:
        user_sessions = {}
        num_chosen_sessions = 0

        for user in training_users:
            user_sessions[user] = []

        while True:
            for user in training_users:
                already_chosen_templates = []
                session = []

                for phrase, similar_phrases in phrase_mappings.items():
                    # select random phrase
                    while True:
                        random_phrase = random.choice(similar_phrases)

                        # get all templates at particular phrase and user
                        phrase_templates = SRAVITemplate.get(
                            filter=(
                                (SRAVITemplate.user_id == user) &
                                (SRAVITemplate.phrase_id == random_phrase) &
                                (SRAVITemplate.feature_type == 'AE_norm_2')
                            )
                        )

                        if len(phrase_templates) > 0:
                            break

                    # pick random template
                    while True:
                        random_template = random.choice(phrase_templates)

                        # check it hasn't been chosen already
                        if random_template.key not in already_chosen_templates:
                            session.append(random_template)
                            already_chosen_templates.append(random_template.key)
                            break

                if len(session) == num_phrases:
                    user_sessions[user].append(session)
                    num_chosen_sessions += 1

                if num_chosen_sessions == num_sessions:
                    stop = True
                    break

            if stop:
                break

        template_count = 0
        for user, sessions in user_sessions.items():
            for session in sessions:
                if not check_unique_templates(session): 
                    stop = False
                template_count += len(session)

        print('Template count: ', template_count)
        attempts += 1

    for user, sessions in user_sessions.items():
        print(user)
        for session in sessions:
            print(len(session))
            training_templates.extend(session)
        print()

    return training_templates


def check_unique_templates(templates):
    # check for no duplicates
    unique_templates = set()
    for template in templates:
        unique_templates.add(template)

    if len(unique_templates) == len(templates): 
        return True

    return False


def get_training_templates_2(num_sessions, phrase_mappings, training_users): 
    training_templates = []    
    num_phrases = len(phrase_mappings.keys())
    num_chosen_sessions = 0

    user_templates = {}
    user_templates_init = {}
    for user_id in training_users: 
        user_templates[user_id] = {}
        user_templates_init[user_id] = {}
        for phrase in phrase_mappings.keys():
            user_templates[user_id][phrase] = [] 
            user_templates_init[user_id][phrase] = False

    stop = False

    while True: 
        for user_id in training_users:
            # constructing a session 
            for phrase, similar_phrases in phrase_mappings.items():
 
                # get templates for particular phrase
                if not user_templates_init[user_id][phrase]:
                    templates = SRAVITemplate.get(
                        filter=(
                            (SRAVITemplate.user_id == user_id) & 
                            (SRAVITemplate.phrase_id.in_(similar_phrases))
                        )
                    )
                    user_templates[user_id][phrase] = templates
                    user_templates_init[user_id][phrase] = True                

                # check if there are any templates left to use
                if len(user_templates[user_id][phrase]) > 0:
                    template_to_use = user_templates[user_id][phrase].pop(0)
                    training_templates.append(template_to_use)
                else:
                    print('Unable to complete session: ', user_id, phrase)
                    exit(0)

            # session completed
            num_chosen_sessions += 1

            if num_chosen_sessions == num_sessions: 
                stop = True
                break

        if stop: 
            break

    assert len(training_templates) == num_sessions * num_phrases

    return training_templates


def full_session_templates(**kwargs):
    """Finding best full session templates for best users 2, 3 and 7 against
    other patient users
    """
    setup_db()

    training_users = [2, 3, 7]
    patient_users = [201, 202, 203, 204, 207, 208, 210, 212, 213, 214, 215]

    phrase_mappings = find_phrase_mappings('PAVA-DEFAULT')
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)
    dtw_params = Config().__dict__
    dtw_params['threshold'] = None

    testing = get_templates(phrase_mappings, patient_users)
    training = get_templates(phrase_mappings, training_users)

    max_num_sessions = len(training) // len(phrase_mappings.keys())
    start_from_session = kwargs['start_from_session'] \
        if kwargs['start_from_session'] else len(training_users)

    print('Total number of reference templates: ', len(training))
    print('Max number of sessions: ', max_num_sessions)

    max_num_attempts = 10

    # num sessions from 3 to max number of sessions
    for i in range(start_from_session, max_num_sessions + 1):
        num_sessions = i
        num_attempts_since_last_improvement = 0
        best_accuracy = 0

        while num_attempts_since_last_improvement != max_num_attempts:
            training = get_training_templates_2(num_sessions,
                                                phrase_mappings,
                                                training_users)

            if not check_unique_templates(training): 
                num_attempts_since_last_improvement += 1
                print('Non unique for num sessions: ', num_sessions)
                time.sleep(5)
                continue

            print('Num sessions: ', num_sessions)
            print('Number of training templates: ', len(training))
            print('Number of testing templates: ', len(testing))

            # pre-process reference signals
            ref_signals = pre_process_signals(
                signals=[template.blob for template in training],
                **dtw_params
            )
            ref_signals = [(template.phrase_id, ref_signal)
                           for template, ref_signal in
                           zip(training, ref_signals)]

            top_3_accuracy = 0

            for test_template in tqdm(testing):
                # first pre-process test template
                test_signal = pre_process_signals(
                    signals=[test_template.blob], **dtw_params
                )[0]

                # get predictions by transcribing template signal
                predictions = transcribe_signal(ref_signals=ref_signals,
                                                test_signal=test_signal,
                                                **dtw_params)

                pava_prediction_keys = \
                    [inverse_phrase_mappings[k['label']] for k in predictions]

                pava_phrase = inverse_phrase_mappings[test_template.phrase_id]
                if pava_phrase in pava_prediction_keys:
                    top_3_accuracy += 1

            top_3_accuracy = (top_3_accuracy / len(testing)) * 100

            # test users are the same for every test
            with open('best_templates.csv', 'a') as f:
                training_keys = [t.key for t in training]
                line = f'{training_users},{num_sessions},{top_3_accuracy},' \
                       f'{training_keys}\n'
                f.write(line)

            if top_3_accuracy > best_accuracy:
                best_accuracy = top_3_accuracy
                num_attempts_since_last_improvement = 0
            else:
                num_attempts_since_last_improvement += 1


def analyse_2(**kwargs):
    data = []

    file_path = kwargs['file_path']
    if not file_path: 
        file_path = 'best_templates.csv'

    with open(file_path, 'r') as f:
        for line in f.readlines(): 
            num_sessions, accuracy, templates = \
                re.match(BEST_TEMPLATES_REGEX, line).groups()
            templates = ast.literal_eval(templates)
            data.append([int(num_sessions), float(accuracy), templates])

    df = pd.DataFrame(data, columns=['Num Sessions', 'Accuracy', 'Templates'])

    # get the row with the max accuracy
    max_accuracy_row = df.loc[df['Accuracy'].idxmax()]
    print('Max row accuracy:')
    print('Accuracy: ', max_accuracy_row['Accuracy'])
    print('Num Sessions: ', max_accuracy_row['Num Sessions'])
    print('Num templates: ', len(max_accuracy_row['Templates']))

    if check_unique_templates(max_accuracy_row['Templates']): 
        print('Max row has unique templates')
    else:
        print('Max row has non unique templates')

    # plot mean accuracies by num sessions
    df.groupby(['Num Sessions'])['Accuracy'].mean().plot(legend=True)
    plt.show()

    # make sure all rows have unique templates
    best_accuracy, best_index = 0, 0
    for index, row in df.iterrows():
        assert check_unique_templates(row['Templates'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser1 = sub_parsers.add_parser('analyse')

    parser2 = sub_parsers.add_parser('full_session_templates')
    parser2.add_argument('--start_from_session', type=int, default=None)

    parser3 = sub_parsers.add_parser('analyse_2')
    parser3.add_argument('--file_path', type=str, default=None)

    f = {
        'analyse': analyse,
        'analyse_2': analyse_2,
        'full_session_templates': full_session_templates
    }

    args = parser.parse_args()
    if args.run_type in list(f.keys()):
        f[args.run_type](**args.__dict__)
    else:
        parser.print_help()
