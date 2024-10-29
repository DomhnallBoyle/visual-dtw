"""
Experiments to find the best templates among the already liopa dataset

Testing different permutations of liopa users vs patient speakers to extract
the best templates
"""
import argparse
import ast
import random
import re
from itertools import combinations, permutations

import matplotlib.pyplot as plt
import pandas as pd
from main import configuration
from main.models import SRAVITemplate
from main.research.utils import generate_dtw_params
from main.services.transcribe import transcribe_signal
from main.utils.db import find_phrase_mappings, invert_phrase_mappings,\
    setup as setup_db
from main.utils.pre_process import pre_process_signals
from main.utils.io import read_json_file
from tqdm import tqdm

PHRASE_SET = 'PAVA'
LIOPA_USERS = [1, 2, 3, 4, 5, 6, 7]
PATIENT_USERS = [201, 202, 203, 204, 207, 208, 210, 212, 213, 214, 215]
PHRASE_REGEX = r'(S\w?[A-Z])(\d\d*)'
BY_PHRASE_REGEX = r'(\[.+\]),(\d+),(\{.+\}),(.+)'
BY_PHRASE_2_REGEX = r'(\[.+\]),(\d+.\d+),(\d+.\d+),(\{.+\})'
TEMPLATE_REGEX = r'AE_norm_2_(\d+)_(S\w?[A-Z]\d\d*)_(\d+)'
USER_MAPPINGS = {
    str(_id): name
    for _id, name in zip(LIOPA_USERS,
                         ['Fabian', 'Alex', 'Adrian', 'Yogi', 'Liam',
                          'Richard', 'Conor'])
}


def get_templates(phrase_mappings, users):
    setup_db()

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


def group_similar_phrases(phrase_mappings, templates):
    # group similar phrases together by phrase id
    phrase_sets = {
        PHRASE_SET + k: []
        for k in read_json_file(configuration.PHRASES_PATH)[PHRASE_SET].keys()
    }
    for phrase_set, related_phrases in phrase_mappings.items():
        for template in templates:
            if template.phrase_id in related_phrases:
                phrase_sets[phrase_set].append(template)

    return phrase_sets


def all_vs_all(**kwargs):
    """All liopa users vs all patient users.

    Args:
        **kwargs:

    Returns:

    """
    phrase_mappings = find_phrase_mappings(PHRASE_SET)
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)

    training = get_templates(phrase_mappings, LIOPA_USERS)
    testing = get_templates(phrase_mappings, PATIENT_USERS)

    print('Num liopa templates: ', len(training))
    print('Num patient templates: ', len(testing))

    if len(training) == 0 or len(testing) == 0:
        exit()

    phrase_sets = group_similar_phrases(phrase_mappings, training)
    dtw_params = generate_dtw_params()

    phase = kwargs['from_phase']
    best_top_3_accuracy = kwargs['best_top_3_accuracy']
    num_samples = kwargs['num_samples']
    num_phases_since_last_improvement = 0
    num_attempts_to_improve = kwargs['attempts_to_improve']

    while True:
        print(f'\nPhase: {phase}')
        print('Num samples: ', num_samples)

        # select random n from each group of phrases (ref templates)
        ref_templates = []
        for phrase in phrase_sets.keys():
            random.shuffle(phrase_sets[phrase])
            try:
                templates = random.sample(phrase_sets[phrase], num_samples)
            except ValueError:
                # sample size too large
                templates = phrase_sets[phrase]
            ref_templates.extend(templates)

        test_templates = testing
        print('Training, Testing: ', len(ref_templates), len(test_templates))

        # pre-process reference signals
        ref_signals = pre_process_signals(
            signals=[template.blob for template in ref_templates],
            **dtw_params
        )
        ref_signals = [(template.phrase_id, ref_signal)
                       for template, ref_signal in
                       zip(ref_templates, ref_signals)]

        top_3_accuracy = 0

        for test_template in tqdm(test_templates):
            # first pre-process test template
            test_signal = pre_process_signals(
                signals=[test_template.blob], **dtw_params
            )[0]

            # get predictions by transcribing template signal
            predictions = transcribe_signal(ref_signals=ref_signals,
                                            test_signal=test_signal,
                                            **dtw_params)

            prediction_keys = [list(prediction.keys())[0]
                               for prediction in predictions]
            prediction_keys = [inverse_phrase_mappings.get(key, key)
                               for key in prediction_keys]

            if inverse_phrase_mappings[test_template.phrase_id] in prediction_keys \
                    or test_template.phrase_id in prediction_keys:
                top_3_accuracy += 1

        top_3_accuracy /= len(test_templates)
        top_3_accuracy *= 100

        if top_3_accuracy > best_top_3_accuracy:
            best_top_3_accuracy = top_3_accuracy
            print('New best: ', best_top_3_accuracy)
            num_phases_since_last_improvement = 0
        else:
            # increment sample size every 20 phases
            # if no improvement in accuracy
            if num_phases_since_last_improvement == num_attempts_to_improve:
                num_samples += 5
                num_phases_since_last_improvement = 0
            else:
                num_phases_since_last_improvement += 1

        with open(f'{PHRASE_SET}_{len(training)}_{len(testing)}_all_vs_all.csv', 'a') as f:
            line = ','.join([str(result) for result in [
                phase,
                num_samples,
                len(ref_templates),
                top_3_accuracy,
                [t.key for t in ref_templates]
            ]])
            f.write(line + '\n')

        phase += 1


def all_vs_all_2(**kwargs):
    """Pre-defined users against ALL patient users.

    Templates are sampled fairly at random i.e. 5 templates per user per phrase
    Num samples incremented each time

    Args:
        **kwargs:

    Returns:

    """
    users = kwargs['users']
    print('Users', users)

    attempts_to_improve = 10
    phrase_mappings = find_phrase_mappings(PHRASE_SET)
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)

    training_templates = get_templates(phrase_mappings, users)
    test_templates = get_templates(phrase_mappings, PATIENT_USERS)

    duplicate_users = kwargs['duplicate_users']
    if duplicate_users:
        for user in duplicate_users:
            templates = get_templates(phrase_mappings, [user])
            training_templates.extend(templates)

    phrase_sets = group_similar_phrases(phrase_mappings, training_templates)
    num_samples = kwargs['num_samples']
    dtw_params = generate_dtw_params()

    best_top_3_accuracy = kwargs['best_top_3_accuracy']
    attempts_since_last_improvement = 0

    while True:
        # select random n from each group of phrases (ref templates)
        ref_templates = []
        for i in range(num_samples):
            for phrase in phrase_sets.keys():
                random.shuffle(phrase_sets[phrase])
                for user_id in users:
                    for template in phrase_sets[phrase]:
                        if template.user_id == int(user_id) and \
                                template not in set(ref_templates):
                            ref_templates.append(template)
                            break

        print('Training, Testing: ', len(ref_templates), len(test_templates))

        # pre-process reference signals
        ref_signals = pre_process_signals(
            signals=[template.blob for template in ref_templates],
            **dtw_params
        )
        ref_signals = [(template.phrase_id, ref_signal)
                       for template, ref_signal in
                       zip(ref_templates, ref_signals)]

        top_3_accuracy = 0

        for test_template in tqdm(test_templates):
            # first pre-process test template
            test_signal = pre_process_signals(
                signals=[test_template.blob],
                **dtw_params
            )[0]

            # get predictions by transcribing template signal
            predictions = transcribe_signal(ref_signals=ref_signals,
                                            test_signal=test_signal,
                                            **dtw_params)

            pava_prediction_keys = \
                [inverse_phrase_mappings[k['label']] for k in predictions]

            if inverse_phrase_mappings[test_template.phrase_id] \
                    in pava_prediction_keys:
                top_3_accuracy += 1

        top_3_accuracy = (top_3_accuracy / len(test_templates)) * 100
        print(top_3_accuracy)

        if top_3_accuracy > best_top_3_accuracy:
            best_top_3_accuracy = top_3_accuracy
            print('New best', best_top_3_accuracy)
            attempts_since_last_improvement = 0
        else:
            if attempts_since_last_improvement == attempts_to_improve:
                num_samples += 1
                attempts_since_last_improvement = 0
            else:
                attempts_since_last_improvement += 1
        print()

        output_file = kwargs['output_file']
        with open(output_file, 'a') as f:
            line = f'{num_samples},{len(ref_templates)},{top_3_accuracy},'
            line += str([t.key for t in ref_templates])
            f.write(line + '\n')


def by_phrases(**kwargs):
    """Every combination of liopa users vs single patient users

    Args:
        **kwargs:

    Returns:

    """
    combo_index = kwargs['combo_index']
    phrase_mappings = find_phrase_mappings(PHRASE_SET)
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)
    dtw_params = generate_dtw_params()

    for i in range(combo_index, len(LIOPA_USERS) + 1):
        for training_users in list(combinations(LIOPA_USERS, i)):
            training_users = list(training_users)
            training = get_templates(phrase_mappings, training_users)

            for test_user in PATIENT_USERS:
                phrase_accuracies = {}
                testing = get_templates(phrase_mappings, [test_user])

                print(f'Liopa templates: {training_users}, {len(training)}')
                print(f'Patient templates: {test_user}, {len(testing)}')

                # pre-process reference signals
                ref_signals = pre_process_signals(
                    signals=[template.blob for template in training],
                    **dtw_params
                )
                ref_signals = [(template.phrase_id, ref_signal)
                               for template, ref_signal in
                               zip(training, ref_signals)]

                for test_template in tqdm(testing):
                    # first pre-process test template
                    test_signal = pre_process_signals(
                        signals=[test_template.blob], **dtw_params
                    )[0]

                    # get predictions by transcribing template signal
                    predictions = transcribe_signal(ref_signals=ref_signals,
                                                    test_signal=test_signal,
                                                    **dtw_params)

                    prediction_keys = [list(prediction.keys())[0]
                                       for prediction in predictions]
                    prediction_keys = [inverse_phrase_mappings.get(key, key)
                                       for key in prediction_keys]
                    if inverse_phrase_mappings[
                        test_template.phrase_id] in prediction_keys \
                            or test_template.phrase_id in prediction_keys:
                        accuracy = phrase_accuracies.get(test_template.phrase_id, [0, 0])
                        accuracy = [accuracy[0] + 1, accuracy[1] + 1]
                        phrase_accuracies[test_template.phrase_id] = accuracy
                    else:
                        accuracy = phrase_accuracies.get(test_template.phrase_id, [0, 0])
                        accuracy = [accuracy[0], accuracy[1] + 1]
                        phrase_accuracies[test_template.phrase_id] = accuracy

                for k, value in phrase_accuracies.items():
                    f = value[0] / value[1]
                    phrase_accuracies[k] = f

                with open('PAVA_by_phrase.csv', 'a') as f:
                    average_phrase_accuracy = sum(phrase_accuracies.values()) \
                                              / len(phrase_accuracies.keys())
                    f.write(f'{training_users},{test_user},{phrase_accuracies},{average_phrase_accuracy}\n')


def by_phrases_2(**kwargs):
    """Every combination of LIOPA users vs ALL patient users.

    Getting phrase accuracies

    Args:
        **kwargs:

    Returns:

    """
    phrase_mappings = find_phrase_mappings(PHRASE_SET)
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)
    dtw_params = generate_dtw_params()

    for i in range(1, len(LIOPA_USERS) + 1):
        for training_users in list(combinations(LIOPA_USERS, i)):
            training_users = list(training_users)

            training = get_templates(phrase_mappings, training_users)
            testing = get_templates(phrase_mappings, PATIENT_USERS)

            # pre-process reference signals
            ref_signals = pre_process_signals(
                signals=[template.blob for template in training],
                **dtw_params
            )
            ref_signals = [(template.phrase_id, ref_signal)
                           for template, ref_signal in
                           zip(training, ref_signals)]

            top_3_accuracy, phrase_accuracies = 0, {}

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
                    p_accuracy = phrase_accuracies.get(pava_phrase, [0, 0])
                    p_accuracy = [p_accuracy[0] + 1, p_accuracy[1] + 1]
                    phrase_accuracies[pava_phrase] = p_accuracy
                else:
                    p_accuracy = phrase_accuracies.get(pava_phrase, [0, 0])
                    p_accuracy = [p_accuracy[0], p_accuracy[1] + 1]
                    phrase_accuracies[pava_phrase] = p_accuracy

            top_3_accuracy = (top_3_accuracy / len(testing)) * 100
            for k, value in phrase_accuracies.items():
                phrase_accuracies[k] = (value[0] / value[1]) * 100

            # test users are the same for every test
            with open('PAVA_by_phrase_2.csv', 'a') as f:
                average_phrase_accuracy = sum(phrase_accuracies.values()) \
                                          / len(phrase_accuracies.keys())
                line = f'{training_users},{top_3_accuracy},' \
                       f'{average_phrase_accuracy},{phrase_accuracies}'
                f.write(line + '\n')


def bar_graph(x, y, title, xlabel, ylabel, rotation=0, colours=None):
    plt.bar(x, y, color=colours)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.xticks(rotation=rotation)
    plt.axhline(0, color='black')
    plt.tight_layout()
    plt.show()


def extract_counts(rows):
    phrase_mappings = find_phrase_mappings(PHRASE_SET)
    inverse_phrase_mappings = invert_phrase_mappings(phrase_mappings)

    user_count = {}
    phrase_set_count = {
        k1: {p: 0 for p in configuration.SRAVI_PHRASE_SETS}
        for k1 in phrase_mappings.keys()
    }

    def process(row):
        templates = row['Templates']
        for template in templates:
            template_id = template.replace('AE_norm_2_', '').split('_')
            user_id, phrase, session = template_id

            user_count[USER_MAPPINGS[user_id]] = \
                user_count.get(USER_MAPPINGS[user_id], 0) + 1

            pava_phrase = inverse_phrase_mappings[phrase]
            sravi_phrase = re.match(PHRASE_REGEX, phrase).groups()[0]
            phrase_set_count[pava_phrase][sravi_phrase] += 1

    if len(rows) == 1:
        process(rows[0])
    else:
        for index, row in rows.iterrows():
            process(row)

    bar_graph(user_count.keys(), user_count.values(),
              'No. of times users appear in the templates', 'Users', 'Count')

    # plot stacked column bar graph of phrase frequencies in templates
    data = {}
    for sravi_phrase in configuration.SRAVI_PHRASE_SETS:
        data[sravi_phrase] = []
        for pava_phrase in phrase_set_count.keys():
            data[sravi_phrase]\
                .append(phrase_set_count[pava_phrase][sravi_phrase])

    print('Total Counts')
    for sravi_phrase, pava_counts in data.items():
        print(sravi_phrase, sum(pava_counts))

    df = pd.DataFrame(data, index=list(phrase_mappings.keys()))
    df.plot(title='Phrase frequency in templates', kind='bar', stacked=True,
            rot=70)
    plt.xlabel('Phrases')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()


def analyse(**kwargs):

    def analyse_all_vs_all():
        with open('PAVA_1484_263_all_vs_all.csv') as f:
            df = pd.DataFrame(f.readlines(), columns=['A'])

        columns = ['Phase', 'Num Samples Per Phrase', 'Num Ref', 'Accuracy',
                   'Templates']
        df[columns] = df.A.str.split(',', n=4, expand=True)
        df['Templates'] = df['Templates'].str.split(',')
        df = df.drop(['A'], axis=1)

        # convert columns to correct dtypes
        for c in columns[:-1]:
            df[c] = pd.to_numeric(df[c])

        # max and min rows
        max_row = df.loc[df['Accuracy'].idxmax()]
        print(max_row)
        min_row = df.loc[df['Accuracy'].idxmin()]
        print(min_row)

        # group rows by num references and average accuracy
        # plot graph
        ref_accuracies = df.groupby('Num Ref').mean()[['Accuracy']]
        ref_accuracies = ref_accuracies.sort_values('Num Ref')
        ref_accuracies.plot(
            title='Average Accuracy vs. No. of Reference Templates', grid=True
        )
        plt.xlabel('No. Reference Templates')
        plt.ylabel('Accuracy (%)')
        plt.show()

        # best average accuracy is above 575
        best = df.loc[df['Num Ref'] >= 575]
        extract_counts(best)

        worst = df.loc[df['Num Ref'] < 575]
        extract_counts(worst)

    def analyse_by_phrases():
        columns = ['Ref Users', 'Test User', 'Phrase Accuracies',
                   'Avg Accuracy']

        df = pd.DataFrame(columns=columns)

        with open('PAVA_by_phrase.csv') as f:
            for line in f.readlines():
                ref_users, test_user, phrase_accuracies, avg_accuracy = \
                    re.match(BY_PHRASE_REGEX, line).groups()
                ref_users = ast.literal_eval(ref_users)
                avg_accuracy = float(avg_accuracy)
                phrase_accuracies = ast.literal_eval(phrase_accuracies)
                df = df.append({
                    k: v for k, v in zip(columns, [ref_users, test_user,
                                                   phrase_accuracies,
                                                   avg_accuracy])
                }, ignore_index=True)

        # for each combination of liopa users, average out phrase accuracies
        best_users = []
        best_accuracy = 0
        overall_avg_phrase_accuracy = {}
        for i in range(1, len(LIOPA_USERS) + 1):
            for ref_users in list(combinations(LIOPA_USERS, i)):
                ref_users = list(ref_users)
                user_rows = df.loc[
                    df['Ref Users'].apply(lambda x: x == ref_users)]
                avg_phrase_accuracies = {}
                for index, row in user_rows.iterrows():
                    phrase_accuracies = row['Phrase Accuracies']
                    for k, v in phrase_accuracies.items():
                        avg_phrase_accuracies[k] = \
                            avg_phrase_accuracies.get(k, 0) + v

                for k, v in avg_phrase_accuracies.items():
                    avg_phrase_accuracies[k] = v / len(user_rows)

                overall_accuracy = sum(avg_phrase_accuracies.values()) / \
                                   len(avg_phrase_accuracies)

                print(ref_users, overall_accuracy)

                if overall_accuracy > best_accuracy:
                    best_accuracy = overall_accuracy
                    best_users = ref_users

                for k, v in avg_phrase_accuracies.items():
                    overall_avg_phrase_accuracy[k] = \
                        overall_avg_phrase_accuracy.get(k, 0) + v

        for k, v in overall_avg_phrase_accuracy.items():
            overall_avg_phrase_accuracy[k] = \
                v / len(list(combinations(LIOPA_USERS, i)))

        print('Best', best_users, best_accuracy)

        phrase_data = read_json_file(configuration.PHRASES_PATH)

        # plot overall phrase accuracies
        sorted_d = {}
        for i, (phrase, accuracy) in \
                enumerate(overall_avg_phrase_accuracy.items()):
            phrase_set, id = re.match(PHRASE_REGEX, phrase).groups()
            phrase_content = phrase_data[phrase_set][id]
            sorted_d[f'{phrase_set} - {phrase_content}'] = accuracy

        sorted_d = {
            k: v for k, v in sorted(sorted_d.items(), key=lambda item: item[1])
        }
        colour_set = 'rgbyk'
        colours = ''
        for phrase in sorted_d.keys():
            phrase_set = phrase.split('-')[0].strip()
            colours += \
                colour_set[configuration.SRAVI_PHRASE_SETS.index(phrase_set)]

        bar_graph(sorted_d.keys(), sorted_d.values(),
                  'Average Phrase Accuracy over every permutation of liopa users',
                  'Phrases', 'Accuracy (%)', rotation=70, colours=colours)

        # group PAVA phrases together and plot
        similar_phrases = {}
        for phrase, accuracy in overall_avg_phrase_accuracy.items():
            phrase_set, id = re.match(PHRASE_REGEX, phrase).groups()
            phrase_content = phrase_data[phrase_set][id]

            if phrase_content in similar_phrases:
                current = similar_phrases[phrase_content]
                similar_phrases[phrase_content] = [current[0] + accuracy,
                                                   current[1] + 1]
            else:
                similar_phrases[phrase_content] = [accuracy, 1]

        for k, v in similar_phrases.items():
            similar_phrases[k] = v[0] / v[1]

        sorted_d = {
            k: v for k, v in sorted(similar_phrases.items(),
                                    key=lambda item: item[1])
        }
        bar_graph(sorted_d.keys(), sorted_d.values(),
                  'Grouped PAVA Equivalent',
                  'Phrases', 'Accuracy (%)', rotation=70)

    _type = {
        'all_vs_all': analyse_all_vs_all,
        'by_phrases': analyse_by_phrases
    }

    _type[kwargs['type']]()


def analyse_2(**kwargs):

    def analyse_by_phrases():
        columns = ['Ref Users', 'Top 3 Accuracy', 'Av Phrase Accuracy',
                   'Phrase Accuracies']

        df = pd.DataFrame(columns=columns)

        with open('PAVA_by_phrase_2.csv') as f:
            for line in f.readlines():
                ref_users, top_3_accuracy, avg_phrase_accuracy, \
                    phrase_accuracies = re.match(BY_PHRASE_2_REGEX, line).groups()
                ref_users = ast.literal_eval(ref_users)
                top_3_accuracy = float(top_3_accuracy)
                avg_phrase_accuracy = float(avg_phrase_accuracy)
                phrase_accuracies = ast.literal_eval(phrase_accuracies)
                df = df.append({
                    k: v for k, v in zip(columns, [ref_users, top_3_accuracy,
                                                   avg_phrase_accuracy,
                                                   phrase_accuracies])
                }, ignore_index=True)

        max_by_top_3 = df.loc[df['Top 3 Accuracy'].idxmax()]
        min_by_top_3 = df.loc[df['Top 3 Accuracy'].idxmin()]
        print(max_by_top_3)
        print(min_by_top_3)

        max_by_phrase_accuracy = df.loc[df['Av Phrase Accuracy'].idxmax()]
        min_by_phrase_accuracy = df.loc[df['Av Phrase Accuracy'].idxmin()]
        print(max_by_phrase_accuracy)
        print(min_by_phrase_accuracy)

        def sort_by_accuracy(d):
            return {
                k: v for k, v in sorted(d.items(), key=lambda x: x[1])
            }

        def convert_pava_phrases_to_names(d):
            pava_phrases = read_json_file(configuration.PHRASES_PATH)
            new_d = {}
            for pava_phrase in d.keys():
                phrase_set, id = re.match(r'(PAVA)(\d+)', pava_phrase).groups()
                content = pava_phrases[phrase_set][id]
                new_d[content] = d[pava_phrase]

            return new_d

        d = sort_by_accuracy(max_by_phrase_accuracy['Phrase Accuracies'])
        d = convert_pava_phrases_to_names(d)
        bar_graph(d.keys(), d.values(), 'Best Phrase Accuracy', 'Phrases',
                  'Accuracy (%)', rotation=70)

        d = sort_by_accuracy(min_by_phrase_accuracy['Phrase Accuracies'])
        d = convert_pava_phrases_to_names(d)
        bar_graph(d.keys(), d.values(), 'Worst Phrase Accuracy', 'Phrases',
                  'Accuracy (%)', rotation=70)

        overall_phrase_accuracies = {}
        accuracy_lookup = {}
        for index, row in df.iterrows():
            ref_users = row['Ref Users']

            # add same accuracy to all permutations of same length
            for p in permutations(ref_users, len(ref_users)):
                accuracy_lookup[p] = row['Top 3 Accuracy']

            phrase_accuracies = row['Phrase Accuracies']
            for phrase, accuracy in phrase_accuracies.items():
                accuracy_lst = overall_phrase_accuracies.get(phrase, [])
                accuracy_lst.append(accuracy)
                overall_phrase_accuracies[phrase] = accuracy_lst

        for phrase, accuracy_lst in overall_phrase_accuracies.items():
            overall_phrase_accuracies[phrase] = \
                sum(accuracy_lst) / len(accuracy_lst)

        overall_phrase_accuracies = sort_by_accuracy(overall_phrase_accuracies)
        overall_phrase_accuracies = \
            convert_pava_phrases_to_names(overall_phrase_accuracies)
        bar_graph(overall_phrase_accuracies.keys(),
                  overall_phrase_accuracies.values(),
                  'Overall Phrase Accuracy',
                  'Phrases', 'Accuracy (%)', rotation=70)

        percentage_improvement = {}
        for i in range(1, len(LIOPA_USERS) + 1):
            for training_users in list(permutations(LIOPA_USERS, i)):
                added_user = list(training_users)[-1]
                previous_users = list(training_users)[:-1]

                if len(previous_users) == 0:
                    continue

                previous_users_accuracy = \
                    accuracy_lookup[tuple(previous_users)]
                added_users_accuracy = \
                    accuracy_lookup[tuple(training_users)]

                percentage_increase = \
                    ((added_users_accuracy - previous_users_accuracy)
                     / previous_users_accuracy) * 100

                # print(previous_users, training_users, added_user,
                #       percentage_increase)

                percentages = percentage_improvement.get(added_user, [])
                percentages.append(percentage_increase)
                percentage_improvement[added_user] = percentages

        for k, v in percentage_improvement.items():
            percentage_improvement[k] = sum(v) / len(v)

        bar_graph([USER_MAPPINGS[str(k)]
                   for k in percentage_improvement.keys()],
                  percentage_improvement.values(),
                  'Average Percentage Increase in Top 3 Accuracy',
                  'Users', 'Percentage Increase')

        best_users = [2, 3, 7]
        for i in range(1, len(best_users) + 1):
            for c in list(combinations(best_users, i)):
                for index, row in df.iterrows():
                    users = row['Ref Users']
                    if users == list(c):
                        print(row)

    def analyse_all_vs_all():
        # file_path = 'PAVA_all_vs_all_2.csv'
        file_path = 'PAVA_subset_vs_all.csv'

        with open(file_path) as f:
            df = pd.DataFrame(f.readlines(), columns=['A'])

        columns = ['Num Samples', 'Num Templates', 'Accuracy', 'Templates']
        df[columns] = df.A.str.split(',', n=3, expand=True)
        df['Templates'] = df['Templates'].str.split(',')
        df = df.drop(['A'], axis=1)

        for c in columns[:-1]:
            df[c] = pd.to_numeric(df[c])

        # group rows by num references and average accuracy
        # plot graph
        ref_accuracies = df.groupby('Num Templates').mean()[['Accuracy']]
        ref_accuracies = ref_accuracies.sort_values('Num Templates')
        ref_accuracies.plot(
            title='Average Accuracy vs. No. of Reference Templates', grid=True
        )
        plt.xlabel('No. Reference Templates')
        plt.ylabel('Accuracy (%)')
        plt.show()

        max_row = df.loc[df['Accuracy'].idxmax()]
        print(max_row)
        min_row = df.loc[df['Accuracy'].idxmin()]
        print(min_row)

        average_accuracy = df['Accuracy'].mean()
        print('Average Accuracy: ', average_accuracy)

        max_rows = df.loc[df['Accuracy'] > average_accuracy]
        min_rows = df.loc[df['Accuracy'] < average_accuracy]

        extract_counts(max_rows)
        extract_counts(min_rows)

        def number_per_session(rows):
            user_templates_lookup = {}
            for index, row in rows.iterrows():
                templates = row['Templates']
                for template in templates:
                    user_id, phrase_id, session_id = \
                        re.match(TEMPLATE_REGEX, template).groups()
                    user_templates = user_templates_lookup.get(user_id, {})
                    sessions = user_templates.get(phrase_id, {})
                    sessions[session_id] = sessions.get(session_id, 0) + 1

                    user_templates[phrase_id] = sessions
                    user_templates_lookup[user_id] = user_templates

            return user_templates_lookup

        sessions_per_user_max = number_per_session(max_rows)
        sessions_per_user_min = number_per_session(min_rows)

        if 'subset' in file_path:
            users = ['2', '3', '7']
        else:
            users = LIOPA_USERS

        # for user_id in users:
        #     for phrase_id, max_sessions in \
        #             sessions_per_user_max[str(user_id)].items():
        #         min_sessions = sessions_per_user_min[str(user_id)][phrase_id]
        #         data = {session: [frequency]
        #                 for session, frequency in max_sessions.items()}
        #         for session, frequency in min_sessions.items():
        #             session_frequency = data.get(session, [])
        #             session_frequency.append(frequency)
        #             data[session] = session_frequency
        #
        #         df = pd.DataFrame(data, index=['Best', 'Worst'])
        #         df.plot(title=f'User {user_id}, Phrase {phrase_id}',
        #                 kind='bar')
        #         plt.xlabel('Best vs Worst')
        #         plt.ylabel('Count')
        #         plt.tight_layout()
        #         plt.show()

        # for user_id in users:
        #     for s in [sessions_per_user_max, sessions_per_user_min]:
        #         data = {}
        #         for phrase_id, sessions in s[str(user_id)].items():
        #             for session, frequency in sessions.items():
        #                 count = data.get(session, [])
        #                 count.append(frequency)
        #                 data[session] = count
        #
        #         df = pd.DataFrame(data, index=list(s[str(user_id)].keys()))
        #         df.plot(title=f'User {user_id} - Frequency of phrases',
        #                 kind='bar')
        #         plt.xlabel('Phrases')
        #         plt.ylabel('Count')
        #         plt.tight_layout()
        #         plt.show()

    _type = {
        'by_phrases': analyse_by_phrases,
        'all_vs_all': analyse_all_vs_all,
    }

    _type[kwargs['type']]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser1 = sub_parsers.add_parser('all_vs_all')
    parser1.add_argument('--num_samples', type=int, default=5)
    parser1.add_argument('--from_phase', type=int, default=1)
    parser1.add_argument('--attempts_to_improve', type=int, default=50)
    parser1.add_argument('--best_top_3_accuracy', type=float, default=0.0)

    parser2 = sub_parsers.add_parser('by_phrases')
    parser2.add_argument('--combo_index', type=int, default=1)

    parser3 = sub_parsers.add_parser('all_vs_all_2')
    parser3.add_argument('--users', type=list, default=LIOPA_USERS)
    parser3.add_argument('--num_samples', type=int, default=1)
    parser3.add_argument('--best_top_3_accuracy', type=float, default=0.0)
    parser3.add_argument('--duplicate_users', type=list, default=None)
    parser3.add_argument('--output_file', type=str,
                         default='PAVA_all_vs_all_2.csv')

    parser4 = sub_parsers.add_parser('by_phrases_2')

    parser_analyse = sub_parsers.add_parser('analyse')
    parser_analyse.add_argument('type', type=str, choices=('all_vs_all',
                                                           'by_phrases'))

    parser_analyse = sub_parsers.add_parser('analyse_2')
    parser_analyse.add_argument('type', type=str, choices=('all_vs_all',
                                                           'by_phrases'))

    args = parser.parse_args()
    f = {
        'all_vs_all': all_vs_all,
        'all_vs_all_2': all_vs_all_2,
        'by_phrases': by_phrases,
        'by_phrases_2': by_phrases_2,
        'analyse': analyse,
        'analyse_2': analyse_2
    }

    if args.run_type in list(f.keys()):
        f[args.run_type](**args.__dict__)
    else:
        parser.print_help()
