import argparse
import ast
import gc
import glob
import json
import multiprocessing
import os
import random
import statistics

import matplotlib.pyplot as plt
from main.models import Config
from main.research.research_utils import get_accuracy, create_sessions, \
    sessions_to_templates
from main.utils.confusion_matrix import ConfusionMatrix
from main.utils.io import read_csv_file, read_pickle_file
from scripts.session_selection import \
    session_selection_with_cross_validation, MINIMUM_NUM_SESSIONS_TO_ADD

K = 5  # num cross-validation folds
NUM_SESSIONS_EACH = 20
NUM_SESSIONS_TO_ADD = 10
NUM_REPEATS = 5
SESSION_SPLIT = 0.5
TRAINING_TEST_SPLIT = 0.7
DTW_PARAMS = Config().__dict__
VIDEO_REGEX = r'(.+)_P0*(\d+)_S0*(\d+).mp4'
PHRASES_IN_ENGLISH = [
    'What\'s the plan?',
    'I feel depressed',
    'Call my family',
    'I\'m hot',
    'I\'m cold',
    'I feel anxious',
    'What time is it?',
    'I don\'t want that',
    'How am I doing?',
    'I need the bathroom',
    'I\'m comfortable',
    'I\'m thirsty',
    'It\'s too bright',
    'I\'m in pain',
    'Move me',
    'It\'s too noisy',
    'Doctor',
    'I\'m hungry',
    'Can I have a cough?',
    'I am scared',
    'My head hurts',
    'My arm is sore',
    'My leg is sore',
    'My chest feels tight',
    'My throat is dry',
    'I have toothache',
    'My ear is sore',
    'My skin is itchy',
    'My nose is runny',
    'I\'m tired',
    'How much is that?',
    'Can I have a bag?',
    'Can you help me?',
    'I would like tea',
    'How are you?',
    'I am feeling good',
    'I feel great!',
    'It\'s good to see you',
    'Thank you',
    'Let\'s go out',
    'Can you speak louder?',
    'Better than yesterday',
    'Worse than yesterday',
    'I\'m happy',
    'I\'m sad',
    'I don\'t want to do that',
    'I want to be alone',
    'I need fresh air',
    'I feel lonely',
    'Row',
    'walking',
    'popular',
    'travel',
    'ranks',
    'industry',
    'bank',
    'thought',
    'deam',
    'become',
    'good man',
    'easy to use',
    'The road is easy to work',
    'habit',
    'curious',
    'yummy',
    'weight',
    'important',
    'repeat',
    'get along',
    'deal with'
]
PHRASES = {
    f'P{i+1}': phrase
    for i, phrase in enumerate(PHRASES_IN_ENGLISH)
}

# using globals to share among multiple processes without using too much
# memory
global default_ref_templates, user_sessions


def create_sessions_generator(videos_directory, k, num_sessions_to_add):
    sessions = create_sessions(videos_directory, VIDEO_REGEX)
    random.shuffle(sessions)
    num_folds, step = 0, 2
    while num_folds < k:
        start = num_folds * step
        end = start + num_sessions_to_add
        num_folds += 1

        yield sessions[start:end], sessions[:start] + sessions[end:]


def get_phrase_accuracies(groundtruths, predictions):
    # first count the groundtruths
    totals = {}
    for groundtruth in groundtruths:
        totals[groundtruth] = totals.get(groundtruth, 0) + 1

    # now tally the correct predictions
    num_correct = {}
    for groundtruth, prediction in zip(groundtruths, predictions):
        if groundtruth == prediction:
            num_correct[groundtruth] = num_correct.get(groundtruth, 0) + 1
        else:
            num_correct[groundtruth] = num_correct.get(groundtruth, 0)

    # divide by total num
    phrase_accuracies = {}
    for phrase, amount in num_correct.items():
        accuracy = amount / totals[phrase]
        phrase_accuracies[phrase] = accuracy

    return phrase_accuracies


def initial_experiment_1(args):
    """PAVA-211 create some speaker-dependent models and test
    Just checking the accuracies hold up for the mandarin set
    """
    sessions = create_sessions(args.videos_directory, VIDEO_REGEX)

    # repeat a number of times
    for repeat in range(args.num_repeats):
        # shuffle and split to  add/training/test sessions
        random.shuffle(sessions)
        split = int(len(sessions) * args.session_split)
        sessions_to_add = sessions[:split]
        training_test_sessions = sessions[split:]

        training_test_split = int(len(training_test_sessions) * 0.7)
        training_sessions = training_test_sessions[:training_test_split]
        test_sessions = training_test_sessions[training_test_split:]

        training_templates = [
            (label, blob)
            for session_id, session in training_sessions
            for label, blob in session
        ]

        test_templates = [
            (label, blob)
            for session_id, session in test_sessions
            for label, blob in session
        ]

        # run algorithm, get session ids out
        print(f'User {args.user_id}: starting algorithm. '
              f'Num sessions to add = {len(sessions_to_add)}, '
              f'num training templates = {len(training_templates)}, '
              f'initial max = {args.initial_max}')
        selected_session_ids = session_selection_with_cross_validation(
            _sessions=sessions_to_add,
            _training_templates=training_templates,
            initial_max=args.initial_max
        )
        print(f'Selected {len(selected_session_ids)} sessions')

        print(f'Testing model vs {len(test_templates)} external templates')
        accuracies = get_accuracy(training_templates, test_templates)[0]

        with open(f'speech_ocean_initial_experiment_1_{args.user_id}.csv',
                  'a') as f:
            f.write(f'{repeat+1},'
                    f'{len(test_templates)},'
                    f'{len(training_templates)},'
                    f'{len(sessions_to_add)},'
                    f'{len(selected_session_ids)},'
                    f'{accuracies}\n')


def initial_experiment_1_analysis(args):
    df = read_csv_file(
        args.results_path,
        ['Repeat', 'Num Test Templates',
         'Num Training Templates', 'Num Sessions To Add',
         'Num Selected Sessions', 'Accuracies'],
        r'(\d+),(\d+),(\d+),(\d+),(\d+),(\[.+\])',
        lambda row: [
            int(row[0]),
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            ast.literal_eval(row[5])
        ]
    )

    # get mean and variance of ranks
    ranks = [[], [], []]
    for index, row in df.iterrows():
        accuracies = row['Accuracies']
        for i, accuracy in enumerate(accuracies):
            ranks[i].append(accuracy)
    for i, rank in enumerate(ranks):
        mean = statistics.mean(rank)
        variance = statistics.variance(rank)
        print(f'Rank: {i+1}, mean: {mean}, variance: {variance}')

    # plots results
    x = [1, 2, 3]
    ys, labels = [], []
    for index, row in df.iterrows():
        ys.append(row['Accuracies'])
        labels.append(f'Repeat {row["Repeat"]}')
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label, marker='x')
    plt.xlabel('Rank')
    plt.ylabel('Accuracy')
    plt.ylim((0, 101))
    plt.xticks(x, x)
    plt.legend()
    plt.title(f'{args.user_id} model accuracies')
    plt.show()


def experiment_2(args):
    """Following 2.1 in the evaluation plan - per speaker models"""
    # running k-fold cross validation
    for i, (sessions_to_add, training_test_sessions) in enumerate(
        create_sessions_generator(
            videos_directory=args.videos_directory,
            k=args.k,
            num_sessions_to_add=args.num_sessions_to_add
        )
    ):
        print(f'CV fold {i+1}')
        print('Sessions to add:',
              [s[0] for s in sessions_to_add])
        print('Training/test sessions:',
              [s[0] for s in training_test_sessions])

        assert len(training_test_sessions) + len(sessions_to_add) \
               == NUM_SESSIONS_EACH
        assert len(training_test_sessions) == len(sessions_to_add) \
               == NUM_SESSIONS_TO_ADD
        assert not any([s1[0] in [s2[0] for s2 in training_test_sessions]
                        for s1 in sessions_to_add])

        # split into training and test templates
        random.shuffle(training_test_sessions)
        training_test_split = int(len(training_test_sessions)
                                  * args.training_test_split)
        training_sessions = training_test_sessions[:training_test_split]
        test_sessions = training_test_sessions[training_test_split:]
        training_templates = [
            (label, blob)
            for session_id, session in training_sessions
            for label, blob in session
        ]
        test_templates = [
            (label, blob)
            for session_id, session in test_sessions
            for label, blob in session
        ]
        del training_sessions, test_sessions, training_test_sessions
        gc.collect()

        # run algorithm, get session ids out
        print(f'User {args.user_id}: starting algorithm. '
              f'Num sessions to add = {len(sessions_to_add)}, '
              f'num training templates = {len(training_templates)}, '
              f'initial max = {args.initial_max}')
        selected_session_ids = session_selection_with_cross_validation(
            _sessions=sessions_to_add,
            _training_templates=training_templates,
            initial_max=args.initial_max
        )
        print(f'Selected {len(selected_session_ids)} sessions')

        selected_sessions = [s for s in sessions_to_add
                             if s[0] in selected_session_ids]
        assert len(selected_sessions) == len(selected_session_ids)
        selected_sessions_templates = [
            (label, blob)
            for session_id, session in selected_sessions
            for label, blob in session
        ]

        print(f'Testing model vs {len(test_templates)} external templates')
        accuracies, confusion_matrix = get_accuracy(
            selected_sessions_templates,
            test_templates
        )
        print(accuracies)

        if args.plot_cm:
            confusion_matrix.plot(blocking=False)
        else:
            confusion_matrix.save(f'{args.user_id}_cm_fold_{i+1}.pkl')

        phrase_accuracies = get_phrase_accuracies(
            confusion_matrix.ground_truths,
            confusion_matrix.predictions
        )
        print(phrase_accuracies)

        with open(f'speech_ocean_experiment_2_{args.user_id}.csv',
                  'a') as f:
            f.write(f'{i+1},'
                    f'{len(sessions_to_add)},'
                    f'{len(training_templates)},'
                    f'{len(test_templates)},'
                    f'{selected_session_ids},'
                    f'{accuracies},'
                    f'{phrase_accuracies}\n')

        # delete no longer needed objects
        del accuracies, phrase_accuracies, confusion_matrix, \
            selected_session_ids
        gc.collect()

    if args.plot_cm:
        plt.show()  # required to prevent windows from closing


def experiment_2_analysis(args):
    csv_directory = os.path.dirname(args.results_path)

    df = read_csv_file(
        args.results_path,
        ['Num Folds', 'Num Sessions To Add',
         'Num Training Templates', 'Num Test Templates',
         'Selected Sessions', 'Accuracies',
         'Phrase Accuracies'],
        r'(\d+),(\d+),(\d+),(\d+),(\[.+\]),(\[.+\]),(\{.+\})',
        lambda row: [
            int(row[0]),
            int(row[1]),
            int(row[2]),
            int(row[3]),
            ast.literal_eval(row[4]),
            ast.literal_eval(row[5]),
            json.loads(row[6].replace('\'', '"'))
        ]
    )

    # average number of selected sessions
    df['Num Selected Sessions'] = \
        df.apply(lambda x: len(x['Selected Sessions']), axis=1)
    print('Average number of selected sessions:',
          df['Num Selected Sessions'].mean())

    # plot rank accuracy graphs
    average_rank_accuracies = [[], [], []]
    x = [1, 2, 3]
    for index, row in df.iterrows():
        y = [round(rank, 1) for rank in row['Accuracies']]
        print(row['Num Selected Sessions'], y)
        plt.plot(x, y, label=f'Repeat {index+1}', marker='o')

        for i, accuracy in enumerate(row['Accuracies']):
            average_rank_accuracies[i].append(accuracy)
    plt.title('Rank Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Rank')
    plt.ylim((0, 100))
    plt.legend()
    plt.xticks(x, x)
    if args.save:
        plt.savefig(os.path.join(csv_directory, f'rank_accuracies.png'))
    plt.show()

    # plot phrase accuracy bar graphs (increasing order)
    for index, row in df.iterrows():
        phrase_accuracies = {k.replace('P', ''): v
                             for k, v in
                             sorted(row['Phrase Accuracies'].items(),
                                    key=lambda item: item[1])}
        x = list(phrase_accuracies.keys())
        y = list(phrase_accuracies.values())
        plt.bar(x, y)
        plt.title(f'{args.user_id} - Repeat {index+1} - Phrase Accuracies')
        plt.ylabel('Accuracy')
        plt.xlabel('Phrases')
        plt.xticks(x, x, rotation='vertical')
        plt.tight_layout()
        if args.save:
            plt.savefig(os.path.join(csv_directory,
                                     f'phrase_accuracies_{index+1}.png'))
        plt.show()

    print('Average Rank Accuracies:')
    for i, rank_accuracies in enumerate(average_rank_accuracies):
        average_rank_accuracy = sum(rank_accuracies) / len(rank_accuracies)
        print(f'R{i+1} - ', round(average_rank_accuracy, 1))

    # 1) find phrases that performed poorly in all 5 folds
    # 2) find the phrases that performed poorly and appeared in every fold
    threshold = 0.4  # anything <= 2/5
    fold_sets = [set() for _ in range(5)]
    for index, row in df.iterrows():
        for phrase, accuracy in row['Phrase Accuracies'].items():
            if accuracy <= threshold:
                fold_sets[index].add(int(phrase.replace('P', '')))
    set_1 = set([p for s in fold_sets for p in s])
    set_2 = set.intersection(*fold_sets)
    print(f'\nWorst performing phrases (threshold <= {threshold}):')
    print('In all folds:', sorted(set_1))
    print('Across all folds:', sorted(set_2))

    # plot box plots of rank accuracies across all 5 repeats/folds
    plt.boxplot(average_rank_accuracies)
    plt.title(f'{args.user_id} - distribution of rank accuracies across all '
              f'folds')
    plt.ylabel('Accuracy')
    plt.xlabel('Rank')
    plt.ylim((0, 100))
    if args.save:
        plt.savefig(os.path.join(csv_directory, f'boxplot.png'))
    plt.show()

    # plot cms if applicable
    if args.plot_cms:
        # plot individual confusion matrices
        cm_paths = glob.glob(os.path.join(csv_directory, '*fold_*.pkl'))
        for i, cm_path in enumerate(cm_paths):
            cm = read_pickle_file(cm_path)
            cm.plot(save_path=os.path.join(csv_directory, f'cm_{i+1}.png'))

        # show average cms plot
        cms = [read_pickle_file(cm_path) for cm_path in cm_paths]
        average_cm = ConfusionMatrix()
        for cm in cms:
            average_cm.extend(
                [pred.replace('P', '') for pred in cm.predictions],
                [gt.replace('P', '') for gt in cm.ground_truths]
            )
        average_cm.plot(save_path=os.path.join(csv_directory, 'av_cm.png'))
        average_cm.plot_phrase_accuracies(
            save_path=os.path.join(csv_directory, f'av_phrase_accuracies.png')
        )
        average_cm.plot_phrase_confusion_count(
            save_path=os.path.join(csv_directory,
                                   f'phrase_confusion_count.png')
        )
        average_cm.plot_network_graph(
            save_path=os.path.join(csv_directory,
                                   f'phrase_confusion_network.png')
        )


def show_phrase_recognition_accuracy_across_all_users(dfs,
                                                      accuracy_column,
                                                      figsize=None,
                                                      save_path=None):
    # find phrase recognition accuracy across all users
    av_phrase_accuracies = {}
    count = 0
    for df in dfs:
        count += len(df)
        for index, row in df.iterrows():
            phrase_accuracies = row[accuracy_column]
            for phrase, accuracy in phrase_accuracies.items():
                av_phrase_accuracies[phrase] = \
                    av_phrase_accuracies.get(phrase, 0) + accuracy
    for phrase in av_phrase_accuracies:
        av_phrase_accuracies[phrase] = av_phrase_accuracies[phrase] / count
    av_phrase_accuracies = {k.replace('P', ''): v
                            for k, v in
                            sorted(av_phrase_accuracies.items(),
                                   key=lambda item: item[1])}

    # group labels together to save space
    accuracy_phrases = {}
    for phrase, accuracy in av_phrase_accuracies.items():
        accuracy = round(accuracy, 2)
        phrases = accuracy_phrases.get(accuracy, [])
        phrases.append(phrase)
        accuracy_phrases[accuracy] = phrases
    accuracy_phrases_copy = accuracy_phrases.copy()
    for accuracy, phrases in accuracy_phrases_copy.items():
        phrases_str = ''
        for i, phrase in enumerate(phrases):
            phrases_str += phrase
            if (i + 1) % 15 == 0:
                phrases_str += '\n'
            elif (i + 1) != len(phrases):
                phrases_str += ', '
        accuracy_phrases[accuracy] = phrases_str

    if figsize:
        plt.figure(figsize=figsize)
    x = list(accuracy_phrases.keys())
    y = range(len(accuracy_phrases.values()))
    plt.barh(y, x)
    plt.yticks(y, list(accuracy_phrases.values()))
    plt.title(f'Average {accuracy_column} across all users')
    plt.ylabel('Phrases')
    plt.xlabel('Accuracy %')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def experiment_2_analysis_all_users(args):
    user_ids = os.listdir(args.results_directory)
    dfs = [
        read_csv_file(
            os.path.join(args.results_directory,
                         user_id,
                         f'speech_ocean_experiment_2_{user_id}.csv'),
            ['Num Folds', 'Num Sessions To Add',
             'Num Training Templates', 'Num Test Templates',
             'Selected Sessions', 'Accuracies',
             'Phrase Accuracies'],
            r'(\d+),(\d+),(\d+),(\d+),(\[.+\]),(\[.+\]),(\{.+\})',
            lambda row: [
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
                ast.literal_eval(row[4]),
                ast.literal_eval(row[5]),
                json.loads(row[6].replace('\'', '"'))
            ]
        )
        for user_id in user_ids if '.png' not in user_id]

    show_phrase_recognition_accuracy_across_all_users(
        dfs,
        'Phrase Accuracies',
        save_path=os.path.join(args.results_directory,
                               f'all_phrase_accuracies.png')
    )


def limit_templates_to_phrases(templates, labels):
    if not labels:
        return templates
    templates = [t for t in templates if t[0] in labels]
    print('Num labels:', len(set([t[0] for t in templates])))

    return templates


def experiment_3_process_fold(index,
                              user_id,
                              personalised_model_session_ids,
                              test_session_ids,
                              limit_to_phrases,
                              results_directory):
    personalised_model = [s for s in user_sessions
                          if s[0] in personalised_model_session_ids]
    test_sessions = [s for s in user_sessions
                     if s[0] in test_session_ids]

    personalised_ref_templates = sessions_to_templates(personalised_model)
    personalised_ref_templates = limit_templates_to_phrases(
        personalised_ref_templates,
        limit_to_phrases
    )
    del personalised_model
    gc.collect()

    test_templates = sessions_to_templates(test_sessions)
    test_templates = limit_templates_to_phrases(test_templates,
                                                limit_to_phrases)
    del test_sessions
    gc.collect()

    # get accuracies and confusion matrices
    default_accuracies, default_cm = get_accuracy(default_ref_templates,
                                                  test_templates)
    personalised_accuracies, personalised_cm = \
        get_accuracy(personalised_ref_templates, test_templates)

    # get phrase accuracies
    default_phrase_accuracies = get_phrase_accuracies(
        default_cm.ground_truths, default_cm.predictions
    )
    personalised_phrase_accuracies = get_phrase_accuracies(
        personalised_cm.ground_truths, personalised_cm.predictions
    )

    # save results
    default_cm.save(os.path.join(results_directory,
                                 f'default_cm_{index}.pkl'))
    personalised_cm.save(os.path.join(results_directory,
                                      f'personalised_cm_{index}.pkl'))

    with open(os.path.join(
            results_directory,
            f'speech_ocean_experiment_3_{user_id}.csv'), 'a') as f:
        f.write(f'{index},'
                f'{default_accuracies},'
                f'{default_phrase_accuracies},'
                f'{personalised_accuracies},'
                f'{personalised_phrase_accuracies}\n')


def experiment_3(args):
    """Following 2.2 in the evaluation plan - speaker independent models

    Testing default model vs personalised models
    """
    results_directory = args.results_directory if args.results_directory \
        else os.path.dirname(args.results_path)

    df = read_csv_file(
        args.results_path,
        ['Num Folds', 'Num Sessions To Add',
         'Num Training Templates', 'Num Test Templates',
         'Selected Sessions', 'Accuracies',
         'Phrase Accuracies'],
        r'(\d+),(\d+),(\d+),(\d+),(\[.+\]),(\[.+\]),(\{.+\})',
        lambda row: [
            int(row[0]),
            int(row[1]),
            int(row[2]),
            int(row[3]),
            ast.literal_eval(row[4]),
            ast.literal_eval(row[5]),
            json.loads(row[6].replace('\'', '"'))
        ]
    )

    global default_ref_templates
    default_model = read_pickle_file(args.default_model_path)
    default_ref_templates = sessions_to_templates(default_model)
    default_ref_templates = limit_templates_to_phrases(default_ref_templates,
                                                       args.limit_to_phrases)
    del default_model
    gc.collect()

    global user_sessions
    user_sessions = create_sessions(args.videos_directory, VIDEO_REGEX,
                                    save=False)
    all_session_ids = [s[0] for s in user_sessions]

    process_tasks = []
    for index, row in df.iterrows():
        personalised_model_session_ids = row['Selected Sessions']
        test_session_ids = list(set(all_session_ids) -
                                set(personalised_model_session_ids))
        assert len(personalised_model_session_ids) + len(test_session_ids) \
               == len(all_session_ids)

        process_tasks.append([
            index + 1,
            args.user_id,
            personalised_model_session_ids,
            test_session_ids,
            args.limit_to_phrases,
            results_directory
        ])

    # run in parallel
    num_processes = len(process_tasks)
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(experiment_3_process_fold, process_tasks)


def experiment_3_analysis(args):
    csv_directory = os.path.dirname(args.results_path)

    df = read_csv_file(
        args.results_path,
        columns=['Index', 'Default Accuracies', 'Default Phrase Accuracies',
                 'Personalised Accuracies', 'Personalised Phrase Accuracies'],
        regex=r'(\d+),(\[.+\]),(\{.+\}),(\[.+\]),(\{.+\})',
        process_line_data=lambda row: [
            int(row[0]),
            ast.literal_eval(row[1]),
            json.loads(row[2].replace('\'', '"')),
            ast.literal_eval(row[3]),
            json.loads(row[4].replace('\'', '"'))
        ]
    )

    # plotting rank accuracies
    all_default_accuracies = [[], [], []]
    all_personalised_accuracies = [[], [], []]
    x = [1, 2, 3]
    for index, row in df.iterrows():
        default_accuracies = row['Default Accuracies']
        personalised_accuracies = row['Personalised Accuracies']
        for i, rank in enumerate(default_accuracies):
            all_default_accuracies[i].append(rank)
        for i, rank in enumerate(personalised_accuracies):
            all_personalised_accuracies[i].append(rank)
    average_default_accuracies = [
        sum(ranks) / len(ranks) for ranks in all_default_accuracies]
    average_personalised_accuracies = [
        sum(ranks) / len(ranks) for ranks in all_personalised_accuracies]

    plt.plot(x, average_default_accuracies, label='Default', marker='o')
    plt.plot(x, average_personalised_accuracies, label='Personalised',
             marker='o')
    plt.title('Rank Accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Rank')
    plt.ylim((0, 100))
    plt.legend()
    plt.xticks(x, x)
    if args.save:
        plt.savefig(os.path.join(csv_directory,
                                 f'default_vs_personalised_rank_accuracies.png'))
    plt.show()

    # show box plots
    def set_bp_colours(bp, colour):
        props = ['whiskers', 'boxes', 'caps', 'fliers', 'medians']
        for prop in props:
            for attribute in bp[prop]:
                plt.setp(attribute, color=colour)

    bp1 = plt.boxplot(all_default_accuracies)
    bp2 = plt.boxplot(all_personalised_accuracies)
    set_bp_colours(bp1, 'blue')
    set_bp_colours(bp2, 'orange')
    plt.ylim((0, 100))
    plt.title(f'{args.user_id} - distribution of rank accuracies')
    plt.ylabel('Accuracy')
    plt.xlabel('Rank')
    plt.tight_layout()
    if args.save:
        plt.savefig(os.path.join(csv_directory,
                                 f'default_vs_personalised_boxplots.png'))
    plt.show()

    # confusion matrix & phrase recognition accuracies
    cm_paths = glob.glob(os.path.join(csv_directory, 'default_cm_*.pkl'))
    cms = [read_pickle_file(cm_path) for cm_path in cm_paths]
    average_cm = ConfusionMatrix()
    for cm in cms:
        average_cm.extend(
            [pred.replace('P', '') for pred in cm.predictions],
            [gt.replace('P', '') for gt in cm.ground_truths]
        )
    average_cm.plot(
        save_path=os.path.join(csv_directory, 'default_cm.png')
        if args.save else None
    )
    average_cm.plot_phrase_accuracies(
        save_path=os.path.join(csv_directory,
                               'default_phrase_accuracies.png')
        if args.save else None
    )
    average_cm.plot_phrase_confusion_count(
        save_path=os.path.join(csv_directory,
                               'default_phrase_confusion_count.png')
        if args.save else None
    )


def experiment_3_analysis_all_users(args):
    user_ids = os.listdir(args.results_directory)
    dfs = [
        read_csv_file(
            os.path.join(args.results_directory,
                         user_id,
                         f'speech_ocean_experiment_3_{user_id}.csv'),
            columns=['Index', 'Default Accuracies',
                     'Default Phrase Accuracies',
                     'Personalised Accuracies',
                     'Personalised Phrase Accuracies'],
            regex=r'(\d+),(\[.+\]),(\{.+\}),(\[.+\]),(\{.+\})',
            process_line_data=lambda row: [
                int(row[0]),
                ast.literal_eval(row[1]),
                json.loads(row[2].replace('\'', '"')),
                ast.literal_eval(row[3]),
                json.loads(row[4].replace('\'', '"'))
            ]
        )
        for user_id in user_ids if '.png' not in user_id]

    show_phrase_recognition_accuracy_across_all_users(
        dfs,
        'Default Phrase Accuracies',
        figsize=(7, 7),
        save_path=os.path.join(args.results_directory,
                               'all_default_phrase_accuracies.png')
    )


def main(args):
    f = {
        'initial_experiment_1': initial_experiment_1,
        'initial_experiment_1_analysis': initial_experiment_1_analysis,
        'experiment_2': experiment_2,
        'experiment_2_analysis': experiment_2_analysis,
        'experiment_2_analysis_all_users': experiment_2_analysis_all_users,
        'experiment_3': experiment_3,
        'experiment_3_analysis': experiment_3_analysis,
        'experiment_3_analysis_all_users': experiment_3_analysis_all_users
    }

    if args.run_type not in f:
        print('Choose one from:', list(f.keys()))
        exit()

    f[args.run_type](args)


def phrase_list(s):
    # e.g. 1-10,12,14-50 etc
    phrases = []
    for entry in s.split(','):
        if '-' in entry:
            _from, _to = entry.split('-')
            phrases.extend([f'P{i}' for i in range(int(_from), int(_to)+1)])
        else:
            phrases.append(f'P{entry}')

    return phrases


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('initial_experiment_1')
    parser_1.add_argument('videos_directory')
    parser_1.add_argument('user_id')
    parser_1.add_argument('--num_repeats', type=int, default=NUM_REPEATS)
    parser_1.add_argument('--session_split', type=float, default=SESSION_SPLIT)
    parser_1.add_argument('--initial_max', type=int,
                          default=MINIMUM_NUM_SESSIONS_TO_ADD)

    parser_2 = sub_parsers.add_parser('initial_experiment_1_analysis')
    parser_2.add_argument('results_path')
    parser_2.add_argument('user_id')

    parser_3 = sub_parsers.add_parser('experiment_2')
    parser_3.add_argument('videos_directory')
    parser_3.add_argument('user_id')
    parser_3.add_argument('--num_sessions_to_add', type=int,
                          default=NUM_SESSIONS_TO_ADD)
    parser_3.add_argument('--k', type=int, default=K)
    parser_3.add_argument('--training_test_split', type=float,
                          default=TRAINING_TEST_SPLIT)
    parser_3.add_argument('--initial_max', type=int,
                          default=MINIMUM_NUM_SESSIONS_TO_ADD)
    parser_3.add_argument('--plot_cm', action='store_true')

    parser_4 = sub_parsers.add_parser('experiment_2_analysis')
    parser_4.add_argument('results_path')
    parser_4.add_argument('user_id')
    parser_4.add_argument('--plot_cms', action='store_true')
    parser_4.add_argument('--save', action='store_true')

    parser_5 = sub_parsers.add_parser('experiment_2_analysis_all_users')
    parser_5.add_argument('results_directory')
    parser_5.add_argument('--save', action='store_true')

    parser_6 = sub_parsers.add_parser('experiment_3')
    parser_6.add_argument('user_id')
    parser_6.add_argument('videos_directory')
    parser_6.add_argument('results_path')
    parser_6.add_argument('default_model_path')
    parser_6.add_argument('--results_directory')
    parser_6.add_argument('--limit_to_phrases', type=phrase_list)

    parser_7 = sub_parsers.add_parser('experiment_3_analysis')
    parser_7.add_argument('results_path')
    parser_7.add_argument('user_id')
    parser_7.add_argument('--save', action='store_true')

    parser_8 = sub_parsers.add_parser('experiment_3_analysis_all_users')
    parser_8.add_argument('results_directory')
    parser_8.add_argument('--save', action='store_true')

    main(parser.parse_args())
