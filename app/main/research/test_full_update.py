import argparse
import ast
import math
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config, PAVATemplate
from main.utils.cfe import run_cfe
from main.utils.db import save_default_list_to_file
from main.utils.io import read_json_file, read_pickle_file, write_pickle_file
from main.utils.pre_process import pre_process_signals
from main.services.transcribe import transcribe_signal
from sklearn.model_selection import LeaveOneOut

NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
OUTPUT_FILE_PATHS = ['full_update_at_a_time.csv',
                     'full_update_at_a_time_no_vert.csv',
                     'full_update_bulk.csv']


def get_session_ids(videos_path):
    """Get unique session ids from videos path"""
    # get all sessions
    session_ids = set()
    for video in os.listdir(videos_path):
        if not video.endswith('.mp4'):
            continue
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()
        session_ids.add(int(session_id))

    return list(session_ids)


def train_test_session_split(videos_path, test_session):
    """Split videos into training and test"""
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    # split unseen user templates into sessions
    sessions_to_add = {}
    test_videos = []
    for video in os.listdir(videos_path):
        if not video.endswith('.mp4'):
            continue

        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        video = os.path.join(videos_path, video)

        phrase = pava_phrases[phrase_id]
        if int(session_id) == test_session:
            test_videos.append((phrase, video))
        else:
            session_videos = sessions_to_add.get(int(session_id), [])
            session_videos.append((phrase, video))
            sessions_to_add[int(session_id)] = session_videos

    return sessions_to_add, test_videos


def test_against_saved_sessions(test_videos, pkl_file_path):
    """Get accuracy against saved sessions"""
    accuracy = 0
    num_tests = 0

    sessions = read_pickle_file(pkl_file_path)
    dtw_params = Config().__dict__

    for phrase_label, video in test_videos:
        try:
            with open(video, 'rb') as f:
                test_feature_matrix = run_cfe(f)

            test_feature_matrix = \
                pre_process_signals([test_feature_matrix], **dtw_params)[0]

            ref_signals = [(label, template.blob)
                           for _type, session in sessions
                           for label, template in session]

            predictions = transcribe_signal(ref_signals,
                                            test_feature_matrix,
                                            None, **dtw_params)

            top_prediction = predictions[0]
            prediction_label = top_prediction['label']
            if prediction_label == phrase_label:
                accuracy += 1

            num_tests += 1

        except Exception as e:
            print(e)
            continue

    accuracy /= num_tests
    accuracy *= 100

    return accuracy


def test_against_all_sessions(test_videos):
    """Get accuracy against all saved sessions"""
    return test_against_saved_sessions(test_videos=test_videos,
                                       pkl_file_path='all_sessions.pkl')


def test_against_added_sessions(test_videos):
    """Get accuracy against saved added sessions only"""
    return test_against_saved_sessions(test_videos=test_videos,
                                       pkl_file_path='added_sessions.pkl')


def get_num_templates(lst):
    """Get template count from list"""
    template_count = 0
    for phrase in lst.phrases:
        template_count += len(phrase.templates)

    return template_count


def save_default_as_sessions(file_path='all_sessions.pkl'):
    """Save default list as sessions to pkl file"""

    default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)

    num_templates = get_num_templates(default_lst)
    num_sessions = num_templates // 20

    phrase_templates = {}
    for phrase in default_lst.phrases:
        for template in phrase.templates:
            phrase_ts = phrase_templates.get(phrase.content, [])
            phrase_ts.append(template)
            phrase_templates[phrase.content] = phrase_ts

    sessions = []
    for i in range(num_sessions):
        session = []
        for phrase_content, templates in phrase_templates.items():
            session.append((phrase_content, templates[i]))

        assert len(session) == 20
        sessions.append((f'default_{i+1}', session))

    write_pickle_file(sessions, file_path)


def create_template(video):
    """Create transient template object from video file path"""
    dtw_params = Config().__dict__

    with open(video, 'rb') as f:
        try:
            feature_matrix = run_cfe(f)
        except Exception as e:
            print(e)
            return None

        pre_processed_matrix = \
            pre_process_signals([feature_matrix], **dtw_params)[0]

        return PAVATemplate(blob=pre_processed_matrix)


def add_session(videos, session_id, update_added=True):
    """Add session of videos and save to file"""
    session = []
    for label, video in videos:
        template = create_template(video=video)
        if template:
            session.append((label, template))

    # update all sessions
    sessions = read_pickle_file('all_sessions.pkl')
    sessions.append((f'added_{session_id}', session))
    write_pickle_file(sessions, 'all_sessions.pkl')

    if update_added:
        # updated added sessions only
        added_sessions_file = 'added_sessions.pkl'
        if os.path.exists(added_sessions_file):
            sessions = read_pickle_file(added_sessions_file)
        else:
            sessions = []

        sessions.append((f'added_{session_id}', session))
        write_pickle_file(sessions, added_sessions_file)


def loo_cv(sessions):
    dtw_params = Config().__dict__

    loo = LeaveOneOut()
    accuracies = []
    for train_index, test_index in loo.split(sessions):
        reference = np.array(sessions)[train_index]
        test = np.array(sessions)[test_index][0]
        accuracy = 0

        ref_signals = [(label, template.blob)
                       for _type, session in reference
                       for label, template in session]

        num_tests = 0

        # rank by top 1 prediction
        for actual_label, test_template in test[1]:
            try:
                predictions = transcribe_signal(ref_signals,
                                                test_template.blob,
                                                None, **dtw_params)
            except Exception:
                continue

            top_prediction = predictions[0]
            prediction_label = top_prediction['label']
            if prediction_label == actual_label:
                accuracy += 1

            num_tests += 1

        accuracy /= num_tests
        accuracy *= 100
        accuracies.append(accuracy)

    print(accuracies)

    return accuracies


def remove_session():
    sessions = read_pickle_file('all_sessions.pkl')
    accuracies = loo_cv(sessions)

    # remove session with smallest accuracy
    index_to_remove = accuracies.index(min(accuracies))
    removed_session = sessions[index_to_remove]
    removed_session_type = removed_session[0]

    del sessions[index_to_remove]

    write_pickle_file(sessions, 'all_sessions.pkl')

    return removed_session_type, removed_session, index_to_remove


def remove_worst_session(_type='default'):
    sessions = read_pickle_file('all_sessions.pkl')
    accuracies = loo_cv(sessions)

    removed_sessions = []
    while True:
        # remove session with smallest accuracy
        index_to_remove = accuracies.index(min(accuracies))
        removed_session = sessions[index_to_remove]
        removed_session_type = removed_session[0]

        if _type in removed_session_type:
            del sessions[index_to_remove]
            break
        else:
            # keep track of any that are incorrectly removed
            removed_sessions.append(removed_session)
            del sessions[index_to_remove]
            del accuracies[index_to_remove]

    # re-add the removed sessions
    sessions.extend(removed_sessions)
    write_pickle_file(sessions, 'all_sessions.pkl')

    return removed_session_type, removed_session, index_to_remove


def undo_removal(removed_session, removed_index):
    sessions = read_pickle_file('all_sessions.pkl')
    sessions.insert(removed_index, removed_session)
    write_pickle_file(sessions, 'all_sessions.pkl')


def redo_removal(removed_index):
    sessions = read_pickle_file('all_sessions.pkl')
    del sessions[removed_index]
    write_pickle_file(sessions, 'all_sessions.pkl')


def get_number_of_sessions():
    sessions = read_pickle_file('all_sessions.pkl')

    return len(sessions)


def update_at_a_time(videos_path, user):
    """
    Adding session at a time with reverting:

    1: Hold out session 1 for testing
    2: Get results with default templates
    3: Add session 2, get results
    4: Remove worst session, get results and revert back to default set
    5: Repeat 3 & 4 with other sessions to be added
    6: Repeat 1-5 holding out a different session each time
    """
    print(f'Updating at a time: {user}')

    save_default_list_to_file()
    save_default_as_sessions()

    session_ids = get_session_ids(videos_path)
    for test_session in session_ids:
        sessions_to_add, test_videos = \
            train_test_session_split(videos_path=videos_path,
                                     test_session=test_session)

        default_accuracy = test_against_all_sessions(test_videos=test_videos)

        for session_to_add, videos_to_add in sessions_to_add.items():
            assert len(videos_to_add) == 20

            # add session and get results
            add_session(videos=videos_to_add, session_id=session_to_add)
            accuracy_1 = test_against_all_sessions(test_videos=test_videos)

            # remove session, check whether default or not, get results
            removed_session_type = remove_session()
            accuracy_2 = test_against_all_sessions(test_videos=test_videos)

            # revert back to original list
            save_default_as_sessions()

            # save results
            with open(OUTPUT_FILE_PATHS[0], 'a') as f:
                f.write(f'{user},{test_session},{session_to_add},'
                        f'{default_accuracy},{accuracy_1},{accuracy_2},'
                        f'{removed_session_type}\n')


def update_at_a_time_no_revert(videos_path, user):
    """
    Adding session at a time without reverting:

    1: Hold out session 1 for testing
    2: Get results with default templates
    3: Add session 2, get results
    4: Remove worst session, get results
    5: Repeat 3 & 4 with other sessions to be added
    6: Repeat 1-5 holding out a different session each time
    """
    print(f'Updating at a time (no revert): {user}')

    save_default_list_to_file()

    session_ids = get_session_ids(videos_path)
    for test_session in session_ids:
        # we revert back after each test session
        save_default_as_sessions()

        sessions_to_add, test_videos = \
            train_test_session_split(videos_path=videos_path,
                                     test_session=test_session)

        default_accuracy = test_against_all_sessions(test_videos=test_videos)

        for session_to_add, videos_to_add in sessions_to_add.items():
            assert len(videos_to_add) == 20

            # add session and get results
            add_session(videos=videos_to_add, session_id=session_to_add)
            accuracy_1 = test_against_all_sessions(test_videos=test_videos)

            # remove session, check whether default or not, get results
            removed_session_type = remove_session()
            accuracy_2 = test_against_all_sessions(test_videos=test_videos)

            # no reverting back to original list after removal
            
            # get results vs added sessions so far
            accuracy_3 = test_against_added_sessions(test_videos=test_videos)

            # save results
            with open(OUTPUT_FILE_PATHS[1], 'a') as f:
                f.write(f'{user},{test_session},{session_to_add},'
                        f'{default_accuracy},{accuracy_1},{accuracy_2},'
                        f'{removed_session_type},{accuracy_3}\n')


def update_at_a_time_no_revert_2(videos_path, user):
    """Don't revert after each test session

    Randomly select a test session - get accuracy
    Add every other session randomly until none left to add - get accuracy each time
    Remove session after every add - get accuracy after
    If added session is removed, can it be added again later?
    Repeat
    """
    save_default_as_sessions()

    session_ids = get_session_ids(videos_path)

    while True:
        # get random test session
        test_session = random.sample(session_ids, 1)[0]

        sessions_to_add, test_videos = \
            train_test_session_split(videos_path=videos_path,
                                     test_session=test_session)

        # shuffle the sessions to add
        keys = list(sessions_to_add.keys())
        random.shuffle(keys)
        sessions_to_add = [(key, sessions_to_add[key]) for key in keys]

        removed_added_sessions = []
        default_accuracy = test_against_all_sessions(test_videos=test_videos)
        for session_to_add, videos_to_add in sessions_to_add:
            # add session and get results
            add_session(videos=videos_to_add, session_id=session_to_add)
            accuracy_1 = test_against_all_sessions(test_videos=test_videos)

            # remove session, check whether default or not, get results
            removed_session_type = remove_session()
            accuracy_2 = test_against_all_sessions(test_videos=test_videos)

            if 'added' in removed_session_type:
                removed_added_sessions.append((session_to_add, videos_to_add))

        default_accuracy = test_against_all_sessions(test_videos=test_videos)
        for session_to_add, videos_to_add in removed_added_sessions:
            # add session and get results
            add_session(videos=videos_to_add, session_id=session_to_add)
            accuracy_1 = test_against_all_sessions(test_videos=test_videos)


def update_at_a_time_continually_remove(videos_path, user):
    """
    Continually remove the worst defaults
    """
    save_default_list_to_file()

    session_ids = get_session_ids(videos_path)
    for test_session in session_ids:
        # we revert back after each test session
        save_default_as_sessions()

        sessions_to_add, test_videos = \
            train_test_session_split(videos_path=videos_path,
                                     test_session=test_session)

        # shuffle the sessions to add
        keys = list(sessions_to_add.keys())
        random.shuffle(keys)
        sessions_to_add = [(key, sessions_to_add[key]) for key in keys]

        default_accuracy = test_against_all_sessions(test_videos=test_videos)
        for session_to_add, videos_to_add in sessions_to_add:

            # add session and get results
            add_session(videos=videos_to_add, session_id=session_to_add,
                        update_added=False)
            accuracy_1 = test_against_all_sessions(test_videos=test_videos)

            # remove session, check whether default or not, get results
            removed_session_type = remove_worst_session()
            accuracy_2 = test_against_all_sessions(test_videos=test_videos)

            # save results
            with open('update_at_a_time_continually_remove.csv', 'a') as f:
                f.write(f'{user},{test_session},{session_to_add},'
                        f'{default_accuracy},{accuracy_1},{accuracy_2},'
                        f'{removed_session_type}\n')


def update_at_a_time_undo_redo(videos_path, user):
    """Removing worst added vs removing worst default
    Randomly choose test session
    Continually add other sessions, get accuracy
    Remove session using algorithm, get accuracy, revert
    if removed session is an added session, remove worst default, get accuracy
    and revert
    """
    save_default_list_to_file()

    session_ids = get_session_ids(videos_path)
    for test_session in session_ids:
        # we revert back after each test session
        save_default_as_sessions()

        sessions_to_add, test_videos = \
            train_test_session_split(videos_path=videos_path,
                                     test_session=test_session)

        # shuffle the sessions to add
        keys = list(sessions_to_add.keys())
        random.shuffle(keys)
        sessions_to_add = [(key, sessions_to_add[key]) for key in keys]

        default_accuracy = test_against_all_sessions(test_videos=test_videos)
        for session_to_add, videos_to_add in sessions_to_add:
            # add session and get results
            add_session(videos=videos_to_add, session_id=session_to_add,
                        update_added=False)
            accuracy_1 = test_against_all_sessions(test_videos=test_videos)

            # remove session, check whether default or not, get results
            removed_session_type, removed_session, removed_index = \
                remove_session()
            accuracy_2 = test_against_all_sessions(test_videos=test_videos)

            accuracies = [accuracy_1, accuracy_2]
            removed_session_types = [removed_session_type]

            if 'added' in removed_session_type:
                removed_added_session, removed_added_index = \
                    removed_session, removed_index

                # undo added removal (add added back in)
                undo_removal(removed_added_session, removed_added_index)

                # remove worst default
                removed_type, removed_default_session, removed_default_index = \
                    remove_worst_session('default')
                removed_session_types.append(removed_type)

                # get accuracy
                accuracy_3 = test_against_all_sessions(test_videos=test_videos)
                accuracies.append(accuracy_3)

                # undo default removal
                undo_removal(removed_default_session, removed_default_index)

                # redo added removal
                redo_removal(removed_added_index)
            else:
                # remove an added instead and retest
                removed_default_session, removed_default_index = \
                    removed_session, removed_index

                # undo default removal (add default back in)
                undo_removal(removed_default_session, removed_default_index)

                # remove worst added
                removed_type, removed_added_session, removed_added_index = \
                    remove_worst_session('added')
                removed_session_types.append(removed_type)

                # get accuracy
                accuracy_3 = test_against_all_sessions(test_videos=test_videos)
                accuracies.append(accuracy_3)

                # undo added removal
                undo_removal(removed_added_session, removed_added_index)

                # redo default removal
                redo_removal(removed_default_index)

            # save results
            with open('update_at_a_time_undo_redo.csv', 'a') as f:
                f.write(f'{user},{test_session},{session_to_add},'
                        f'{default_accuracy},{accuracies}'
                        f'{removed_session_types}\n')


def update_at_a_time_new_list(videos_path, user):
    """
    Pretend new list has been created
    Continually add and remove sessions (keep best)
    Hows does the keep-best algorithm do? Does it keep the best?
    What are the accuracy changes? (Keep 2 sessions out for testing maybe)
    """
    from main.research.test_update_list import find_top_sessions

    session_ids = get_session_ids(videos_path)

    for num_to_keep in range(1, 6):
        for test_session in session_ids:
            sessions_to_add, test_videos = \
                train_test_session_split(videos_path=videos_path,
                                         test_session=test_session)

            # shuffle the sessions to add
            keys = list(sessions_to_add.keys())
            random.shuffle(keys)
            sessions_to_add = [(key, sessions_to_add[key]) for key in keys]

            num_added_sessions = 0
            for session_to_add, videos_to_add in sessions_to_add:
                # add session and get results
                add_session(videos=videos_to_add, session_id=session_to_add,
                            update_added=False)

                num_added_sessions += 1

                if num_added_sessions > num_to_keep:
                    all_sessions = read_pickle_file('all_sessions.pkl')
                    top_sessions = find_top_sessions(all_sessions,
                                                     n=num_to_keep)
                    write_pickle_file(top_sessions, 'top_sessions.pkl')
                    num_added_sessions -= 1
                else:
                    top_sessions = read_pickle_file('all_sessions.pkl')
                    write_pickle_file(top_sessions, 'top_sessions.pkl')

                # TODO: Test also vs default sessions, all sessions added at
                #  that point and user & default sessions
                #  this is same as algo on page but with no default sessions
                #  redo both using page and this as references
                accuracy = test_against_saved_sessions(
                    test_videos=test_videos, pkl_file_path='top_sessions.pkl')

                # save results
                with open('update_at_a_time_new_list.csv', 'a') as f:
                    f.write(f'{user},{test_session},'
                            f'{num_to_keep},{session_to_add},{accuracy}\n')


def update_bulk(videos_path, user):
    """
    Adding sessions in bulk:

    1: Hold out session 1 for testing
    2: Get results with default templates
    3: Add session 2-5 in bulk, get results
    4: Continually remove worst session, get results
    5: Repeat 1-4 holding out a different session each time
    """
    print(f'Updating in bulk: {user}')

    save_default_list_to_file()

    session_ids = get_session_ids(videos_path)
    for test_session in [3, 4, 5, 6, 7, 8, 9]:
        save_default_as_sessions()

        sessions_to_add, test_videos = \
            train_test_session_split(videos_path=videos_path,
                                     test_session=test_session)

        # get default accuracy before any sessions added
        default_accuracy = \
            test_against_all_sessions(test_videos=test_videos)

        # add all sessions in bulk
        for session_to_add, videos_to_add in sessions_to_add.items():
            add_session(videos=videos_to_add, session_id=session_to_add,
                        update_added=False)

        # test after add in bulk
        bulk_accuracy = \
            test_against_all_sessions(test_videos=test_videos)

        # continually remove worst session and get results
        remove_results = []
        while True:
            removed_session_type = remove_session()
            accuracy = \
                test_against_all_sessions(test_videos=test_videos)
            remove_results.append((removed_session_type, accuracy))

            num_sessions_left = get_number_of_sessions()
            if num_sessions_left == 1:
                break

        # save results
        with open(OUTPUT_FILE_PATHS[2], 'a') as f:
            f.write(f'{user},{test_session},{default_accuracy},'
                    f'{bulk_accuracy},{remove_results}\n')


def analyse_update_at_a_time(file_path):
    df = pd.read_csv(file_path,
                     names=['User', 'Test Session', 'Added Session',
                            'Default Accuracy', 'Add Accuracy',
                            'Remove Accuracy', 'Remove Type',
                            'Added Session Accuracy'],
                     header=None)

    users = df['User'].unique()

    def percentage_change(x, new, original):
        change = x[new] - x[original]
        percent_change = change / (x[original] * 100)

        return percent_change

    # # getting averages per user
    # for user in users:
    #     print('User: ', user)
    #     sub_df = df[df['User'] == user]
    #
    #     # find average default, added and removed accuracy
    #     average_default = sub_df['Default Accuracy'].mean()
    #     average_added = sub_df['Add Accuracy'].mean()
    #     average_removed = sub_df['Remove Accuracy'].mean()
    #
    #     print('Average default: ', average_default)
    #     print('Average added: ', average_added)
    #     print('Average removed: ', average_removed)
    #
    #     # find average % change when session added and removed
    #     av_percentage_change_add = sub_df.apply(percentage_change,
    #                                             new='Add Accuracy',
    #                                             original='Default Accuracy',
    #                                             axis=1).mean()
    #     av_percentage_change_remove = sub_df.apply(percentage_change,
    #                                                new='Remove Accuracy',
    #                                                original='Add Accuracy',
    #                                                axis=1).mean()
    #     print('Average percentage change add: ', av_percentage_change_add)
    #     print('Average percentage change remove: ', av_percentage_change_remove)
    #
    #     # find frequency of removed sessions
    #     print(sub_df['Remove Type'].value_counts())
    #
    #     print()

    # plotting graphs over user test sessions
    for user in users:
        sub_df = df[df['User'] == user]

        user_sessions = sub_df['Test Session'].unique()
        for session in user_sessions:
            session_df = sub_df[sub_df['Test Session'] == session]
            session_df = session_df.sort_values('Added Session')

            added_removed_sessions =\
                [f'Added {added}\nRemoved {removed}'
                 for added, removed in
                 zip(session_df['Added Session'], session_df['Remove Type'])]

            plt.plot(list(session_df['Default Accuracy']),
                     label=f'Default - Session {session} vs Default List')
            plt.plot(list(session_df['Add Accuracy']), label='Added')
            plt.plot(list(session_df['Remove Accuracy']), label='Removed')
            plt.xticks(np.arange(len(added_removed_sessions)),
                       added_removed_sessions)
            plt.title(f'User {user}, Test Session {session}')
            plt.xlabel('Added & Removed Sessions')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

    # show accuracy increase/decrease as sessions are added and removed
    for user in users:
        sub_df = df[df['User'] == user]

        user_sessions = sub_df['Test Session'].unique()
        num_sessions = len(user_sessions)
        plots_per_row = 3
        num_rows = math.ceil(num_sessions / plots_per_row)

        fig, ax = plt.subplots(num_rows, plots_per_row)
        fig.tight_layout()
        user_accuracies, user_labels = [], []

        for session in user_sessions:
            session_df = sub_df[sub_df['Test Session'] == session]

            default_accuracy = session_df.iloc[0]['Default Accuracy']
            accuracies = [default_accuracy]
            x_labels = ['Default']

            for index, row in session_df.iterrows():
                added_session = row['Added Session']
                added_accuracy = row['Add Accuracy']

                removed_session = row['Remove Type']
                removed_accuracy = row['Remove Accuracy']

                accuracies.extend([added_accuracy, removed_accuracy])
                # if 'default' in removed_session:
                #     removed_session = f'D{removed_session.split("_")[1]}'
                removed_session = removed_session.split('_')[0][0].upper() + \
                    removed_session.split('_')[1]

                x_labels.extend([f'A {added_session}',
                                 f'R {removed_session}'])

            user_accuracies.append(accuracies)
            user_labels.append(x_labels)

            # plt.plot(accuracies)
            # plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
            # plt.xlabel('Added/Removed Sessions')
            # plt.ylabel('Accuracy')
            # plt.title(f'User {user}, Test Session {session}')
            # plt.show()

        k = 0
        for i in range(num_rows):
            for j in range(plots_per_row):
                if k == len(user_accuracies):
                    break
                ax[i, j].plot(user_accuracies[k])
                ax[i, j].set_xticks(np.arange(len(user_labels[k])))
                ax[i, j].set_xticklabels(user_labels[k], rotation=45)
                ax[i, j].set_ylabel('Accuracy')
                ax[i, j].set_title(f'User {user}, Test Session {user_sessions[k]}')
                k += 1

        plt.show()

    # show average accuracy increase and decrease when different sessions
    # where added/removed for each user
    def bar_graphs(x1, y1, x2, y2, title):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.bar(x1, y1)
        ax1.set_xlabel('Added Sessions')
        ax1.set_ylabel('% Increase/Decrease in accuracy')
        ax1.set_title(f'{title} - Added')
        ax1.axhline(0, color='black')

        ax2.bar(x2, y2)
        ax2.set_xlabel('Removed Sessions')
        ax2.set_ylabel('% Increase/Decrease in accuracy')
        ax2.set_title(f'{title} - Removed')
        for tick in ax2.get_xticklabels():
            tick.set_rotation(45)
        ax2.axhline(0, color='black')

        plt.show()

    def bar_graph(x, y, title):
        plt.bar(x, y)
        plt.xlabel('Sessions')
        plt.ylabel('% Increase/Decrease in accuracy')
        plt.title(title)
        plt.axhline(0, color='black')
        plt.show()

    pc = lambda new, original: ((new - original) / original) * 100
    user_default_removed_accuracies = {}
    for user in users:
        sub_df = df[df['User'] == user]
        added_accuracies = {}
        removed_accuracies = {}

        user_sessions = sub_df['Test Session'].unique()
        for session in user_sessions:
            session_df = sub_df[sub_df['Test Session'] == session]

            for i, (index, row) in enumerate(session_df.iterrows()):
                added_session = row['Added Session']
                added_accuracy = row['Add Accuracy']

                if i == 0:
                    previous_accuracy = row['Default Accuracy']
                else:
                    previous_accuracy = session_df.iloc[i-1]['Remove Accuracy']

                add_percent_change = pc(added_accuracy, previous_accuracy)
                accuracies = added_accuracies.get(added_session, [])
                accuracies.append(add_percent_change)
                added_accuracies[added_session] = accuracies

                removed_session = row['Remove Type']
                removed_accuracy = row['Remove Accuracy']
                previous_accuracy = added_accuracy

                remove_percent_change = pc(removed_accuracy, previous_accuracy)
                accuracies = removed_accuracies.get(removed_session, [])
                accuracies.append(remove_percent_change)
                removed_accuracies[removed_session] = accuracies

        # average accuracies
        added_accuracies = {k: sum(v) / len(v)
                            for k, v in added_accuracies.items()}
        removed_accuracies = {k: sum(v) / len(v)
                              for k, v in removed_accuracies.items()}

        # plot graphs
        bar_graphs(list(added_accuracies.keys()),
                   list(added_accuracies.values()),
                   list(removed_accuracies.keys()),
                   list(removed_accuracies.values()),
                   f'User {user}')

        for session, accuracy in removed_accuracies.items():
            if 'default' in session:
                accuracies = user_default_removed_accuracies.get(session, [])
                accuracies.append(accuracy)
                user_default_removed_accuracies[session] = accuracies

    user_default_removed_accuracies = {
        k: sum(v) / len(v) for k, v in user_default_removed_accuracies.items()
    }
    bar_graph(list(user_default_removed_accuracies.keys()),
              list(user_default_removed_accuracies.values()),
              'Av accuracy increase/decrease with removed default sessions')

    # pie chart of removed sessions - average over use test sessions
    def pie_charts(removed_session_counts, default_vs_added_counts, title):
        _sum = sum(list(removed_session_counts.values()))
        p = lambda x: (x / _sum) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(title)

        # pie chart 1
        labels = list(removed_session_counts.keys())
        sizes = [p(v) for v in removed_session_counts.values()]
        patches, texts = ax1.pie(sizes, startangle=90)
        ax1.legend(patches, labels)

        # pie chart 2
        labels = list(default_vs_added_counts.keys())
        sizes = [p(v) for v in default_vs_added_counts.values()]
        patches, texts = ax2.pie(sizes, startangle=90)
        ax2.legend(patches, labels)

        plt.show()

    for user in users:
        sub_df = df[df['User'] == user]

        removed_session_counts = {}

        for index, row in sub_df.iterrows():
            removed_session = row['Remove Type']
            removed_session_counts[removed_session] = \
                removed_session_counts.get(removed_session, 0) + 1

        # combine to default and added removed counts
        combined_removed_session_counts = {'default': 0, 'added': 0}
        for k, v in removed_session_counts.items():
            if 'default' in k:
                combined_removed_session_counts['default'] += v
            else:
                combined_removed_session_counts['added'] += v

        pie_charts(removed_session_counts, combined_removed_session_counts,
                   f'User {user}, default vs added')


def analyse_update_in_bulk(file_path):
    full_update_regex = r'(\d+),(\d+),(\d+.\d+),(\d+.\d+),(\[.+\])'

    columns = ['User', 'Test Session', 'Default Accuracy', 'Bulk Accuracy',
               'Remove results']
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            user, test_session, default_accuracy, bulk_accuracy, \
                remove_results = re.match(full_update_regex, line).groups()
            data.append([
                int(user), int(test_session), float(default_accuracy),
                float(bulk_accuracy), ast.literal_eval(remove_results)
            ])

    df = pd.DataFrame(data=data, columns=columns)

    users = df['User'].unique()

    # show accuracy as more are removed (including default and bulk)
    for user in users:
        sub_df = df[df['User'] == user]
        ys = []
        for index, row in sub_df.iterrows():
            y = [row['Default Accuracy'], row['Bulk Accuracy']]
            for removed_session, removed_accuracy in row['Remove results']:
                y.append(removed_accuracy)
            # plt.plot(y, label=f'Test Session {row["Test Session"]}')
            ys.append(y)

        # plt.xlabel('Number of removed sessions')
        # plt.ylabel('% Accuracy')
        # plt.legend()
        # plt.title(f'User {user} - Accuracy changes after sessions are removed')
        # plt.axvline(0, color='black')
        # plt.axvline(1, color='black')
        # plt.show()

        average_y = []
        num_entries = len(ys[0])
        for i in range(num_entries):
            average = 0
            for y in ys:
                average += y[i]
            average /= len(ys)
            average_y.append(average)

        plt.plot(average_y)
        plt.xlabel('Number of removed sessions')
        plt.ylabel('% Accuracy')
        plt.legend()
        plt.title(f'User {user} - Average accuracy changes after sessions '
                  f'are removed')
        plt.axvline(0, color='orange', label='default')
        plt.axvline(1, color='black', label='bulk')
        plt.legend()
        plt.show()

    # lets see what the composition of the first and second half of the
    # removed sessions looks like
    for user in users:
        sub_df = df[df['User'] == user]
        first_half_counts = {'default': 0, 'added': 0}
        second_half_counts = {'default': 0, 'added': 0}

        for index, row in sub_df.iterrows():
            remove_results = row['Remove results']
            half = len(remove_results) // 2
            first_half = remove_results[:half]
            second_half = remove_results[half:]

            for removed_session, removed_accuracy in first_half:
                first_half_counts[removed_session.split('_')[0]] += 1

            for removed_session, removed_accuracy in second_half:
                second_half_counts[removed_session.split('_')[0]] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f'User {user} - Composition of removed sessions')
        fig.tight_layout()

        print(first_half_counts, second_half_counts)

        _sum = sum(list(first_half_counts.values()))
        p = lambda x: (x / _sum) * 100
        labels = list(first_half_counts.keys())
        sizes = [p(v) for v in first_half_counts.values()]
        patches, texts = ax1.pie(sizes, startangle=90)
        ax1.legend(patches, labels)
        ax1.set_title('First Half')

        _sum = sum(list(second_half_counts.values()))
        p = lambda x: (x / _sum) * 100
        labels = list(second_half_counts.keys())
        sizes = [p(v) for v in second_half_counts.values()]
        patches, texts = ax2.pie(sizes, startangle=90)
        ax2.legend(patches, labels)
        ax2.set_title('Second Half')

        plt.show()

    # what are the chances of default/added sessions being removed from each position?
    for user in users:
        sub_df = df[df['User'] == user]
        position_probs = {}
        num_of_removes = len(sub_df.iloc[0]['Remove results'])
        for i in range(num_of_removes):
            position_probs[i] = {'added': 0, 'default': 0}
            for index, row in sub_df.iterrows():
                removed_session, removed_accuracy = row['Remove results'][i]
                position_probs[i][removed_session.split('_')[0]] += 1

        for position, probs in position_probs.items():
            sum_counts = probs['added'] + probs['default']
            probs['added'] /= sum_counts
            probs['default'] /= sum_counts

        default_line = [probs['default'] for probs in position_probs.values()]
        added_line = [probs['added'] for probs in position_probs.values()]
        labels = list(position_probs.keys())

        plt.plot(labels, default_line)
        plt.ylabel('Probability of removal')
        plt.xlabel('Position')
        plt.title(f'Probability of default sessions being removed at each position for user {user}')
        plt.tight_layout()
        plt.show()

    # most likely for defaults to be first removed, second removed etc
    user_position_counts = {}
    for user in users:
        sub_df = df[df['User'] == user]

        position_counts = {}
        num_of_removes = len(sub_df.iloc[0]['Remove results'])
        for i in range(num_of_removes):
            position_counts[i] = {}
            for index, row in sub_df.iterrows():
                removed_session, removed_accuracy = row['Remove results'][i]
                if 'default' in removed_session:
                    position_sessions = position_counts[i]
                    position_sessions[removed_session] = position_sessions.get(removed_session, 0) + 1
                    position_counts[i] = position_sessions

        user_position_counts[user] = position_counts
        sum_position_counts = {k: sum(v.values()) for k, v in position_counts.items()}
        x = list(sum_position_counts.keys())
        y = list(sum_position_counts.values())
        plt.bar(x, y)
        plt.xlabel('Removed Position')
        plt.ylabel('Removed counts')
        plt.title(f'Count of default sessions removed at each position for user {user}')
        plt.show()

    # show most common removed for each position for all users
    common_per_position = {}
    for user, position_counts in user_position_counts.items():
        for position, session_counts in position_counts.items():
            common_sessions = common_per_position.get(position, {})
            for session, count in session_counts.items():
                common_sessions[session] = common_sessions.get(session, 0) + count
            common_per_position[position] = common_sessions

    to_remove = []
    for k, v in common_per_position.items():
        if not v:
            to_remove.append(k)
            continue
        max_entry = max(v, key=v.get)
        common_per_position[k] = [max_entry, v[max_entry]]

    for k in to_remove:
        del common_per_position[k]

    x = list(common_per_position.keys())
    x_labels = [v[0] for v in common_per_position.values()]
    y = [v[1] for v in common_per_position.values()]
    plt.bar(x, y)
    plt.xlabel('Position')
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    plt.ylabel('Remove counts')
    plt.title(f'Most common removed per position over every user')
    plt.show()

    # what defaults are removed more than others? - grab default counts
    default_counts = {}
    for index, row in df.iterrows():
        remove_results = row['Remove results']
        for removed_session, removed_accuracy in remove_results:
            if 'default' in removed_session:
                default_counts[removed_session] = default_counts.get(removed_session, 0) + 1

    x = list(default_counts.keys())
    y = list(default_counts.values())
    plt.bar(x, y)
    plt.xlabel('Default Sessions')
    plt.ylabel('Removed counts')
    plt.title('Count of default sessions removed')
    plt.show()

    # what's the average % increase/decrease when defaults are removed
    def bar_graph(x, y, title):
        plt.bar(x, y)
        plt.xlabel('Sessions')
        plt.ylabel('% Increase/Decrease in accuracy')
        plt.title(title)
        plt.axhline(0, color='black')
        plt.show()

    pc = lambda new, original: ((new - original) / original) * 100
    default_session_av_acc_changes = {}
    for index, row in df.iterrows():
        remove_results = row['Remove results']
        for i, (removed_session, removed_accuracy) in enumerate(remove_results):
            if 'default' in removed_session:
                if i == 0:
                    previous_accuracy = row['Bulk Accuracy']
                else:
                    previous_accuracy = remove_results[i - 1][1]

                percent_change = pc(removed_accuracy, previous_accuracy)
                acc_changes = \
                    default_session_av_acc_changes.get(removed_session, [])
                acc_changes.append(percent_change)
                default_session_av_acc_changes[removed_session] = acc_changes

    default_session_av_acc_changes = {
        k: sum(v) / len(v) for k, v in default_session_av_acc_changes.items()
    }

    x = list(default_session_av_acc_changes.keys())
    y = list(default_session_av_acc_changes.values())
    bar_graph(x, y, 'Av. % accuracy changes for removed defaults')


def analyse_at_a_time_undo_redo(file_path):
    columns = ['User', 'Test Session', 'Added Session', 'Default Accuracy',
               'Accuracies', 'Removed Session Types']
    line_regex = r'(\d+),(\d+),(\d+),(\d+.\d+),(\[.+\])(\[.+\])'

    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            user, test_session, added_session, default_accuracy, accuracies, \
                removed_session_types = re.match(line_regex, line).groups()
            accuracies = ast.literal_eval(accuracies)
            removed_session_types = ast.literal_eval(removed_session_types)
            data.append([int(user), int(test_session), int(added_session),
                         float(default_accuracy), accuracies,
                         removed_session_types])

    df = pd.DataFrame(columns=columns, data=data)

    users = df['User'].unique()

    pc = lambda new, original: ((new - original) / original) * 100
    list_av = lambda l: sum(l) / len(l)

    # # % accuracy difference between removing added sessions and removing worst
    # # default sessions?
    # for user in users:
    #     sub_df = df[df['User'] == user]
    #
    #     added_pcs, default_pcs = [], []
    #
    #     for index, row in sub_df.iterrows():
    #         accuracy_before, accuracy_after_first_remove, \
    #             accuracy_after_second_remove = row['Accuracies']
    #
    #         first_removed_type, second_removed_type = \
    #             row['Removed Session Types']
    #
    #         pc_first_accuracy = pc(accuracy_after_first_remove, accuracy_before)
    #         pc_second_accuracy = pc(accuracy_after_second_remove, accuracy_before)
    #
    #         if 'added' in first_removed_type:
    #             added_pcs.append(pc_first_accuracy)
    #             default_pcs.append(pc_second_accuracy)
    #         else:
    #             added_pcs.append(pc_second_accuracy)
    #             default_pcs.append(pc_first_accuracy)
    #
    #     average_added_pc = list_av(added_pcs)
    #     average_default_pc = list_av(default_pcs)
    #
    #     x = np.arange(2)
    #     plt.bar(x, [average_added_pc, average_default_pc])
    #     plt.ylabel('% change')
    #     plt.title(f'% change in accuracy after session removals - user {user}')
    #     plt.axhline(0, color='black')
    #     plt.xticks(x, ['Added', 'Default'])
    #     plt.tight_layout()
    #     plt.show()

    # show accuracy changes as sessions are added and removed
    for user in users:
        sub_df = df[df['User'] == user]
        unique_test_sessions = sub_df['Test Session'].unique()

        plots_per_row = 3
        num_rows = math.ceil(len(unique_test_sessions) / plots_per_row)
        fig, ax = plt.subplots(num_rows, plots_per_row)
        fig.tight_layout()
        row_index, column_index = 0, 0

        for test_session in unique_test_sessions:
            sub_sub_df = sub_df[sub_df['Test Session'] == test_session]

            added_accuracies, first_removed_accuracies, \
                second_removed_accuracies = [], [], []
            first_removed_types, second_removed_types = [], []
            default_accuracy = None

            for index, row in sub_sub_df.iterrows():
                if not default_accuracy:
                    default_accuracy = row['Default Accuracy']

                added_accuracy, first_removed_accuracy, second_removed_accuracy \
                    = row['Accuracies']
                added_accuracies.append(added_accuracy)
                first_removed_accuracies.append(first_removed_accuracy)
                second_removed_accuracies.append(second_removed_accuracy)

                first_removed_type, second_removed_type = \
                    row['Removed Session Types']
                first_removed_types.append(first_removed_type[0])
                second_removed_types.append(second_removed_type[0])

            x = [i for i in range(1, len(sub_sub_df) + 1)]
            ax[row_index, column_index].plot(x, added_accuracies,
                                             label='Added', marker='o')
            ax[row_index, column_index].plot(x, first_removed_accuracies,
                                             label='First Removed', marker='o')
            ax[row_index, column_index].plot(x, second_removed_accuracies,
                                             label='Second Removed', marker='o')
            ax[row_index, column_index].plot(x[0], default_accuracy, marker='x')

            for i, (first_removed_type, second_removed_type) \
                in enumerate(zip(first_removed_types, second_removed_types)):
                ax[row_index, column_index].annotate(first_removed_type,
                                                     (x[i], first_removed_accuracies[i]))
                ax[row_index, column_index].annotate(second_removed_type,
                                                     (x[i], second_removed_accuracies[i]))

            ax[row_index, column_index].set_ylabel('% Accuracy')
            ax[row_index, column_index].legend()

            if column_index == plots_per_row - 1:
                column_index = 0
                row_index += 1
            else:
                column_index += 1

        fig.suptitle(f'User {user}')
        plt.show()


def main(args):
    run_type = args.run_type

    if run_type == 'update':
        users = args.users

        # for path in OUTPUT_FILE_PATHS:
        #     if os.path.exists(path):
        #         os.remove(path)

        # loop over all users
        for user in users:
            videos_path = f'/home/domhnall/Documents/sravi_dataset/liopa/{user}'

            # update_at_a_time(videos_path, user)
            # update_at_a_time_no_revert(videos_path, user)
            # update_at_a_time_no_revert_2(videos_path, user)
            # update_bulk(videos_path, user)
            # update_at_a_time_continually_remove(videos_path, user)
            update_at_a_time_undo_redo(videos_path, user)
    elif run_type == 'analyse_bulk':
        analyse_update_in_bulk(args.file_path)
    elif run_type == 'analyse_at_a_time':
        analyse_update_at_a_time(args.file_path)
    elif run_type == 'analyse_at_a_time_undo_redo':
        analyse_at_a_time_undo_redo(args.file_path)


def list_type(s):
    return [int(i) for i in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser1 = sub_parsers.add_parser('update')
    parser1.add_argument('users', type=list_type)

    parser2 = sub_parsers.add_parser('analyse_bulk')
    parser2.add_argument('file_path', type=str)

    parser3 = sub_parsers.add_parser('analyse_at_a_time')
    parser3.add_argument('file_path', type=str)

    parser4 = sub_parsers.add_parser('analyse_at_a_time_undo_redo')
    parser4.add_argument('file_path', type=str)

    main(parser.parse_args())
