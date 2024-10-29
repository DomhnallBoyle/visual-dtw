import glob
import os
import re
import time

import pandas as pd
from main import configuration
from main.models import Config, PAVASession
from main.services.transcribe import transcribe_signal
from main.utils.cfe import run_cfe
from main.utils.cmc import CMC
from main.utils.confusion_matrix import ConfusionMatrix
from main.utils.db import db_session
from main.utils.io import read_json_file, read_pickle_file, \
    write_pickle_file
from main.utils.pre_process import pre_process_signals
from tqdm import tqdm

RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
SRAVI_EXTENDED_REGEX = r'SRAVIExtended-*(.+)-*P(\d+)-*S(\d+)'
DTW_PARAMS = Config().__dict__


class TemplateSlot:
    __slots__ = ['blob']

    def __init__(self, blob):
        self.blob = blob


def preprocess_signal(m):
    return pre_process_signals([m], **DTW_PARAMS)[0]


def unpreprocess_signal(m):
    return m[:, :283]


def get_phrases(phrase_set):
    return read_json_file(configuration.PHRASES_PATH)[phrase_set]


def create_template(video_path, debug=False, preprocess=True):
    with open(video_path, 'rb') as f:
        try:
            feature_matrix = run_cfe(f)
        except Exception as e:
            if debug:
                print(e)
            return None

        if preprocess:
            feature_matrix = preprocess_signal(feature_matrix)

        return TemplateSlot(blob=feature_matrix)


def create_templates(videos_directory, regexes=None, phrase_lookup=None,
                     redo=False, save=False, include_templates=True, include_video_paths=False,
                     phrase_column='Phrase',
                     debug=False, preprocess=True):
    print(f'Getting templates from {videos_directory}')
    templates_path = os.path.join(videos_directory, 'templates.pkl')
    if os.path.exists(templates_path):
        if redo:
            os.remove(templates_path)
        else:
            templates = read_pickle_file(templates_path)
            return templates

    video_paths = glob.glob(os.path.join(videos_directory, '*.mp4'))
    csv_path = os.path.join(videos_directory, 'data.csv')
    csv_exists = os.path.exists(csv_path)

    templates = []
    for video_path in tqdm(video_paths):
        if not csv_exists and regexes:
            for regex in regexes:
                try:
                    user_id, phrase_id, session_id = re.match(
                        regex,
                        os.path.basename(video_path)
                    ).groups()
                    break
                except AttributeError:
                    continue

            phrase_label = f'P{phrase_id}'
            if phrase_lookup:
                phrase_label = phrase_lookup.get(phrase_id)
                if not phrase_label:
                    continue  # skip if phrase doesn't exist in phrase lookup
        else:
            df = pd.read_csv(os.path.join(videos_directory, 'data.csv'))
            sample_id = os.path.basename(video_path).replace('.mp4', '')
            phrase_label = \
                df[df['Sample ID'] == sample_id][phrase_column].values[0]

        combo = [phrase_label]

        if include_templates:
            template = create_template(video_path, debug=debug, preprocess=preprocess)
            if not template:
                continue
            combo.append(template.blob)

        if include_video_paths:
            combo.append(video_path)

        templates.append(combo)

    if save:
        write_pickle_file(templates, templates_path)

    return templates


def create_sessions(videos_directory, regexes, phrase_lookup=None, redo=False,
                    save=False, include_video_paths=False, debug=False,
                    user_id_tag=None):
    print(f'Getting sessions from {videos_directory}')
    sessions_path = os.path.join(videos_directory, 'sessions.pkl')
    if os.path.exists(sessions_path):
        if redo:
            os.remove(sessions_path)
        else:
            sessions = read_pickle_file(sessions_path)
            return sessions

    video_paths = glob.glob(os.path.join(videos_directory, '*.mp4'))
    sessions = {}
    for video_path in tqdm(video_paths):
        for regex in regexes:
            try:
                user_id, phrase_id, session_id = re.match(
                    regex,
                    os.path.basename(video_path)
                ).groups()
                break
            except AttributeError:
                continue

        phrase_label = f'P{phrase_id}'
        if phrase_lookup:
            phrase_label = phrase_lookup.get(phrase_id)
            if not phrase_label:
                continue  # skip if phrase doesn't exist in phrase lookup

        template = create_template(video_path, debug=debug)
        if not template:
            continue

        combo = [phrase_label, template.blob]
        if include_video_paths:
            combo.append(video_path)

        session_templates = sessions.get(session_id, [])
        session_templates.append(tuple(combo))
        sessions[session_id] = session_templates

    if not user_id_tag:
        user_id_tag = user_id

    sessions = [(f'{user_id_tag}-S{session_id}', templates)
                for session_id, templates in sessions.items()]
    if save:
        write_pickle_file(sessions, sessions_path)

    return sessions


def create_templates_doctors(videos_directory, save=False,
                            include_video_paths=False):
    regex = r'(\d+)_(S\d*[A-Z]?)0+(\d+)_S0+(\d+)'

    example_video_path = glob.glob(os.path.join(videos_directory, '*.mp4'))[0]
    phrase_set = re.match(regex, os.path.basename(example_video_path)).groups()[1]

    phrase_lookup = read_json_file(configuration.PHRASES_PATH)[phrase_set]

    return create_templates(
        videos_directory,
        [r'(\d+)_.+0+(\d+)_S0+(\d+)'],
        phrase_lookup,
        save=save,
        include_video_paths=include_video_paths
    )


def get_default_sessions():
    with db_session() as s:
        default_sessions = PAVASession.get(s, filter=(
            PAVASession.list_id == configuration.DEFAULT_PAVA_LIST_ID
        ))

        assert len(default_sessions) == configuration.NUM_DEFAULT_SESSIONS

        return [(f'default_{i+1}',
                 [(template.phrase.content, template.blob)
                  for template in session.templates])
                for i, session in enumerate(default_sessions)]


def sessions_to_templates(sessions):
    return [
        combo
        for session_id, templates in sessions
        for combo in templates
    ]


def transcribe(ref_templates, test_blob):
    predictions = transcribe_signal(ref_templates,
                                    test_blob,
                                    classes=None,
                                    **DTW_PARAMS)

    return predictions


def get_accuracy(ref_templates, test_templates, num_ranks=3,
                 show_av_time=False, debug=False):
    cmc = CMC(num_ranks=num_ranks)
    confusion_matrix = ConfusionMatrix()
    all_predictions = []

    DTW_PARAMS['top_n'] = num_ranks

    if show_av_time:
        start_time = time.time()

    for actual_label, test_blob in test_templates:
        try:
            predictions = transcribe(ref_templates, test_blob)
            prediction_labels = [prediction['label']
                                 for prediction in predictions]

            cmc.tally(prediction_labels, actual_label)
            confusion_matrix.append(prediction_labels[0], actual_label)
            all_predictions.append(predictions)
        except Exception as e:
            if debug:
                print(e)
            all_predictions.append(None)

    if show_av_time:
        end_time = time.time()
        total = end_time - start_time
        average = total / len(test_templates)
        print(f'Total {total} seconds')
        print(f'Average {average} seconds')

    cmc.calculate_accuracies(len(test_templates), count_check=False)
    accuracies = cmc.all_rank_accuracies[0]
    assert len(accuracies) == num_ranks

    return accuracies, confusion_matrix, all_predictions
