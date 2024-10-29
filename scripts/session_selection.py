import argparse
import functools
import json
import gc
import logging
import os
import random
import multiprocessing
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'  # make sure individual cores are used

from main import cache, configuration, redis_cache
from main.models import Config, PAVAList, PAVAModel, PAVAModelSession, \
    PAVAPhrase, PAVASession, PAVATemplate
from main.services.transcribe import transcribe_signal
from main.utils.cmc import CMC
from main.utils.db import db_session
from main.utils.enums import Environment, ListStatus
from main.utils.tasks import test_models
from sqlalchemy.orm import joinedload

DTW_PARAMS = Config().__dict__
MINIMUM_NUM_SESSIONS_TO_ADD = 2
MINIMUM_NUM_SESSIONS_AS_TRAINING = 5
MINIMUM_NUM_TRAINING_TEMPLATES = 60
K = 4
SESSION_SPLIT = 0.7
RANK_WEIGHTS = [0.6, 0.3, 0.1]
BLOCKING_SLEEP_TIME = 5
POLLING_SLEEP_TIME = 0.1


logging.basicConfig(
    filename='/shared/session_selection.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
session_templates, session_template_indexes_lookup, training_templates = \
    None, None, None


def get_session_template_indexes(session_indexes):
    session_template_indexes = []
    for j in session_indexes:
        session_template_indexes.extend([
            *list(range(session_template_indexes_lookup[j][0],
                        session_template_indexes_lookup[j][1]))
        ])

    return session_template_indexes


def get_accuracy(test_data, session_template_indexes):
    num_ranks = 3
    cmc = CMC(num_ranks=num_ranks)

    for actual_label, test_signal in test_data:
        try:
            predictions = transcribe_signal(
                [session_templates[i] for i in session_template_indexes],
                test_signal,
                None,
                **DTW_PARAMS
            )
            prediction_labels = [prediction['label']
                                 for prediction in predictions]
            cmc.tally(prediction_labels, actual_label)
        except Exception:
            continue

    cmc.calculate_accuracies(len(test_data), count_check=False)
    accuracies = cmc.all_rank_accuracies[0]
    assert len(accuracies) == num_ranks

    return accuracies


def process_fold(process_index, testing_start, testing_end, initial_max):
    testing_data = training_templates[testing_start:testing_end]
    training_data = training_templates[:testing_start] + \
                    training_templates[testing_end:]
    assert len(testing_data) + len(training_data) == len(training_templates)

    # first do forward session selection
    selected_session_indexes = []
    session_indexes = [i for i in range(len(session_template_indexes_lookup))]
    while True:
        accuracies = []
        for i in session_indexes:
            session_template_indexes_this_round = \
                get_session_template_indexes([i] + selected_session_indexes)
            accuracy = get_accuracy(
                test_data=training_data,
                session_template_indexes=session_template_indexes_this_round
            )[0]
            accuracies.append(accuracy)

        max_accuracy_index = accuracies.index(max(accuracies))
        best_session_index = session_indexes.pop(max_accuracy_index)
        selected_session_indexes.append(best_session_index)

        logging.info(f'Process {process_index}, '
                     f'num selected sessions {len(selected_session_indexes)}, '
                     f'num leftover {len(session_indexes)}')

        if len(selected_session_indexes) == initial_max:
            best_accuracy = accuracies[max_accuracy_index]

            while True:
                if len(session_indexes) == 0:
                    break

                accuracies = []
                for i in session_indexes:
                    session_template_indexes_this_round = \
                        get_session_template_indexes(
                            [i] + selected_session_indexes
                        )

                    accuracy = get_accuracy(
                        test_data=training_data,
                        session_template_indexes=session_template_indexes_this_round
                    )[0]
                    accuracies.append(accuracy)

                max_accuracy_index = accuracies.index(max(accuracies))
                max_accuracy = accuracies[max_accuracy_index]
                if max_accuracy >= best_accuracy:
                    best_session_index = session_indexes.pop(max_accuracy_index)
                    selected_session_indexes.append(best_session_index)
                    best_accuracy = max_accuracy

                    logging.info(
                        f'Process {process_index}, '
                        f'num selected sessions {len(selected_session_indexes)}, '
                        f'num leftover {len(session_indexes)}'
                    )
                else:
                    break

            break

    # now get accuracy vs the left out test set
    session_template_indexes = \
        get_session_template_indexes(selected_session_indexes)
    accuracy = get_accuracy(
        test_data=testing_data,
        session_template_indexes=session_template_indexes
    )[0]

    # return accuracy and selected sessions indexes only
    return accuracy, selected_session_indexes


def setup(_sessions, _training_templates):
    # remove any sessions that don't have the correct no. of templates
    max_session_length = max([len(templates) for _id, templates in _sessions])
    og_num_sessions = len(_sessions)
    _sessions = [s for s in _sessions if len(s[1]) == max_session_length]
    num_removed = og_num_sessions - len(_sessions)
    logging.info(f'Removed {num_removed} sessions, {len(_sessions)} '
                 f'remaining...')
    if not _sessions:
        logging.info('No sessions available')
        exit()

    # set global variables
    global session_templates, \
        session_template_indexes_lookup, \
        training_templates
    session_templates = [(label, blob)
                         for session_label, ref_session in _sessions
                         for label, blob in ref_session]
    session_template_indexes_lookup = []
    session_labels = []
    for i, (label, templates) in enumerate(_sessions):
        if not session_template_indexes_lookup:
            start = i * len(templates)
        else:
            start = session_template_indexes_lookup[-1][1]
        end = start + len(templates)
        session_template_indexes_lookup.append([start, end])
        session_labels.append(label)
    training_templates = _training_templates
    del _sessions
    gc.collect()

    return session_labels


def session_selection_with_cross_validation(_sessions,
                                            _training_templates,
                                            initial_max):
    """Run the SSA using multiprocessing

    Builds at least N(N+1)/2 models = O(N^2)

    Linux uses copy-on-write which means globals can be used by the spawning
    processes so no extra memory is required for the pickling starmap function.
    This is ideal for the read-only variables. See:
    multiprocessing-in-python-with-read-only-shared-memory [stackoverflow] g!

    Args:
        _sessions (list): sessions to select from
        _training_templates (list): criteria for selecting sessions
        initial_max (int): minimum number of sessions

    Returns:
        list of selected session ids
    """
    if initial_max > len(_sessions):
        logging.info('Initial max cannot be greater than the number of '
                     'sessions')
        exit()

    session_labels = setup(_sessions, _training_templates)

    random.shuffle(training_templates)
    subset_size = len(training_templates) // K

    # divide training and testing between processes
    process_tasks = []
    for i in range(K):
        testing_start = i * subset_size
        testing_end = testing_start + subset_size

        process_tasks.append([
            i+1,
            testing_start,
            testing_end,
            initial_max
        ])

    # multi core processing
    num_processes = K
    accuracies, mixes = [], []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_fold, process_tasks)
        for accuracy, session_set in results:
            accuracies.append(accuracy)
            mixes.append(session_set)

    # check for multiple max accuracies
    # do further testing to get the best best
    max_accuracy = max(accuracies)
    mixes_to_test = [mixes[i] for i, accuracy in enumerate(accuracies)
                     if max_accuracy == accuracy]

    if len(mixes_to_test) > 1:
        accuracies = []
        for mix in mixes_to_test:
            session_template_indexes = \
                get_session_template_indexes(mix)
            accuracies.append(get_accuracy(
                test_data=training_templates,
                session_template_indexes=session_template_indexes
            )[0])
        best_mix = mixes_to_test[accuracies.index(max(accuracies))]
    else:
        best_mix = mixes_to_test[0]

    return [session_labels[i] for i in best_mix]


def weighted_sum(l):
    return sum([l[i] * RANK_WEIGHTS[i] for i in range(len(l))])


def weighted_comparison(x, y):
    return weighted_sum(x[1]) - weighted_sum(y[1])


def process_fold_fast(process_index, testing_start, testing_end,
                      max_num_sessions):
    testing_data = training_templates[testing_start:testing_end]
    training_data = training_templates[:testing_start] + \
                    training_templates[testing_end:]
    assert len(testing_data) + len(training_data) == len(training_templates)

    selected_session_indexes = []
    session_indexes = [i for i in range(len(session_template_indexes_lookup))]

    # first do the ranking - first pass
    all_accuracies = []
    for session_index in session_indexes:
        session_template_indexes_this_round = \
            get_session_template_indexes([session_index])
        accuracies = get_accuracy(
            test_data=training_data,
            session_template_indexes=session_template_indexes_this_round
        )
        all_accuracies.append((session_index, accuracies))
    all_accuracies = sorted(all_accuracies,
                            key=functools.cmp_to_key(weighted_comparison),
                            reverse=True)
    logging.info(f'Process {process_index} finished ranking')

    # now run the second pass
    best = all_accuracies.pop(0)
    selected_session_indexes.append(best[0])
    best_accuracies = best[1]
    for session_index, accuracies in all_accuracies:
        session_template_indexes_this_round = \
            get_session_template_indexes([session_index] +
                                         selected_session_indexes)
        logging.info(f'Process {process_index} processing '
                     f'{len([session_index] + selected_session_indexes)}')
        accuracies = get_accuracy(
            test_data=training_data,
            session_template_indexes=session_template_indexes_this_round
        )
        if weighted_sum(accuracies) > weighted_sum(best_accuracies):
            best_accuracies = accuracies
            selected_session_indexes.append(session_index)

            if max_num_sessions and \
                    len(selected_session_indexes) == max_num_sessions:
                break  # optional early stopping

    # now get accuracy vs the left out test set
    session_template_indexes = \
        get_session_template_indexes(selected_session_indexes)
    accuracies = get_accuracy(
        test_data=testing_data,
        session_template_indexes=session_template_indexes
    )

    return accuracies, selected_session_indexes


def session_selection_with_cross_validation_fast(_sessions,
                                                 _training_templates,
                                                 max_num_sessions=None):
    """Run a faster implementation of SSA using multiprocessing

    Builds at most 2N models = O(N)

    Inspiration from:
    https://medium.com/square-corner-blog/
    comparing-two-forward-feature-selection-algorithms-c52f42868f55

    Args:
        _sessions:
        _training_templates:
        max_num_sessions:

    Returns:

    """
    session_labels = setup(_sessions, _training_templates)

    random.shuffle(training_templates)
    subset_size = len(training_templates) // K

    # divide training and testing between processes
    process_tasks = []
    for i in range(K):
        testing_start = i * subset_size
        testing_end = testing_start + subset_size

        process_tasks.append([
            i+1,
            testing_start,
            testing_end,
            max_num_sessions
        ])

    # multi core processing
    num_processes = K
    all_accuracies, mixes = [], []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_fold_fast, process_tasks)
        for accuracies, session_set in results:
            all_accuracies.append(accuracies)
            mixes.append(session_set)

    indices = list(range(len(all_accuracies)))
    all_accuracies = list(zip(indices, all_accuracies))
    max_accuracies = max(all_accuracies,
                         key=functools.cmp_to_key(weighted_comparison))[1]
    mixes_to_test = [mixes[i] for i, accuracies in all_accuracies
                     if max_accuracies == accuracies]
    if len(mixes_to_test) > 1:
        all_accuracies = []
        for i, mix in enumerate(mixes_to_test):
            session_template_indexes = \
                get_session_template_indexes(mix)
            accuracies = get_accuracy(
                test_data=training_templates,
                session_template_indexes=session_template_indexes
            )
            all_accuracies.append((i, accuracies))
        best_mix_index = max(all_accuracies,
                             key=functools.cmp_to_key(weighted_comparison))[0]
        best_mix = mixes[best_mix_index]
    else:
        best_mix = mixes_to_test[0]

    return [session_labels[i] for i in best_mix]


def session_selection(list_id, build_id=None):
    logging.info('Start')

    def _session_selection():
        with db_session() as s:
            lst = PAVAList.get(s, loading_options=(joinedload('sessions')),
                               filter=(PAVAList.id == list_id), first=True)
            user_id = str(lst.user_id)

        # check if sessions are new and completed i.e. only run algorithm
        # if there are new completed sessions to be added
        new_completed_sessions = [session for session in lst.sessions
                                  if session.completed and session.new]
        if not new_completed_sessions:
            return

        logging.info(f'User {user_id}, list {lst.str_id}: '
                     f'found {len(new_completed_sessions)} '
                     f'new completed session/s')

        # get all user completed sessions for the update (new or not)
        # convert them to correct format
        with db_session() as s:
            user_sessions = [
                (session.id,
                 [(template.phrase.content, template.blob)
                  for template in PAVATemplate.get(
                      s, filter=(PAVATemplate.session_id == session.id))
                  ])
                for session in lst.sessions if session.completed
            ]
            num_user_added_sessions = len(user_sessions)

        logging.info(f'User {user_id}, list {lst.str_id}: '
                     f'total num user added sessions = '
                     f'{len(user_sessions)}, default = {lst.default}')

        # get default sessions if applicable
        if lst.default:
            default_list_id = {
                configuration.DEFAULT_PAVA_LIST_NAME: configuration.DEFAULT_PAVA_LIST_ID,
                configuration.DEFAULT_PAVA_SUB_LIST_NAME: configuration.DEFAULT_PAVA_SUB_LIST_ID
            }[lst.name]

            with db_session() as s:
                default_sessions = [
                    (session.id, [
                        (template.phrase.content, template.blob)
                        for template in session.templates
                    ])
                    for session in PAVASession.get(
                        s, loading_options=(joinedload('templates')),
                        filter=(PAVASession.list_id == default_list_id)
                    )
                ]
            if not default_sessions:
                logging.info(f'User {user_id}, list {lst.str_id}: '
                             f'no default sessions...skipping')
                return

        if num_user_added_sessions >= MINIMUM_NUM_SESSIONS_AS_TRAINING:
            # rely on sessions for training templates if we have enough
            random.shuffle(user_sessions)
            split = int(num_user_added_sessions * SESSION_SPLIT)
            sessions_to_add = user_sessions[:split]
            training_sessions = user_sessions[split:]
            training_templates = [
                (label, blob)
                for session_id, session in training_sessions
                for label, blob in session
            ]
            if lst.default and not lst.has_added_phrases:
                sessions_to_add.extend(default_sessions)

            logging.info(f'User {user_id}, list {lst.str_id}: '
                         f'using sessions as training')
        else:
            # can only use default sessions if using a default list and no phrases have been added
            # TODO: What if the user added 5 phrases to default but then archived all 5 again,
            #  should we use default sessions?
            if lst.default and not lst.has_added_phrases:
                sessions_to_add = user_sessions + default_sessions
            else:
                sessions_to_add = user_sessions

            # can't perform session updating with < min sessions to add
            if len(sessions_to_add) < MINIMUM_NUM_SESSIONS_TO_ADD:
                logging.info(f'User {user_id}, list {lst.str_id}: '
                             f'not enough sessions to add')
                return

            # use other training templates as a backup if we have enough
            # get any training templates that aren't associated with a session
            with db_session() as s:
                # TODO: Should we be checking for equal distribution in training templates here?
                training_templates = [
                    (template.phrase.content, template.blob)
                    for template in PAVATemplate.get(
                        s, filter=((PAVATemplate.session_id == None)
                                   & (PAVATemplate.phrase_id == PAVAPhrase.id)
                                   & (PAVAPhrase.list_id == list_id))
                    )
                    if not template.phrase.is_nota and not template.phrase.archived
                ]

            logging.info(f'User {user_id}, list {lst.str_id}: '
                         f'using ground-truth templates as training')

            # must have a minimum number of training templates
            if len(training_templates) < MINIMUM_NUM_TRAINING_TEMPLATES:
                logging.info(f'User {user_id}, list {lst.str_id}: '
                             f'not enough ground-truth templates')
                return

        PAVAList.update(id=list_id, status=ListStatus.UPDATING)
        if build_id:
            cache.set(build_id, ListStatus.UPDATING.name, configuration.REDIS_MAX_TIMEOUT)

        initial_max = configuration.NUM_DEFAULT_SESSIONS \
            if lst.default and not lst.has_added_phrases \
            else MINIMUM_NUM_SESSIONS_TO_ADD

        logging.info(f'User {user_id}, list {lst.str_id}: starting algorithm. '
                     f'Num sessions to add = {len(sessions_to_add)}, '
                     f'num training templates = {len(training_templates)}, '
                     f'has added phrases = {lst.has_added_phrases}, '
                     f'initial max = {initial_max}')

        # run algorithm, get session ids out
        session_ids = session_selection_with_cross_validation(
            _sessions=sessions_to_add,
            _training_templates=training_templates,
            initial_max=initial_max
        )

        # create user model
        model = PAVAModel.create(list_id=list_id)

        # create model sessions
        for session_id in session_ids:
            PAVAModelSession.create(model_id=model.id, session_id=session_id)

        # set applicable sessions to no longer be new
        for session in new_completed_sessions:
            PAVASession.update(id=session.id, new=False)

        logging.info(f'User {user_id}, list {lst.str_id}: '
                     f'created new model {model.str_id} w/ {len(session_ids)} sessions')

        # update list to point to new model
        PAVAList.update(id=list_id, current_model_id=model.id,
                        status=ListStatus.READY)
        if build_id:
            cache.set(build_id, 'SUCCESS', configuration.REDIS_MAX_TIMEOUT)

        # test all list models
        test_models.delay(list_id)

        return True

    result = _session_selection()

    logging.info('End')

    return result


def pull_and_build():
    # checks for new lists ids before running the selection
    queue_item = redis_cache.lpop(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME)
    if not queue_item:
        time.sleep(POLLING_SLEEP_TIME)
        return

    queue_item = json.loads(queue_item)
    list_id = queue_item['list_id']
    build_id = queue_item.get('build_id')

    if list_id == 'block' and configuration.ENVIRONMENT == Environment.TESTING:
        # if you encounter this signal in testing, sleep
        time.sleep(BLOCKING_SLEEP_TIME)
        return

    PAVAList.update(id=list_id, status=ListStatus.POLLED)
    if build_id:
        cache.set(build_id, ListStatus.POLLED.name, configuration.REDIS_MAX_TIMEOUT)

    result = session_selection(list_id=list_id, build_id=build_id)
    if not result:
        # if ss not successful, need to put the list back to the READY state from POLLED
        PAVAList.update(id=list_id, status=ListStatus.READY)
        if build_id:
            cache.set(build_id, 'FAILED', configuration.REDIS_MAX_TIMEOUT)


def session_selection_loop():
    # constantly pull list ids from the queue and attempt to build models
    while True:
        pull_and_build()


def queue_list_ids():
    # get all unarchived lists and queue them
    with db_session() as s:
        list_ids = PAVAList.get(s, query=(PAVAList.id,),
                                filter=(PAVAList.status == ListStatus.READY))

    for list_id in list_ids:
        # queue list id and update status
        queue_item = json.dumps({'list_id': str(list_id[0])})
        redis_cache.rpush(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME, queue_item)
        PAVAList.update(id=list_id[0], status=ListStatus.QUEUED)


def main(args):
    f = {
        'session_selection_loop': session_selection_loop,
        'queue_list_ids': queue_list_ids
    }
    f[args.run_type]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('session_selection_loop')

    parser_2 = sub_parsers.add_parser('queue_list_ids')

    main(parser.parse_args())
