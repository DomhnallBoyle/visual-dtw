import json
import random

import numpy as np
import pytest
import scripts.session_selection as module
from main import configuration
from main.utils.enums import ListStatus
from tests.unit.test_utils import matrix_equal

USER_ID = '81dc30dd-2363-4fd0-869b-007a17561a67'
LIST_ID = 'a436f586-6894-411f-ac0d-064238cc6fba'
MODEL_ID = '234ef4af-4d29-4534-9996-4b9919c3d375'


def generate_templates(mocker, random_signal, amount):
    mock_templates = [
        mocker.create_autospec(module.PAVATemplate) for _ in range(amount)
    ]
    for template in mock_templates:
        template.phrase = mocker.create_autospec(module.PAVAPhrase)
        template.phrase.content = 'Hello'
        template.phrase.is_nota = False
        template.phrase.archived = False
        template.blob = random_signal(250, 250)

    return mock_templates


def generate_data(random_signal, amount):
    return [('I\'m in pain', random_signal(250, 250)) for _ in range(amount)]


def generate_sessions(random_signal, num_sessions=20):
    return [(f'session_{i}', generate_data(random_signal, 20))
            for i in range(num_sessions)]


def generate_selection_order(session_indexes, initial_max, stop_at=None):
    """
    Generate accuracies for get_accuracy() that corresponds to a particular
    session selection order
    """
    selection, accuracies = [], []
    initial_num_sessions = len(session_indexes)

    for i in range(len(session_indexes)):
        if i < initial_max:
            random_accuracies = \
                np.random.randint(1, 100, len(session_indexes)).tolist()
            max_accuracy_index = \
                random_accuracies.index(max(random_accuracies))
            selection.append(session_indexes.pop(max_accuracy_index))
            accuracies.extend([[acc] * 3 for acc in random_accuracies])
        else:
            if stop_at and i == stop_at:
                accuracies.extend([[0] * 3
                                   for _ in range(len(session_indexes))])
                break
            else:
                max_r1_accuracy = max([ranks[0] for ranks in accuracies])
                accuracies.extend([[max_r1_accuracy] * 3
                                   for _ in range(len(session_indexes))])
                selection.append(session_indexes.pop(0))

    count = initial_num_sessions
    if stop_at:
        count = stop_at + 1

    assert len(accuracies) == \
           sum([initial_num_sessions - i for i in range(count)])

    # extra accuracy for final get_accuracy() call in process_fold()
    accuracies.append([np.random.randint(100)] * 3)

    return selection, accuracies


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch.object(module, 'logging')


@pytest.fixture
def mock_db_session(mocker):
    return mocker.patch.object(module, 'db_session')


class TestSessionSelection:

    def test_get_accuracy(self, mocker, random_signal):
        test_data = generate_data(random_signal, 100)
        session_templates = generate_data(random_signal, 100)
        session_template_indexes = [0, 1, 2]

        mock_transcribe = mocker.patch.object(module, 'transcribe_signal')

        module.session_templates = session_templates
        accuracies = module.get_accuracy(
            test_data=test_data,
            session_template_indexes=session_template_indexes
        )

        expected_ref_labels = [session_templates[i][0]
                               for i in session_template_indexes]
        expected_ref_blobs = [session_templates[i][1]
                              for i in session_template_indexes]

        assert all([0 <= accuracy <= 100 for accuracy in accuracies])
        assert mock_transcribe.call_count == len(test_data)

        args = mock_transcribe.call_args
        assert len(args[0]) == 3  # number of args

        # testing args
        actual_ref_signals = args[0][0]
        actual_ref_labels = [ref_signal[0]
                             for ref_signal in actual_ref_signals]
        actual_ref_blobs = [ref_signal[1]
                            for ref_signal in actual_ref_signals]
        assert actual_ref_labels == expected_ref_labels
        assert matrix_equal(actual_ref_blobs, expected_ref_blobs)
        assert matrix_equal(args[0][1], test_data[-1][1])  # test signal
        assert args[0][2] is None

        # testing kwargs
        assert args[1] == module.DTW_PARAMS

    def test_get_accuracy_exception_raised(self, mocker, random_signal):
        test_data = generate_data(random_signal, 100)
        session_templates = generate_data(random_signal, 100)
        session_template_indexes = [0, 1, 2]

        module.session_templates = session_templates
        mock_transcribe = mocker.patch.object(module, 'transcribe_signal')
        mock_transcribe.side_effect = Exception

        accuracy = module.get_accuracy(
            test_data=test_data,
            session_template_indexes=session_template_indexes
        )

        assert accuracy == [0, 0, 0]

    @pytest.mark.parametrize(
        'num_sessions, num_per_session, initial_max, stop_at', [
            (20, 5, 10, None),
            (10, 8, 10, None),
            (20, 9, 10, 14)
        ]
    )
    def test_process_fold(self, mocker, random_signal, num_sessions,
                          num_per_session, initial_max, stop_at):
        session_indexes = [i for i in range(num_sessions)]

        expected_selected_indexes, expected_accuracies = \
            generate_selection_order(session_indexes, initial_max, stop_at)

        module.session_templates = \
            generate_data(random_signal, num_sessions * num_per_session)
        module.training_templates = generate_data(random_signal, 60)
        module.session_template_indexes_lookup = [
            [i * num_per_session, (i * num_per_session) + num_per_session]
            for i in range(num_sessions)
        ]

        mock_get_accuracy = mocker.patch.object(module, 'get_accuracy')
        mock_get_accuracy.side_effect = expected_accuracies

        accuracy, actual_selected_indexes = module.process_fold(
            process_index=1,
            testing_start=0,
            testing_end=15,
            initial_max=initial_max
        )

        assert actual_selected_indexes == expected_selected_indexes
        assert 0 <= accuracy <= 100 and accuracy == expected_accuracies[-1][0]

    @pytest.mark.parametrize('accuracies, multi_max, expected_mix_index', [
        ([50, 60, 80, 70], [], 2),
        ([70, 90, 80, 90], [80, 90], 3)
    ])
    def test_session_selection_with_cross_validation(self, mocker,
                                                     random_signal,
                                                     accuracies,
                                                     multi_max,
                                                     expected_mix_index):
        sessions = generate_sessions(random_signal)
        session_indexes = [i for i in range(len(sessions))]
        training_templates = generate_data(random_signal, 60)
        initial_max = 10
        k = 4
        num_test_templates = 15

        # randomly create session mixes
        accuracies = [[acc] * 3 for acc in accuracies]
        mixes = [random.sample(session_indexes, 5)
                 for _ in range(len(accuracies))]

        mock_random = mocker.patch.object(module, 'random')

        # magic mock has implementations for magic methods e.g. __enter__
        mock_pool = mocker.MagicMock()
        mock_pool.__enter__().starmap.return_value = [
            (accuracy, mix) for accuracy, mix in zip(accuracies, mixes)
        ]
        mock_multipro = mocker.Mock()
        mock_multipro.Pool.return_value = mock_pool
        mocker.patch.object(module, 'multiprocessing', mock_multipro)

        if multi_max:
            mock_get_accuracy = mocker.patch.object(module, 'get_accuracy')
            mock_get_accuracy.side_effect = [[acc] * 3 for acc in multi_max]

        expected_session_ids = [sessions[i][0]
                                for i in mixes[expected_mix_index]]

        actual_session_ids = module.session_selection_with_cross_validation(
            sessions, training_templates, initial_max
        )

        assert actual_session_ids == expected_session_ids
        mock_random.shuffle.assert_called_once_with(training_templates)
        mock_multipro.Pool.assert_called_once_with(processes=module.K)
        mock_pool.__enter__().starmap.assert_called_once_with(
            module.process_fold, [
                [i + 1,
                 i * num_test_templates,
                 (i * num_test_templates) + num_test_templates,
                 initial_max] for i in range(k)
            ]
        )

    @pytest.mark.parametrize('queue_item, ss_result', [
        ({'list_id': LIST_ID}, True),
        ({'list_id': LIST_ID}, None),
        ({'list_id': LIST_ID}, False),
        (None, None),
        ({'list_id': 'block'}, None),
        ({'list_id': LIST_ID, 'build_id': 'fake_build_id'}, True)
    ])
    def test_pull_and_build(self, mocker, queue_item, ss_result):
        if queue_item:
            orig_queue_item = queue_item
            queue_item = json.dumps(queue_item)

        mock_redis_cache = mocker.patch.object(module, 'redis_cache')
        mock_redis_cache.lpop.side_effect = [queue_item]

        mock_cache = mocker.patch.object(module, 'cache')

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_session_selection = mocker.patch.object(module, 'session_selection')
        mock_session_selection.side_effect = [ss_result]
        mock_time = mocker.patch.object(module, 'time')

        module.pull_and_build()

        mock_redis_cache.lpop.assert_called_once_with(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME)
        if queue_item:
            list_id = orig_queue_item['list_id']
            build_id = orig_queue_item.get('build_id')

            if list_id == 'block':
                mock_time.sleep.assert_called_once_with(module.BLOCKING_SLEEP_TIME)
            else:
                pava_cache_calls = []
                if build_id:
                    pava_cache_calls += [mocker.call(build_id, ListStatus.POLLED.name, configuration.REDIS_MAX_TIMEOUT)]

                pava_list_update_calls = [mocker.call(id=list_id, status=ListStatus.POLLED)]
                if not ss_result:
                    pava_list_update_calls += [mocker.call(id=list_id, status=ListStatus.READY)]
                    if build_id:
                        pava_cache_calls += [mocker.call(build_id, 'FAILED', configuration.REDIS_MAX_TIMEOUT)]

                mock_pava_list.update.assert_has_calls(pava_list_update_calls)
                mock_cache.set.assert_has_calls(pava_cache_calls)
                mock_session_selection.assert_called_once_with(list_id=list_id, build_id=build_id)
        else:
            mock_time.sleep.assert_called_once_with(module.POLLING_SLEEP_TIME)

    @pytest.mark.parametrize('list_ids', [
        [('e7db0e84-a0d8-4ab9-9616-7810dc87043c',), ('c299a6d0-700e-49c5-a71d-9be4e3281bbb',)],
        []
    ])
    def test_queue_list_ids(self, mocker, mock_logger, mock_db_session, list_ids):
        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.return_value = list_ids

        mock_redis_cache = mocker.patch.object(module, 'redis_cache')

        module.queue_list_ids()

        mock_pava_list.get.assert_called_once_with(
            mock_db_session().__enter__(),
            query=(mock_pava_list.id,),
            filter=(mock_pava_list.archived == False)
        )
        if list_ids:
            mock_redis_cache.rpush.assert_has_calls([
                mocker.call(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME,
                            '{"list_id": "' + str(list_id[0]) + '"}')
                for list_id in list_ids
            ])

    def test_session_selection_no_new_completed_sessions(self, mocker, mock_logger, mock_db_session):
        mock_list = mocker.create_autospec(module.PAVAList)
        mock_list.user_id = USER_ID
        mock_list.sessions = [
            mocker.MagicMock(completed=False, new=False),
            mocker.MagicMock(completed=True, new=False),
            mocker.MagicMock(completed=False, new=True)
        ]

        mock_joinedload = mocker.patch.object(module, 'joinedload')

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.side_effect = [mock_list]

        module.session_selection(list_id=LIST_ID)

        mock_logger.info.assert_has_calls([
            mocker.call('Start'), mocker.call('End')
        ])
        mock_pava_list.get.assert_called_with(
            mock_db_session().__enter__(),
            loading_options=(mock_joinedload('sessions')),
            filter=(mock_pava_list.id == LIST_ID),
            first=True
        )

    @pytest.mark.parametrize('list_name', ['Default', 'Default sub-list'])
    def test_session_selection_default_list_no_default_sessions(self, mocker, mock_logger,
                                                                random_signal, list_name):
        mock_list = mocker.create_autospec(module.PAVAList)
        mock_list.name = list_name
        mock_list.str_id = LIST_ID
        mock_list.user_id = USER_ID
        mock_list.sessions = [
            mocker.MagicMock(completed=True, new=True),
            mocker.MagicMock(completed=True, new=False)
        ]
        mock_list.default = True

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.side_effect = [mock_list]

        # mocking user templates
        mock_pava_template = mocker.patch.object(module, 'PAVATemplate')
        mock_pava_template.get.return_value = \
            generate_templates(mocker, random_signal, 20)

        # mocking no default sessions
        mock_pava_session = mocker.patch.object(module, 'PAVASession')
        mock_pava_session.get.return_value = []

        module.session_selection(list_id=LIST_ID)

        mock_logger.info.assert_has_calls([
            mocker.call('Start'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'found 1 new completed session/s'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'total num user added sessions = '
                        f'{len(mock_list.sessions)}, '
                        f'default = {mock_list.default}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'no default sessions...skipping'),
            mocker.call('End')
        ])

    @pytest.mark.parametrize('num_user_sessions, default_list, has_added_phrases, '
                             'list_name, initial_max',
                             [(5, False, True, 'Personal List', module.MINIMUM_NUM_SESSIONS_TO_ADD),
                              (10, True, False, 'Default', configuration.NUM_DEFAULT_SESSIONS),
                              (15, True, True, 'Default sub-list', module.MINIMUM_NUM_SESSIONS_TO_ADD)])
    def test_session_selection_enough_user_sessions(self, mocker, mock_db_session,
                                                    mock_logger, random_signal,
                                                    num_user_sessions, default_list,
                                                    has_added_phrases, list_name,
                                                    initial_max):
        mock_list = mocker.create_autospec(module.PAVAList)
        mock_list.name = list_name
        mock_list.str_id = LIST_ID
        mock_list.user_id = USER_ID
        mock_list.sessions = [
            mocker.MagicMock(id=str(_id), completed=True, new=True)
            for _id in range(1, num_user_sessions + 1)
        ]
        mock_list.default = default_list
        mock_list.has_added_phrases = has_added_phrases
        split = int(num_user_sessions * module.SESSION_SPLIT)

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.side_effect = [mock_list]

        # mocking user templates
        mock_pava_template = mocker.patch.object(module, 'PAVATemplate')
        user_templates = generate_templates(mocker, random_signal, 10)
        expanded_user_templates = [(t.phrase.content, t.blob)
                                   for t in user_templates]
        mock_pava_template.get.return_value = user_templates
        expected_user_sessions = [(s.id, expanded_user_templates)
                                  for s in mock_list.sessions]
        expected_sessions_to_add = expected_user_sessions[:split]
        expected_training_templates = \
            [*expanded_user_templates] * (num_user_sessions - split)

        mock_random = mocker.patch.object(module, 'random')

        mock_selection_algorithm = \
            mocker.patch.object(module,
                                'session_selection_with_cross_validation')
        mock_selection_algorithm.return_value = \
            [s.id for s in mock_list.sessions[:split]]

        mock_pava_model = mocker.patch.object(module, 'PAVAModel')
        mock_pava_model.create.return_value = \
            mocker.MagicMock(id=MODEL_ID, str_id=MODEL_ID)

        mock_pava_model_session = mocker.patch.object(module,
                                                      'PAVAModelSession')

        mock_pava_session = mocker.patch.object(module, 'PAVASession')
        if default_list:
            # mocking default sessions
            default_sessions = [
                mocker.MagicMock(
                    id=str(i),
                    templates=generate_templates(mocker, random_signal, 10)
                )
                for i in range(1, configuration.NUM_DEFAULT_SESSIONS + 1)
            ]
            mock_pava_session = mocker.patch.object(module, 'PAVASession')
            mock_pava_session.get.return_value = default_sessions
            if not has_added_phrases:
                expected_sessions_to_add.extend([
                    (s.id, [(t.phrase.content, t.blob) for t in s.templates])
                    for s in default_sessions
                ])

        mock_test_models = mocker.patch.object(module, 'test_models')

        module.session_selection(list_id=LIST_ID)

        mock_random.shuffle.assert_called_once_with(expected_user_sessions)
        mock_pava_list.update.assert_has_calls([
            mocker.call(id=LIST_ID, status=ListStatus.UPDATING),
            mocker.call(id=LIST_ID, current_model_id=MODEL_ID,
                        status=ListStatus.READY)
        ])
        mock_selection_algorithm.assert_called_once_with(
            _sessions=expected_sessions_to_add,
            _training_templates=expected_training_templates,
            initial_max=initial_max
        )
        mock_pava_model.create.assert_called_once_with(list_id=LIST_ID)
        mock_pava_model_session.create.assert_has_calls([
            mocker.call(model_id=MODEL_ID, session_id=session.id)
            for session in mock_list.sessions[:split]
        ])
        mock_pava_session.update.assert_has_calls([
            mocker.call(id=session.id, new=False)
            for session in mock_list.sessions
        ])
        mock_test_models.delay.assert_called_once_with(LIST_ID)
        mock_logger.info.assert_has_calls([
            mocker.call('Start'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'found {num_user_sessions} new completed session/s'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'total num user added sessions = '
                        f'{num_user_sessions}, '
                        f'default = {mock_list.default}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'using sessions as training'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'starting algorithm. Num sessions to add = '
                        f'{len(expected_sessions_to_add)}, num training templates = '
                        f'{len(expected_training_templates)}, '
                        f'has added phrases = {has_added_phrases}, '
                        f'initial max = {initial_max}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'created new model {MODEL_ID} w/ {split} sessions'),
            mocker.call('End')
        ])

    @pytest.mark.parametrize('list_name, default_list, has_added_phrases', [
        ('Personal List', False, False),
        ('Personal List', False, True),
        ('Default', True, True)
    ])
    def test_session_selection_not_enough_user_sessions_to_add(
            self, mocker, mock_logger, random_signal,
            list_name, default_list, has_added_phrases):
        num_user_sessions = 1

        mock_list = mocker.create_autospec(module.PAVAList)
        mock_list.name = list_name
        mock_list.str_id = LIST_ID
        mock_list.user_id = USER_ID
        mock_list.sessions = [
            mocker.MagicMock(id=str(_id), completed=True, new=True)
            for _id in range(1, num_user_sessions + 1)
        ]
        mock_list.default = default_list
        mock_list.has_added_phrases = has_added_phrases

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.side_effect = [mock_list]

        mock_pava_template = mocker.patch.object(module, 'PAVATemplate')
        mock_pava_template.get.return_value = \
            generate_templates(mocker, random_signal, 10)

        # mocking default sessions
        if default_list:
            default_sessions = [
                mocker.MagicMock(
                    id=str(i),
                    templates=generate_templates(mocker, random_signal, 10)
                )
                for i in range(1, configuration.NUM_DEFAULT_SESSIONS + 1)
            ]
            mock_pava_session = mocker.patch.object(module, 'PAVASession')
            mock_pava_session.get.return_value = default_sessions

        module.session_selection(list_id=LIST_ID)

        mock_logger.info.assert_has_calls([
            mocker.call('Start'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'found {num_user_sessions} new completed session/s'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'total num user added sessions = '
                        f'{num_user_sessions}, '
                        f'default = {mock_list.default}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'not enough sessions to add'),
            mocker.call('End')
        ])

    @pytest.mark.parametrize('num_user_sessions, num_training_templates', [
        (num_user_sessions, num_training_templates)
        for num_user_sessions in list(range(2, 5))
        for num_training_templates in [*list(range(1, 60, 10)), 59]
    ])
    def test_session_selection_enough_user_sessions_to_add_not_enough_training_templates(
        self, mocker, mock_logger, random_signal, num_user_sessions,
            num_training_templates):
        mock_list = mocker.create_autospec(module.PAVAList)
        mock_list.str_id = LIST_ID
        mock_list.user_id = USER_ID
        mock_list.sessions = [
            mocker.MagicMock(id=str(_id), completed=True, new=True)
            for _id in range(1, num_user_sessions + 1)
        ]
        mock_list.default = False

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.side_effect = [mock_list]

        # hack to get around the filtering of the 2nd PAVATemplate.get() call
        mock_pava_phrase = mocker.patch.object(module, 'PAVAPhrase')
        mock_pava_phrase.id = None
        mock_pava_phrase.list_id = LIST_ID

        mock_pava_template = mocker.patch.object(module, 'PAVATemplate')
        mock_pava_template.session_id = None
        mock_pava_template.phrase_id = None
        # for some reason, side_effect does not work here for returning
        # multiple generate_templates() functions because PAVATemplate.get()
        # should be ran twice during this unit test
        mock_pava_template.get.return_value = \
            generate_templates(mocker, random_signal, num_training_templates)

        module.session_selection(list_id=LIST_ID)

        mock_logger.info.assert_has_calls([
            mocker.call('Start'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'found {num_user_sessions} new completed session/s'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'total num user added sessions = '
                        f'{num_user_sessions}, '
                        f'default = {mock_list.default}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'using ground-truth templates as training'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'not enough ground-truth templates'),
            mocker.call('End')
        ])

    @pytest.mark.parametrize(
        'default_list, has_added_phrases, list_name, num_user_sessions, '
        'num_sessions_to_add, num_training_templates, initial_max', [
            (True, False, 'Default', 4, 24, 60, configuration.NUM_DEFAULT_SESSIONS),
            (False, False, 'Personal List', 3, 3, 100, module.MINIMUM_NUM_SESSIONS_TO_ADD),
            (True, True, 'Default sub-list', 4, 4, 75, module.MINIMUM_NUM_SESSIONS_TO_ADD)
        ])
    def test_session_selection_enough_user_sessions_to_add_enough_training_templates(
            self, mocker, mock_logger, random_signal,
            default_list, has_added_phrases, list_name, num_user_sessions,
            num_sessions_to_add, num_training_templates, initial_max
    ):
        mock_list = mocker.create_autospec(module.PAVAList)
        mock_list.name = list_name
        mock_list.str_id = LIST_ID
        mock_list.user_id = USER_ID
        mock_list.sessions = [
            mocker.MagicMock(id=str(_id), completed=True, new=True)
            for _id in range(1, num_user_sessions + 1)
        ]
        mock_list.default = default_list
        mock_list.has_added_phrases = has_added_phrases

        mock_pava_list = mocker.patch.object(module, 'PAVAList')
        mock_pava_list.get.side_effect = [mock_list]

        mock_pava_template = mocker.patch.object(module, 'PAVATemplate')
        expected_training_templates = \
            generate_templates(mocker, random_signal, num_training_templates)
        mock_pava_template.get.return_value = expected_training_templates

        expected_sessions_to_add = [
            (s.id, [(t.phrase.content, t.blob)
                    for t in expected_training_templates])
            for s in mock_list.sessions
        ]

        mock_pava_session = mocker.patch.object(module, 'PAVASession')
        if default_list:
            # mocking default sessions
            default_sessions = [
                mocker.MagicMock(
                    id=str(i),
                    templates=generate_templates(mocker, random_signal, 10)
                )
                for i in range(1, configuration.NUM_DEFAULT_SESSIONS + 1)
            ]
            mock_pava_session.get.return_value = default_sessions
            if not has_added_phrases:
                expected_sessions_to_add.extend([
                    (s.id, [(t.phrase.content, t.blob) for t in s.templates])
                    for s in default_sessions
                ])

        # hack to get around the filtering of the 2nd PAVATemplate.get() call
        mock_pava_phrase = mocker.patch.object(module, 'PAVAPhrase')
        mock_pava_phrase.id = None
        mock_pava_phrase.list_id = LIST_ID

        algorithm_return_value = [s.id for s in mock_list.sessions]
        mock_selection_algorithm = \
            mocker.patch.object(module,
                                'session_selection_with_cross_validation')
        mock_selection_algorithm.return_value = algorithm_return_value

        mock_pava_model = mocker.patch.object(module, 'PAVAModel')
        mock_pava_model.create.return_value = \
            mocker.MagicMock(id=MODEL_ID, str_id=MODEL_ID)

        mock_pava_model_session = mocker.patch.object(module,
                                                      'PAVAModelSession')

        mock_test_models = mocker.patch.object(module, 'test_models')

        module.session_selection(list_id=LIST_ID)

        mock_pava_list.update.assert_has_calls([
            mocker.call(id=LIST_ID, status=ListStatus.UPDATING),
            mocker.call(id=LIST_ID, current_model_id=MODEL_ID,
                        status=ListStatus.READY)
        ])
        mock_selection_algorithm.assert_called_once_with(
            _sessions=expected_sessions_to_add,
            _training_templates=[(t.phrase.content, t.blob)
                                 for t in expected_training_templates],
            initial_max=initial_max
        )
        mock_pava_model.create.assert_called_once_with(list_id=LIST_ID)
        mock_pava_model_session.create.assert_has_calls([
            mocker.call(model_id=MODEL_ID, session_id=_id)
            for _id in algorithm_return_value
        ])
        mock_pava_session.update.assert_has_calls([
            mocker.call(id=session.id, new=False)
            for session in mock_list.sessions
        ])
        mock_test_models.delay.assert_called_once_with(LIST_ID)
        mock_logger.info.assert_has_calls([
            mocker.call('Start'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'found {num_user_sessions} new completed session/s'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'total num user added sessions = '
                        f'{num_user_sessions}, default = {default_list}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'using ground-truth templates as training'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'starting algorithm. Num sessions to add = '
                        f'{num_sessions_to_add}, '
                        f'num training templates = {num_training_templates}, '
                        f'has added phrases = {has_added_phrases}, '
                        f'initial max = {initial_max}'),
            mocker.call(f'User {USER_ID}, list {LIST_ID}: '
                        f'created new model {MODEL_ID} w/ {len(algorithm_return_value)} sessions'),
            mocker.call('End')
        ])
