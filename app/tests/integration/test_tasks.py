import numpy as np
import pytest
from main import cache
from main.models import PAVAList, PAVAModelAccuracy
from main.utils.db import db_session
from main.utils.kvs import KVS
from main.utils.tasks import refresh_cache, test_models as _test_models

USER_ID = '1b7296b8-a7ac-4c9c-a7ed-bb8c6c89795c'
LIST_ID_WITH_MODELS = '23a0a428-1250-47ee-9b60-c7f40e190104'
MODEL_ID = '40e91941-086a-4019-b914-681648342a21'

LIST_ID_NO_MODELS = 'd000ae57-5576-46c8-aabe-8e47602bbf03'
LIST_ID_NO_SESSIONS = '27a8dc84-c2f0-43a3-b04b-da01eb037fdc'
LIST_ID_NO_TEMPLATES = 'fad5d74a-f6ff-42aa-a082-3dd385d285f6'

NON_USER_ID = 'd278bdf9-33f8-4834-b78b-2fcdc51c34c5'
NON_LIST_ID = 'bbbc5742-61ee-4064-91c0-5287f3609cad'
NON_PHRASE_ID = 'd4ff8af5-98d1-4171-88aa-74ae48cbcd63'
NON_SESSION_ID = '9bd9c9ff-11b9-4097-bc18-44623025c440'


def test_refresh_cache_success(client):
    # set model id (cyclic dependency between list and model in fixtures)
    PAVAList.update(id=LIST_ID_WITH_MODELS, current_model_id=MODEL_ID)

    view_args = {'list_id': LIST_ID_WITH_MODELS}
    result = refresh_cache.apply(args=(view_args,)).get()

    assert result

    signals = cache.get(f'{USER_ID}_{LIST_ID_WITH_MODELS}_signals')
    assert signals is not None

    assert isinstance(signals, list)
    assert isinstance(signals[0], tuple)
    assert isinstance(signals[0][0], str)
    assert isinstance(signals[0][1], np.ndarray)  # should be np matrix

    PAVAList.update(id=LIST_ID_WITH_MODELS, current_model_id=None)
    cache.clear()


@pytest.mark.parametrize('view_args', [
    None,
    {},
    {'user_id': NON_USER_ID},
    {'list_id': NON_LIST_ID},
    {'phrase_id': NON_PHRASE_ID},
    {'session_id': NON_SESSION_ID}
])
def test_refresh_cache_failed(client, view_args):
    result = refresh_cache.apply(args=(view_args,)).get()

    assert not result
    assert KVS().keys('*_signals') == []


def test_test_models_success(client):
    result = _test_models.apply(args=(LIST_ID_WITH_MODELS,)).get()
    assert result

    with db_session() as s:
        model_accuracies = \
            PAVAModelAccuracy.get(s, filter=(
                PAVAModelAccuracy.model_id == MODEL_ID
            ))

    # 2 from the fixtures + the 1 new from executing this celery task
    assert len(model_accuracies) == 3

    # get most recently created db entry
    model_accuracy = model_accuracies[-1]
    assert model_accuracy.num_test_templates == 2

    # reset
    PAVAModelAccuracy.delete(id=model_accuracy.id)


@pytest.mark.parametrize('list_id', [
    NON_LIST_ID,
    LIST_ID_NO_MODELS,
    LIST_ID_NO_SESSIONS,
    LIST_ID_NO_TEMPLATES
])
def test_test_models_failed(client, list_id):
    result = _test_models.apply(args=(list_id,)).get()

    assert not result
