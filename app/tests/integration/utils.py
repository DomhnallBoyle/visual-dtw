import io
from http import HTTPStatus

from main import cache, configuration
from main.models import Config, PAVAList, PAVAModel, PAVAModelSession, \
    PAVASession
from main.services.transcribe import transcribe_signal
from main.utils.cmc import CMC
from main.utils.db import db_session
from main.utils.tasks import refresh_cache

DTW_PARAMS = Config().__dict__


def generate_response(response=None, include_status=True,
                      status_message=HTTPStatus.OK.phrase,
                      status_code=HTTPStatus.OK):
    if response is None:
        response = {}

    d = {'response': response}

    if include_status:
        d.update({
            'status': {
                'message': status_message,
                'code': status_code
            }
        })

    return d


def construct_request_video_data(video_path):
    with open(video_path, 'rb') as f:
        return {
            'file': (io.BytesIO(f.read()), video_path)
        }


def create_list_model(list_id, num_default_sessions, to_cache=False):
    """Create custom model from default sessions"""
    model = PAVAModel.create(
        list_id=list_id,
    )

    with db_session() as s:
        default_session_ids = PAVASession.get(
            s, query=(PAVASession.id,),
            filter=(PAVASession.list_id == configuration.DEFAULT_PAVA_LIST_ID)
        )[:num_default_sessions]
        default_session_ids = [str(_id[0]) for _id in default_session_ids]

    # create model sessions from the default sessions
    for session_id in default_session_ids:
        PAVAModelSession.create(
            model_id=model.id,
            session_id=session_id
        )

    # update list to point to new model
    PAVAList.update(
        id=list_id,
        current_model_id=model.id
    )

    assert len(model.get_templates()) == \
           num_default_sessions * configuration.NUM_DEFAULT_PHRASES

    # prompt the cache to update - this would usually be done in API requests
    # administer sleep to make sure cached in time
    if to_cache:
        refresh_cache({'list_id': str(list_id)})

    return model.id, default_session_ids


def are_signals_cached(user_id, list_id):
    return cache.get(f'{str(user_id)}_{str(list_id)}_signals') is not None


def sessions_to_templates(sessions):
    return [
        (label, blob)
        for session_id, templates in sessions
        for label, blob in templates
    ]


def get_accuracy(ref_templates, test_templates):
    num_ranks = 3
    cmc = CMC(num_ranks=num_ranks)

    for actual_label, test_blob in test_templates:
        try:
            predictions = transcribe_signal(ref_templates,
                                            test_blob,
                                            classes=None,
                                            **DTW_PARAMS)
            prediction_labels = [prediction['label']
                                 for prediction in predictions]

            cmc.tally(prediction_labels, actual_label)
        except Exception:
            continue

    cmc.calculate_accuracies(len(test_templates), count_check=False)
    accuracies = cmc.all_rank_accuracies[0]
    assert len(accuracies) == num_ranks

    return accuracies
