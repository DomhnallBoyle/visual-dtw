import json
import time
import uuid
from http import HTTPStatus

import numpy as np
import pytest
from main import configuration, redis_cache
from main.models import PAVAList, PAVASession, PAVATemplate
from main.utils.db import db_session
from main.utils.enums import ListStatus
from main.utils.validators import validate_uuid
from scripts.session_selection import MINIMUM_NUM_TRAINING_TEMPLATES
from tests.integration.utils import generate_response

BASE_ENDPOINT = '/pava/api/v1/lists/{list_id}/model'
MODEL_BUILD_ENDPOINT = BASE_ENDPOINT + '/build'
MODEL_STATUS_ENDPOINT = BASE_ENDPOINT + '/status/{build_id}'

LIST_ID_WITH_MODEL = '23a0a428-1250-47ee-9b60-c7f40e190104'
LIST_ID_FAILED_BUILD = '27a8dc84-c2f0-43a3-b04b-da01eb037fdc'
NON_LIST_ID = 'd85677d0-213e-4cf9-9199-1b154264c4ea'
PHRASE_ID_WITH_LIST_MODEL = 'cef0f766-5245-4bcd-90d4-8c456d6de4cc'
NON_BUILD_ID = '0bf9ede2-0bdb-4b92-852c-8e7818b924e7'


class TestModel:

    @pytest.mark.parametrize('is_build_successful, list_statuses', [
        (False, [3, 4, 1]),  # queued, polled, ready
        (True, [3, 4, 5, 1])  # queued, polled, updating, ready
    ])
    def test_model_build_200(self, client, is_build_successful, list_statuses):
        if is_build_successful:
            # create training templates so model build can kick off properly
            random_blob = np.random.rand(100, 100)
            for _ in range(MINIMUM_NUM_TRAINING_TEMPLATES):
                PAVATemplate.create(blob=random_blob, phrase_id=PHRASE_ID_WITH_LIST_MODEL)

        def get_list_status():
            with db_session() as s:
                return PAVAList.get(s, query=(PAVAList.status,),
                                    filter=(PAVAList.id == LIST_ID_WITH_MODEL),
                                    first=True)[0].name

        # check model status - should be ready
        assert get_list_status() == ListStatus.READY.name

        # apply blocking mechanism to the queue puller
        # signals the actual model is going to be queued next
        # allows time to test queue properties before model is popped off
        redis_cache.rpush(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME, json.dumps({'list_id': 'block'}))
        time.sleep(1)  # ensure blocking is invoked before next code, otherwise num_queued_lists = 2

        # kick off model build
        actual_response = client.post(MODEL_BUILD_ENDPOINT.format(list_id=LIST_ID_WITH_MODEL))
        json_response = actual_response.get_json()
        assert list(json_response['response'].keys()) == ['build_id', 'num_queued_lists']
        build_id = json_response['response']['build_id']
        validate_uuid(build_id)
        assert json_response['response']['num_queued_lists'] == 1
        assert actual_response.status_code == HTTPStatus.OK

        # check session selection queue before it's updated - should contain list id
        ss_queue_item = redis_cache.lindex(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME, 0)
        assert ss_queue_item == '{"list_id": "' + LIST_ID_WITH_MODEL + '", "build_id": "' + build_id + '"}'

        # poll for the list status
        for list_status in list_statuses:
            while True:
                if get_list_status() == ListStatus(list_status).name:
                    break

        # check length of queue at the end - should be empty
        assert redis_cache.llen(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME) == 0

    @pytest.mark.parametrize('list_id, expected_response, status_code', [
        ('invalid_list_id', generate_response(
            status_message='Invalid UUID given: invalid_list_id',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        (NON_LIST_ID, generate_response(
            status_message='List not found',
            status_code=HTTPStatus.NOT_FOUND
        ), HTTPStatus.NOT_FOUND)
    ])
    def test_model_build_failed(self, client, list_id, expected_response, status_code):
        actual_response = client.post(MODEL_BUILD_ENDPOINT.format(list_id=list_id))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    @pytest.mark.parametrize('build_model, build_statuses, list_id', [
        (False, [], LIST_ID_WITH_MODEL),
        (True, ['QUEUED', 'POLLED', 'UPDATING', 'SUCCESS'], LIST_ID_WITH_MODEL),
        (True, ['QUEUED', 'POLLED', 'FAILED'], LIST_ID_FAILED_BUILD)  # no sessions
    ])
    def test_model_status_200(self, client, build_model, build_statuses, list_id):
        if build_model:
            with db_session() as s:
                session_ids = PAVASession.get(s, query=(PAVASession.id,), filter=(PAVASession.list_id == list_id))
            for session_id in session_ids:
                PAVASession.update(id=session_id[0], new=True)

            # kick off model build
            actual_response = client.post(MODEL_BUILD_ENDPOINT.format(list_id=list_id))
            build_id = actual_response.get_json()['response']['build_id']

            # check build status
            for build_status in build_statuses:
                while True:
                    actual_response = client.get(MODEL_STATUS_ENDPOINT.format(list_id=list_id, build_id=build_id))
                    assert actual_response.status_code == HTTPStatus.OK
                    if actual_response.get_json() == generate_response({'status': build_status}):
                        break
        else:
            # build status should be null
            actual_response = client.get(MODEL_STATUS_ENDPOINT.format(list_id=list_id, build_id=NON_BUILD_ID))
            assert actual_response.get_json() == generate_response({'status': None})
            assert actual_response.status_code == HTTPStatus.OK

    @pytest.mark.parametrize('list_id, build_id, expected_response, status_code', [
        ('invalid_list_id', NON_BUILD_ID, generate_response(
            status_message='Invalid UUID given: invalid_list_id',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        (NON_LIST_ID, 'invalid_build_id', generate_response(
            status_message='Invalid UUID given: invalid_build_id',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        (NON_LIST_ID, NON_BUILD_ID, generate_response(
            status_message='List not found',
            status_code=HTTPStatus.NOT_FOUND
        ), HTTPStatus.NOT_FOUND)
    ])
    def test_model_status_failed(self, client, list_id, build_id, expected_response, status_code):
        actual_response = client.get(MODEL_STATUS_ENDPOINT.format(list_id=list_id, build_id=build_id))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code
