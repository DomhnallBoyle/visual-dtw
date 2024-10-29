import io
import os
from http import HTTPStatus

import pytest
from main import configuration
from main.models import PAVASession
from main.utils.db import db_session
from main.utils.validators import validate_uuid
from tests.integration.utils import construct_request_video_data, \
    generate_response

BASE_ENDPOINT = '/pava/api/v1/sessions/{session_id}/phrases/{phrase_id}'
RECORD_ENDPOINT = BASE_ENDPOINT + '/record'

VIDEO_PATH = os.path.join(configuration.VIDEOS_PATH, 'move_me.mp4')

SESSION_ID = 'ab175bfc-7c36-483a-9c4d-7a1df98d3ae8'
PHRASE_ID_WITHOUT_TEMPLATE = 'd779dc7e-66ad-416c-8da4-df56105b6466'
PHRASE_ID_WITH_TEMPLATE = 'adfef690-12e7-46e8-95be-825ac569d550'


class TestRecord:

    @pytest.mark.parametrize(
        'session_id, phrase_id, completed_before, completed_after, '
        'new_status_after', [
            # replace
            (SESSION_ID, PHRASE_ID_WITH_TEMPLATE, False, False, False),
            # new
            (SESSION_ID, PHRASE_ID_WITHOUT_TEMPLATE, False, True, True)
        ])
    def test_record_phrase_201(self, client, session_id, phrase_id,
                               completed_before, completed_after,
                               new_status_after):
        with db_session() as s:
            session = PAVASession.get(s, filter=(PAVASession.id == SESSION_ID),
                                      first=True)
            assert session.completed == completed_before

        actual_response = client.post(
            RECORD_ENDPOINT.format(session_id=session_id, phrase_id=phrase_id),
            data=construct_request_video_data(VIDEO_PATH)
        )

        json_response = actual_response.get_json()

        assert actual_response.status_code == HTTPStatus.CREATED
        assert list(json_response['response'].keys()) == ['id']
        validate_uuid(json_response['response']['id'])
        assert json_response['status'] == {
            'message': HTTPStatus.CREATED.phrase,
            'code': HTTPStatus.CREATED
        }

        with db_session() as s:
            session = PAVASession.get(s, filter=(PAVASession.id == SESSION_ID),
                                      first=True)
            assert session.completed == completed_after
            assert session.new == new_status_after

    @pytest.mark.parametrize(
        'session_id, phrase_id, data, expected_response, status_code', [
            ('bc2cd04c-62c6-4069-8a8b-e60f990a37da',
             'c89fd539-9579-4c66-967c-c0ef3ce93f6d', None,
             generate_response(
                 status_message='Input payload validation failed: '
                                'A video file is required',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('invalid_session_id', 'c89fd539-9579-4c66-967c-c0ef3ce93f6d',
             {'file': (io.BytesIO(open(VIDEO_PATH, 'rb').read()), VIDEO_PATH)},
             generate_response(
                 status_message='Invalid UUID given: invalid_session_id',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('bc2cd04c-62c6-4069-8a8b-e60f990a37da', 'invalid_phrase_id',
             {'file': (io.BytesIO(open(VIDEO_PATH, 'rb').read()), VIDEO_PATH)},
             generate_response(
                 status_message='Invalid UUID given: invalid_phrase_id',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('bc2cd04c-62c6-4069-8a8b-e60f990a37da', 'invalid_phrase_id',
             {'file': (io.BytesIO(open(VIDEO_PATH, 'rb').read()),
                       'video.avi')},
             generate_response(
                 status_message='Invalid UUID given: invalid_phrase_id',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('bc2cd04c-62c6-4069-8a8b-e60f990a37da',
             'c89fd539-9579-4c66-967c-c0ef3ce93f6d',
             {'file': (io.BytesIO(open(VIDEO_PATH, 'rb').read()), VIDEO_PATH)},
             generate_response(
                 status_message='Session not found',
                 status_code=HTTPStatus.NOT_FOUND
             ), HTTPStatus.NOT_FOUND),
            (SESSION_ID, PHRASE_ID_WITH_TEMPLATE,
             {'file': (None, 'video.mp4')},
             generate_response(
                 status_message='The file is empty',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            (SESSION_ID, PHRASE_ID_WITH_TEMPLATE,
             {'file': None},
             generate_response(
                 status_message='Input payload validation failed: '
                                'A video file is required',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST)
        ])
    def test_record_phrase_failed(self, client, session_id, phrase_id,
                                  data, expected_response, status_code):
        actual_response = client.post(
            RECORD_ENDPOINT.format(session_id=session_id, phrase_id=phrase_id),
            data=data
        )

        assert actual_response.status_code == status_code
        assert actual_response.get_json() == expected_response
