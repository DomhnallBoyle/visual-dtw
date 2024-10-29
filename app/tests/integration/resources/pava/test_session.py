from http import HTTPStatus

import pytest
from main.utils.validators import validate_uuid
from tests.integration.utils import generate_response

BASE_ENDPOINT = '/pava/api/v1/lists/{list_id}/sessions'

CREATE_SESSION_ENDPOINT = BASE_ENDPOINT
GET_SESSIONS_ENDPOINT = BASE_ENDPOINT
GET_SESSION_ENDPOINT = BASE_ENDPOINT + '/{session_id}'

LIST_ID = 'a8c2471e-ebb6-4fcb-a1dd-39d54f78bb1e'
LIST_ID_PHRASES = 'd000ae57-5576-46c8-aabe-8e47602bbf03'
LIST_ID_NO_PHRASES = 'bdd65b49-7476-4fa6-9e20-09bbec995e08'

SESSION_ID_INCOMPLETE = '063fbe3c-16ba-403d-a366-68ea0ccd4a7c'
SESSION_ID_COMPLETE = 'fecb5a31-f603-4d8d-b935-8ca80c731b17'


class TestSession:

    def test_create_session_201(self, client):
        actual_response = client.post(
            CREATE_SESSION_ENDPOINT.format(list_id=LIST_ID)
        )
        json_response = actual_response.get_json()['response']

        assert actual_response.status_code == HTTPStatus.CREATED
        assert set(json_response.keys()) == {'id', 'completed', 'phrases'}
        validate_uuid(json_response['id'])
        assert json_response['completed'] is False
        assert json_response['phrases'] == [
            {
                'id': 'adfef690-12e7-46e8-95be-825ac569d550',
                'content': 'Thank you',
                'template_id': None,
                'in_model': False
            },
            {
                'id': 'd779dc7e-66ad-416c-8da4-df56105b6466',
                'content': 'Move me',
                'template_id': None,
                'in_model': False
            }
        ]

    @pytest.mark.parametrize('list_id, expected_response, status_code', [
        ('invalid_list_uuid', generate_response(
            status_message='Invalid UUID given: invalid_list_uuid',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        ('5d15e74e-0b70-49b9-ae91-1c36c5d0db4c', generate_response(
            status_message='List not found',
            status_code=HTTPStatus.NOT_FOUND
        ), HTTPStatus.NOT_FOUND),
        (LIST_ID_NO_PHRASES, generate_response(
            status_message='There are no phrases attached to this list',
            status_code=HTTPStatus.FORBIDDEN
        ), HTTPStatus.FORBIDDEN)
    ])
    def test_create_session_failed(self, client, list_id,
                                   expected_response, status_code):
        actual_response = client.post(
            CREATE_SESSION_ENDPOINT.format(list_id=list_id)
        )

        assert actual_response.status_code == status_code
        assert actual_response.get_json() == expected_response

    @pytest.mark.parametrize('list_id, _filter, expected_response', [
        (LIST_ID_PHRASES, {}, generate_response([
            {
                'id': SESSION_ID_INCOMPLETE,
                'completed': False
            },
            {
                'id': SESSION_ID_COMPLETE,
                'completed': True
            }
        ])),
        (LIST_ID_PHRASES, {'completed': 'true'}, generate_response([
            {
                'id': SESSION_ID_COMPLETE,
                'completed': True
            }
        ])),
        (LIST_ID_PHRASES, {'completed': 'false'}, generate_response([
            {
                'id': SESSION_ID_INCOMPLETE,
                'completed': False
            },
        ])),
        (LIST_ID_NO_PHRASES, {}, generate_response([]))
    ])
    def test_get_sessions_200(self, client, list_id, _filter,
                              expected_response):
        actual_response = client.get(
            GET_SESSIONS_ENDPOINT.format(list_id=list_id),
            query_string=_filter
        )

        assert actual_response.status_code == HTTPStatus.OK
        assert actual_response.get_json() == expected_response

    @pytest.mark.parametrize('list_id, expected_response, status_code', [
        ('invalid_list_uuid', generate_response(
            status_message='Invalid UUID given: invalid_list_uuid',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        ('99851f68-d098-4d07-b7d8-5155bb73fef1', generate_response(
            status_message='List not found',
            status_code=HTTPStatus.NOT_FOUND
        ), HTTPStatus.NOT_FOUND)
    ])
    def test_get_sessions_failed(self, client, list_id, expected_response,
                                 status_code):
        actual_response = client.get(
            GET_SESSIONS_ENDPOINT.format(list_id=list_id)
        )

        assert actual_response.status_code == status_code
        assert actual_response.get_json() == expected_response

    @pytest.mark.parametrize('list_id, session_id, expected_response', [
        (LIST_ID_PHRASES, SESSION_ID_INCOMPLETE, generate_response({
            'id': SESSION_ID_INCOMPLETE,
            'completed': False,
            'phrases': [{
                'id': '21c8c615-57a2-4e41-a2b8-c1cd4ae540a0',
                'content': 'Call my family',
                'in_model': False,
                'template_id': None,
            }]
        })),
        (LIST_ID_PHRASES, SESSION_ID_COMPLETE, generate_response({
            'id': SESSION_ID_COMPLETE,
            'completed': True,
            'phrases': [{
                'id': '21c8c615-57a2-4e41-a2b8-c1cd4ae540a0',
                'content': 'Call my family',
                'in_model': False,
                'template_id': '729bbdf4-0ac6-4570-9f9e-2e717ac4e1fd'
            }]
        }))
    ])
    def test_get_session_200(self, client, list_id, session_id,
                             expected_response):
        actual_response = client.get(
            GET_SESSION_ENDPOINT.format(
                list_id=list_id,
                session_id=session_id
            )
        )

        assert actual_response.status_code == HTTPStatus.OK
        assert actual_response.get_json() == expected_response

    @pytest.mark.parametrize(
        'list_id, session_id, expected_response, status_code', [
            ('invalid_list_uuid', '8f2807e6-cfb8-4b2f-ba26-93a40ff10e8b',
             generate_response(
                 status_message='Invalid UUID given: invalid_list_uuid',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('8f2807e6-cfb8-4b2f-ba26-93a40ff10e8b', 'invalid_session_uuid',
             generate_response(
                 status_message='Invalid UUID given: invalid_session_uuid',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('8f2807e6-cfb8-4b2f-ba26-93a40ff10e8b',
             '7bb5e25b-48da-40b9-97f1-6247cf8dbd8f',
             generate_response(
                 status_message='List not found',
                 status_code=HTTPStatus.NOT_FOUND
             ), HTTPStatus.NOT_FOUND),
            (LIST_ID, '7bb5e25b-48da-40b9-97f1-6247cf8dbd8f',
             generate_response(
                 status_message='Session not found',
                 status_code=HTTPStatus.NOT_FOUND
             ), HTTPStatus.NOT_FOUND)
        ])
    def test_get_session_failed(self, client, list_id, session_id,
                                expected_response, status_code):
        actual_response = client.get(
            GET_SESSION_ENDPOINT.format(
                list_id=list_id,
                session_id=session_id
            )
        )

        assert actual_response.status_code == status_code
        assert actual_response.get_json() == expected_response
