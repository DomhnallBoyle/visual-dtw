from http import HTTPStatus

import pytest
from main.utils.validators import validate_uuid
from tests.integration.utils import generate_response

BASE_ENDPOINT = '/pava/api/v1/users'
CREATE_USER_ENDPOINT = BASE_ENDPOINT
GET_USER_ENDPOINT = BASE_ENDPOINT + '/{user_id}'

USER_ID = '6ca0349f-e820-40c1-93f9-0a6aff85c62d'
CONFIG_ID = '396ce0ec-a4e2-44b8-a615-0f5a3b6d74e9'


class TestUser:

    def test_create_user_201(self, client):
        actual_response = client.post(CREATE_USER_ENDPOINT)
        json_response = actual_response.get_json()

        assert actual_response.status_code == HTTPStatus.CREATED
        assert list(json_response.keys()) == ['response', 'status']

        create_user_response = json_response['response']
        assert list(create_user_response.keys()) == ['id', 'config_id']
        validate_uuid(create_user_response['id'],
                      create_user_response['config_id'])

        status_response = json_response['status']
        assert list(status_response.keys()) == ['message', 'code']
        assert status_response['message'] == HTTPStatus.CREATED.phrase
        assert status_response['code'] == HTTPStatus.CREATED

    @pytest.mark.parametrize('user_id, expected_response, headers', [
        (USER_ID, generate_response({
            'id': USER_ID,
            'config_id': CONFIG_ID
        }), {}),
        (USER_ID, generate_response({
            'id': USER_ID
        }, include_status=False), {'X-Fields': 'response{id}'})
    ])
    def test_get_user_200(self, client, user_id, expected_response, headers):
        actual_response = client.get(
            GET_USER_ENDPOINT.format(user_id=user_id),
            headers=headers
        )

        assert actual_response.status_code == HTTPStatus.OK
        json_response = actual_response.get_json()['response']
        assert json_response.keys() == expected_response['response'].keys()
        validate_uuid(json_response['id'])
        assert json_response['id'] == USER_ID
        if json_response.get('config_id'):
            validate_uuid(json_response['config_id'])

    @pytest.mark.parametrize('user_id, expected_response, status_code', [
        ('invalid_uuid',
         generate_response(
             status_message='Invalid UUID given: invalid_uuid',
             status_code=HTTPStatus.BAD_REQUEST
         ), HTTPStatus.BAD_REQUEST),
        ('8e41a8f5-3917-467c-92e8-784acd6bd0e7',
         generate_response(
             status_message='User not found',
             status_code=HTTPStatus.NOT_FOUND
         ), HTTPStatus.NOT_FOUND)
    ])
    def test_get_user_failed(self, client, user_id, expected_response,
                             status_code):
        actual_response = client.get(GET_USER_ENDPOINT.format(
            user_id=user_id))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code
