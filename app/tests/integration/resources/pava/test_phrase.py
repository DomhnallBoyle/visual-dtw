from http import HTTPStatus

import pytest
from main.models import PAVAList
from main.utils.enums import ListStatus
from main.utils.validators import validate_uuid
from tests.integration.utils import generate_response

BASE_ENDPOINT = '/pava/api/v1/lists/{list_id}/phrases'
CREATE_PHRASE_ENDPOINT = BASE_ENDPOINT
GET_PHRASES_ENDPOINT = BASE_ENDPOINT
DELETE_PHRASE_ENDPOINT = BASE_ENDPOINT + '/{phrase_id}'

LIST_ID_PHRASES_1 = 'd000ae57-5576-46c8-aabe-8e47602bbf03'
LIST_ID_PHRASES_2 = '27a8dc84-c2f0-43a3-b04b-da01eb037fdc'
LIST_ID_NOT_FOUND = 'b55fda31-1891-4682-b037-93fd6ef2c453'
LIST_ID_WITH_PHRASE_TO_DELETE = LIST_ID_PHRASES_1
LIST_ID_WITH_ARCHIVED_PHRASE = 'fad5d74a-f6ff-42aa-a082-3dd385d285f6'
PHRASE_ID_TO_DELETE = '146a75a8-ac64-4cb7-9642-772edc4b1fcb'
PHRASE_ID_LEFTOVER = '21c8c615-57a2-4e41-a2b8-c1cd4ae540a0'
PHRASE_ID_NOT_FOUND = 'ba1cb796-cc55-4bc1-a61e-fc62a2815f1e'


class TestPhrase:

    @pytest.mark.parametrize('list_id, phrase_content', [
        (LIST_ID_PHRASES_2, 'Test phrase'),
        (LIST_ID_WITH_ARCHIVED_PHRASE, 'Archived Phrase')  # there is already a phrase named "Archived Phrase"
    ])
    def test_create_phrase_201(self, client, list_id, phrase_content):
        actual_response = client.post(
            CREATE_PHRASE_ENDPOINT.format(list_id=list_id),
            json={'content': phrase_content}
        )
        json_response = actual_response.get_json()

        assert actual_response.status_code == HTTPStatus.CREATED
        assert list(json_response.keys()) == ['response', 'status']

        create_phrase_response = json_response['response']
        assert list(create_phrase_response.keys()) == ['id', 'content', 'in_model']
        validate_uuid(create_phrase_response['id'])
        assert create_phrase_response['content'] == phrase_content
        assert create_phrase_response['in_model'] == False

        status_response = json_response['status']
        assert list(status_response.keys()) == ['message', 'code']
        assert status_response['message'] == HTTPStatus.CREATED.phrase
        assert status_response['code'] == HTTPStatus.CREATED

    @pytest.mark.parametrize('list_id, json, expected_response, status_code', [
        ('invalid_uuid', {'content': 'Test Phrase'},
         generate_response(
             status_message='Invalid UUID given: invalid_uuid',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        (LIST_ID_NOT_FOUND, {}, {
            'errors': {
                'content': '\'content\' is a required property'
            },
            'message': 'Input payload validation failed'
        }, HTTPStatus.BAD_REQUEST),
        (LIST_ID_NOT_FOUND, {'content': ''}, {
            'errors': {
                'content': '\'\' is too short'
            },
            'message': 'Input payload validation failed'
        }, HTTPStatus.BAD_REQUEST),
        (LIST_ID_NOT_FOUND, {'content': 'a' * 51}, {
            'errors': {
                'content': f'\'{"a" * 51}\' is too long'
            },
            'message': 'Input payload validation failed'
        }, HTTPStatus.BAD_REQUEST),
        (LIST_ID_NOT_FOUND, {'content': 'Test Phrase'},
         generate_response(
             status_message='List not found',
             status_code=HTTPStatus.NOT_FOUND),
         HTTPStatus.NOT_FOUND),
        (LIST_ID_PHRASES_2, {'content': 'Test Phrase'},
         generate_response(
             status_message='Invalid name: \'Test Phrase\' is already in the list',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        (LIST_ID_PHRASES_2, {'content': 'TEST PHRASE'},
         generate_response(
             status_message='Invalid name: \'TEST PHRASE\' is already in the list',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
    ])
    def test_create_phrase_failed(self, client, list_id, json,
                                  expected_response, status_code):
        actual_response = client.post(
            CREATE_PHRASE_ENDPOINT.format(list_id=list_id),
            json=json
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    def test_create_phrase_403_list_updating(self, client):
        expected_response = generate_response(
            status_message='Invalid operation: List not ready',
            status_code=HTTPStatus.FORBIDDEN
        )

        # simulate the session selection algorithm updating
        PAVAList.update(id=LIST_ID_PHRASES_1, status=ListStatus.UPDATING)

        actual_response = client.post(
            CREATE_PHRASE_ENDPOINT.format(list_id=LIST_ID_PHRASES_1),
            json={'content': 'My new phrase'}
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.FORBIDDEN

        # reset status of list
        PAVAList.update(id=LIST_ID_PHRASES_1, status=ListStatus.READY)

    @pytest.mark.parametrize('list_id, expected_response, status_code', [
        (LIST_ID_PHRASES_1, generate_response([
            {
                'id': PHRASE_ID_TO_DELETE,
                'content': 'Turn on the TV',
                'in_model': False,
            },
            {
                'id': PHRASE_ID_LEFTOVER,
                'content': 'Call my family',
                'in_model': False,
            }
        ]), HTTPStatus.OK),
        ('invalid_uuid', generate_response(
            status_message='Invalid UUID given: invalid_uuid',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST)
    ])
    def test_get_list_phrases(self, client, list_id, expected_response,
                              status_code):
        actual_response = client.get(
            GET_PHRASES_ENDPOINT.format(list_id=list_id),
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    def test_delete_phrase_200(self, client):
        expected_response = generate_response()

        actual_response = client.delete(DELETE_PHRASE_ENDPOINT.format(
            list_id=LIST_ID_WITH_PHRASE_TO_DELETE,
            phrase_id=PHRASE_ID_TO_DELETE
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        # check if phrase is gone from list
        expected_response = generate_response([{
            'id': PHRASE_ID_LEFTOVER,
            'content': 'Call my family',
            'in_model': False
        }])

        actual_response = client.get(GET_PHRASES_ENDPOINT.format(
            list_id=LIST_ID_WITH_PHRASE_TO_DELETE
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

    @pytest.mark.parametrize(
        'list_id, phrase_id, expected_response, status_code', [
            ('invalid_list_id', PHRASE_ID_TO_DELETE,
             generate_response(
                 status_message='Invalid UUID given: invalid_list_id',
                 status_code=HTTPStatus.BAD_REQUEST),
             HTTPStatus.BAD_REQUEST),
            (LIST_ID_WITH_PHRASE_TO_DELETE, 'invalid_phrase_id',
             generate_response(
                 status_message='Invalid UUID given: invalid_phrase_id',
                 status_code=HTTPStatus.BAD_REQUEST),
             HTTPStatus.BAD_REQUEST),
            (LIST_ID_NOT_FOUND, PHRASE_ID_TO_DELETE,
             generate_response(
                 status_message='List not found',
                 status_code=HTTPStatus.NOT_FOUND),
             HTTPStatus.NOT_FOUND),
            (LIST_ID_WITH_PHRASE_TO_DELETE, PHRASE_ID_NOT_FOUND,
             generate_response(
                 status_message='Phrase not found',
                 status_code=HTTPStatus.NOT_FOUND),
             HTTPStatus.NOT_FOUND),
        ])
    def test_delete_phrase_failed(self, client, list_id, phrase_id,
                                  expected_response, status_code):
        actual_response = client.delete(DELETE_PHRASE_ENDPOINT.format(
            list_id=list_id, phrase_id=phrase_id
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    def test_delete_phrase_403_list_updating(self, client):
        expected_response = generate_response(
            status_message='Invalid operation: List not ready',
            status_code=HTTPStatus.FORBIDDEN
        )

        # simulate the session selection algorithm updating
        PAVAList.update(id=LIST_ID_PHRASES_1, status=ListStatus.UPDATING)

        actual_response = client.delete(
            DELETE_PHRASE_ENDPOINT.format(
                list_id=LIST_ID_WITH_PHRASE_TO_DELETE,
                phrase_id=PHRASE_ID_LEFTOVER
            ),
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.FORBIDDEN

        # reset status of list
        PAVAList.update(id=LIST_ID_PHRASES_1, status=ListStatus.READY)
