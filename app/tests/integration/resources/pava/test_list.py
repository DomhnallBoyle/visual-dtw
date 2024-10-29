import datetime
from http import HTTPStatus

import pytest
from main import cache
from main.models import PAVAList, PAVAModel, PAVAPhrase
from main.utils.db import db_session
from main.utils.validators import validate_uuid
from tests.integration.resources.pava.test_record import RECORD_ENDPOINT, VIDEO_PATH
from tests.integration.utils import construct_request_video_data, generate_response

BASE_ENDPOINT = '/pava/api/v1/users/{user_id}/lists'
CREATE_LIST_ENDPOINT = BASE_ENDPOINT
GET_LISTS_ENDPOINT = BASE_ENDPOINT
GET_LIST_ENDPOINT = BASE_ENDPOINT + '/{list_id}'
DELETE_LIST_ENDPOINT = GET_LIST_ENDPOINT

USER_ID_WITH_LISTS = 'f3149387-0912-4044-a553-e34c69379e3b'
USER_ID_1_WITHOUT_LISTS = '1ed4abdc-6b8c-4753-b560-3a7dfa5189a1'
USER_ID_2_WITHOUT_LISTS = '4ea2eb48-b1e8-4d7f-ac57-1587912ca335'
USER_ID_WITH_ARCHIVED_LIST = '2ca427e6-58f5-46b4-9d4c-94617b4b743b'
NON_USER_ID = 'a768de2f-59de-4c58-be7a-2d5ef6afd3b7'
USER_ID_WITH_LIST_TO_DELETE = '8960c8b0-45e4-40ca-9a6b-9b639abdf112'
LIST_ID = '27a8dc84-c2f0-43a3-b04b-da01eb037fdc'
LIST_ID_TO_DELETE = 'd58c05df-538a-4ee0-b2ea-ded0765fe9cb'
USER_ID_WITH_MODEL = '1b7296b8-a7ac-4c9c-a7ed-bb8c6c89795c'
LIST_ID_WITH_MODEL = '23a0a428-1250-47ee-9b60-c7f40e190104'
MODEL_ID = '40e91941-086a-4019-b914-681648342a21'


class TestList:

    @pytest.mark.parametrize('user_id, list_name', [
        (USER_ID_1_WITHOUT_LISTS, 'Test list'),
        (USER_ID_WITH_ARCHIVED_LIST, 'Archived List')  # there is already a list named 'Archived List'
    ])
    def test_create_list_201(self, client, user_id, list_name):
        actual_response = client.post(
            CREATE_LIST_ENDPOINT.format(user_id=user_id),
            json={'name': list_name}
        )
        json_response = actual_response.get_json()

        assert actual_response.status_code == HTTPStatus.CREATED
        assert list(json_response.keys()) == ['response', 'status']

        create_list_response = json_response['response']

        list_uuid = create_list_response['id']
        validate_uuid(list_uuid)
        assert create_list_response == {
            'id': list_uuid,
            'name': list_name,
            'default': False,
            'status': 'READY',
            'num_phrases': 0,
            'phrases': [],
            'num_user_sessions': 0,
            'current_model': None,
            'models': [],
            'last_updated': None
        }

        status_response = json_response['status']
        assert list(status_response.keys()) == ['message', 'code']
        assert status_response['message'] == 'Created'
        assert status_response['code'] == HTTPStatus.CREATED

        # make sure NOTA phrase created with the newly created list
        # if it doesn't exist, it will raise an exception
        with db_session() as s:
            PAVAPhrase.get(
                s, filter=(
                    (PAVAPhrase.list_id == list_uuid)
                    & (PAVAPhrase.content == 'None of the above')),
                first=True
            )

            lst = PAVAList.get(s, filter=(PAVAList.id == list_uuid),
                               first=True)
            assert not lst.has_added_phrases

    @pytest.mark.parametrize('user_id, json, expected_response, status_code', [
        ('invalid_uuid', {'name': 'Test list'},
         generate_response(
             status_message='Invalid UUID given: invalid_uuid',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        (USER_ID_WITH_LISTS, {}, {
            'errors': {
                'name': '\'name\' is a required property'
            },
            'message': 'Input payload validation failed'
        }, HTTPStatus.BAD_REQUEST),
        (USER_ID_WITH_LISTS, {'name': ''}, {
            'errors': {
                'name': '\'\' is too short'
            },
            'message': 'Input payload validation failed'
        }, HTTPStatus.BAD_REQUEST),
        (USER_ID_WITH_LISTS, {'name': 'a' * 51}, {
            'errors': {
                'name': f'\'{"a" * 51}\' is too long'
            },
            'message': 'Input payload validation failed'
        }, HTTPStatus.BAD_REQUEST),
        (USER_ID_WITH_LISTS.replace('a', 'b'), {'name': 'Test list'},
         generate_response(
             status_message='User not found',
             status_code=HTTPStatus.NOT_FOUND),
         HTTPStatus.NOT_FOUND),
        (USER_ID_WITH_LISTS, {'name': 'Default'},
         generate_response(
             status_message='Invalid name: \'Default\' cannot be used',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        (USER_ID_WITH_LISTS, {'name': 'Default sub-list'},
         generate_response(
             status_message='Invalid name: \'Default sub-list\' cannot be used',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        (USER_ID_1_WITHOUT_LISTS, {'name': 'Test List'},
         generate_response(
             status_message='Invalid name: \'Test List\' is already being used',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        (USER_ID_WITH_LISTS, {'name': 'DEFAULT SUB-LIST'},
         generate_response(
             status_message='Invalid name: \'DEFAULT SUB-LIST\' cannot be used',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
    ])
    def test_create_list_failed(self, client, user_id, json,
                                expected_response, status_code):
        actual_response = client.post(
            CREATE_LIST_ENDPOINT.format(user_id=user_id),
            json=json
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    @pytest.mark.parametrize('user_id, expected_response, status_code', [
        (USER_ID_WITH_LISTS, generate_response([{
            'id': LIST_ID,
            'name': 'Morning List',
            'default': False,
            'status': 'READY',
            'num_phrases': 1,
            'phrases': [
                {
                    'id': '2d1e2086-a22a-4a7a-abc2-c8fe48bf5ee9',
                    'content': 'Turn on the light',
                    'in_model': False
                }
            ],
            'num_user_sessions': 0,
            'current_model': None,
            'models': [],
            'last_updated': None
        }]), HTTPStatus.OK),
        (USER_ID_2_WITHOUT_LISTS, generate_response([]), HTTPStatus.OK),
        (NON_USER_ID, generate_response(
            status_message='User not found',
            status_code=HTTPStatus.NOT_FOUND),
         HTTPStatus.NOT_FOUND),
        ('invalid_uuid', generate_response(
            status_message='Invalid UUID given: invalid_uuid',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST)
    ])
    def test_get_user_lists(self, client, user_id, expected_response,
                            status_code):
        actual_response = client.get(
            GET_LISTS_ENDPOINT.format(user_id=user_id))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    @pytest.mark.parametrize(
        'user_id, list_id, expected_response, status_code', [
            (USER_ID_WITH_LISTS, LIST_ID, generate_response({
                'id': LIST_ID,
                'name': 'Morning List',
                'default': False,
                'status': 'READY',
                'num_phrases': 1,
                'phrases': [
                    {
                        'id': '2d1e2086-a22a-4a7a-abc2-c8fe48bf5ee9',
                        'content': 'Turn on the light',
                        'in_model': False
                    }
                ],
                'num_user_sessions': 0,
                'current_model': None,
                'models': [],
                'last_updated': None
            }), HTTPStatus.OK),
            ('invalid_user_uuid', '5d36c21c-7c47-4c74-a86b-3b357cded3ff',
             generate_response(
                 status_message='Invalid UUID given: invalid_user_uuid',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('a6bf8a9e-e4a3-4232-b902-2d00caa885fe', 'invalid_list_uuid',
             generate_response(
                 status_message='Invalid UUID given: invalid_list_uuid',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('a6bf8a9e-e4a3-4232-b902-2d00caa885fe',
             'f8580c31-a23e-4198-8e6b-c0e6bcda98da',
             generate_response(
                 status_message='List not found',
                 status_code=HTTPStatus.NOT_FOUND
             ), HTTPStatus.NOT_FOUND)
        ])
    def test_get_user_list(self, client, user_id, list_id,
                           expected_response, status_code):
        actual_response = client.get(GET_LIST_ENDPOINT.format(user_id=user_id,
                                                              list_id=list_id))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    def test_get_user_list_with_models(self, client):
        # set model id (cyclic dependency between list and model in fixtures)
        PAVAList.update(id=LIST_ID_WITH_MODEL, current_model_id=MODEL_ID)

        with db_session() as s:
            lst = PAVAList.get(s, filter=(PAVAList.id == LIST_ID_WITH_MODEL), first=True)
            session_ids = [s.id for s in lst.sessions]
            assert lst.has_added_phrases

        expected_response = generate_response({
            'id': LIST_ID_WITH_MODEL,
            'name': 'List with model',
            'default': False,
            'status': 'READY',
            'num_phrases': 2,
            'phrases': [
                {
                    'id': 'cef0f766-5245-4bcd-90d4-8c456d6de4cc',
                    'content': 'Can I use the toilet?',
                    'in_model': True
                },
                {
                    'id': 'd29d9655-1285-4456-bc55-2c96b3ef5d29',
                    'content': 'I feel unwell',
                    'in_model': True
                }
            ],
            'num_user_sessions': 5,
            'current_model': {
                'id': MODEL_ID,
                'num_sessions': 3,
                'phrase_coverage': 1
            },
            'models': [
                {
                    'id': 'de0bafdb-4564-43c8-8f68-b666afb9d8d4',
                    'date_created': '2020-09-17T11:09:35',
                    'accuracies': []
                },
                {
                    'id': MODEL_ID,
                    'date_created': '2020-09-17T12:10:45',
                    'accuracies': [
                        {
                            'accuracy': 0.77,
                            'date_created': '2020-09-18T15:52:32',
                            'num_test_templates': 50
                        },
                        {
                            'accuracy': 0.82,
                            'date_created': '2020-09-19T16:12:34',
                            'num_test_templates': 75
                        }
                    ]
                }
            ],
            'last_updated': '2020-09-17T12:10:45'
        })

        actual_response = client.get(GET_LIST_ENDPOINT.format(
            user_id=USER_ID_WITH_MODEL, list_id=LIST_ID_WITH_MODEL)
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        # now add a phrase
        new_phrase = PAVAPhrase.create(list_id=LIST_ID_WITH_MODEL, content='Yes please')

        expected_response['response']['num_phrases'] += 1
        expected_response['response']['phrases'].append({
            'id': new_phrase.str_id,
            'content': new_phrase.content,
            'in_model': False
        })
        expected_response['response']['current_model']['phrase_coverage'] = 0.67

        cache.clear()  # it would be 100% phrase coverage unless you clear cache of phrase counts
        actual_response = client.get(GET_LIST_ENDPOINT.format(
            user_id=USER_ID_WITH_MODEL, list_id=LIST_ID_WITH_MODEL)
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        # now provide recordings for the newly added phrase i.e. phrase coverage should still be 67% because
        # phrases aren't in the model yet
        for session_id in session_ids:
            record_response = client.post(
                RECORD_ENDPOINT.format(session_id=session_id, phrase_id=new_phrase.id),
                data=construct_request_video_data(VIDEO_PATH)
            )
            assert record_response.status_code == HTTPStatus.CREATED

        cache.clear()
        actual_response = client.get(GET_LIST_ENDPOINT.format(
            user_id=USER_ID_WITH_MODEL, list_id=LIST_ID_WITH_MODEL
        ))
        assert actual_response.get_json() == expected_response  # should still be same response
        assert actual_response.status_code == HTTPStatus.OK

        # now build model i.e. in_model should be True
        # simulate model being built - just update creation date of model for now
        # i.e. if model creation date > phrase creation dates - phrases are in model
        current_time = datetime.datetime.now()
        PAVAModel.update(id=lst.current_model_id, date_created=current_time)

        expected_response['response']['last_updated'] = current_time.isoformat()  # includes 'T' between date and time
        expected_response['response']['models'][-1]['date_created'] = current_time.isoformat()
        expected_response['response']['current_model']['phrase_coverage'] = 1
        expected_response['response']['phrases'][-1]['in_model'] = True
        cache.clear()
        actual_response = client.get(GET_LIST_ENDPOINT.format(
            user_id=USER_ID_WITH_MODEL, list_id=LIST_ID_WITH_MODEL
        ))
        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

    def test_delete_list_200(self, client):
        expected_response = generate_response()

        actual_response = client.delete(DELETE_LIST_ENDPOINT.format(
            user_id=USER_ID_WITH_LIST_TO_DELETE, list_id=LIST_ID_TO_DELETE
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        # now get user lists again - should be empty
        expected_response = generate_response(
            response=[],
            status_message=HTTPStatus.OK.phrase,
            status_code=HTTPStatus.OK
        )

        actual_response = client.get(GET_LISTS_ENDPOINT.format(
            user_id=USER_ID_WITH_LIST_TO_DELETE
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        # try to get the individual list - should return not found
        expected_response = generate_response(
            status_message='List not found',
            status_code=HTTPStatus.NOT_FOUND
        )

        actual_response = client.get(GET_LIST_ENDPOINT.format(
            user_id=USER_ID_WITH_LIST_TO_DELETE, list_id=LIST_ID_TO_DELETE
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.NOT_FOUND

    @pytest.mark.parametrize(
        'user_id, list_id, expected_response, status_code', [
            ('invalid_user_id', LIST_ID, generate_response(
                status_message='Invalid UUID given: invalid_user_id',
                status_code=HTTPStatus.BAD_REQUEST
            ), HTTPStatus.BAD_REQUEST),
            (NON_USER_ID, 'invalid_list_id', generate_response(
                status_message='Invalid UUID given: invalid_list_id',
                status_code=HTTPStatus.BAD_REQUEST
            ), HTTPStatus.BAD_REQUEST),
            (NON_USER_ID, LIST_ID, generate_response(
                status_message='User not found',
                status_code=HTTPStatus.NOT_FOUND
            ), HTTPStatus.NOT_FOUND),
            (USER_ID_1_WITHOUT_LISTS, NON_USER_ID, generate_response(
                status_message='List not found',
                status_code=HTTPStatus.NOT_FOUND
            ), HTTPStatus.NOT_FOUND)
        ])
    def test_delete_list_failed(self, client, user_id, list_id,
                                expected_response, status_code):
        actual_response = client.delete(DELETE_LIST_ENDPOINT.format(
            user_id=user_id, list_id=list_id
        ))

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code
