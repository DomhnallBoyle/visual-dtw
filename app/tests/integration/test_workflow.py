"""
These tests should be ran in order to mimic an entire workflow through the
system from creating users to transcribing and updating user phrase templates
"""
import io
import os
from http import HTTPStatus

from main import configuration
from main.utils.io import read_json_file
from main.utils.validators import validate_uuid

BASE_ENDPOINT = '/pava/api/v1/'
CREATE_USER_ENDPOINT = BASE_ENDPOINT + 'users'
GET_USER_ENDPOINT = CREATE_USER_ENDPOINT + '/{user_id}'
GET_USER_LISTS_ENDPOINT = GET_USER_ENDPOINT + '/lists'
GET_USER_LIST_ENDPOINT = GET_USER_LISTS_ENDPOINT + '/{list_id}'
TRANSCRIBE_ENDPOINT = BASE_ENDPOINT + 'lists/{list_id}/transcribe/video'
LABEL_TEMPLATE_ENDPOINT = \
    BASE_ENDPOINT + 'phrases/{phrase_id}/templates/{template_id}'

user_id = None
list_id = None
phrase_id = None
template_id = None


def transcribe_video(client, video_name, expected_response):
    video_path = os.path.join(configuration.DATA_PATH, 'videos', video_name)

    # transcribe the video against the default list
    with open(video_path, 'rb') as f:
        data = {
            'file': (io.BytesIO(f.read()), video_name)
        }
        actual_response = client.post(
            TRANSCRIBE_ENDPOINT.format(list_id=list_id),
            data=data
        )

    if actual_response.status_code == 200:
        keys = ['template_id', 'predictions']
        assert len(actual_response) == 2
        assert all(k in actual_response for k in keys)
        validate_uuid(actual_response['template_id'])

        for actual_prediction, expected_prediction in zip(
                actual_response['predictions'],
                expected_response
        ):
            validate_uuid(actual_prediction['id'])
            assert actual_prediction['label'] == expected_prediction['label']
            assert actual_prediction['accuracy'] == \
                expected_prediction['accuracy']

        return actual_response
    else:
        pass


def test_create_user(client):
    actual_response = client.post(CREATE_USER_ENDPOINT)
    json_response = actual_response.get_json()

    assert actual_response.status_code == HTTPStatus.CREATED
    assert list(json_response.keys()) == ['response', 'status']

    create_user_response = json_response['response']
    assert type(create_user_response) == dict
    assert list(create_user_response.keys()) == ['id', 'config_id']
    validate_uuid(create_user_response['id'],
                  create_user_response['config_id'])

    status_response = json_response['status']
    assert type(status_response) == dict
    assert list(status_response.keys()) == ['message', 'code']
    assert status_response['message'] == HTTPStatus.CREATED.phrase
    assert status_response['code'] == HTTPStatus.CREATED

    global user_id
    user_id = create_user_response['id']


def test_get_user(client):
    actual_response = client.get(GET_USER_ENDPOINT.format(user_id=user_id))
    json_response = actual_response.get_json()

    assert actual_response.status_code == HTTPStatus.OK
    assert list(json_response.keys()) == ['response', 'status']

    get_user_response = json_response['response']
    assert type(get_user_response) == dict
    assert list(get_user_response.keys()) == ['id', 'config_id']
    validate_uuid(get_user_response['id'], get_user_response['config_id'])

    status_response = json_response['status']
    assert type(status_response) == dict
    assert list(status_response.keys()) == ['message', 'code']
    assert status_response['message'] == HTTPStatus.OK.phrase
    assert status_response['code'] == HTTPStatus.OK


def test_get_user_lists(client):
    actual_response = client.get(
        GET_USER_LISTS_ENDPOINT.format(user_id=user_id)
    )
    json_response = actual_response.get_json()

    assert actual_response.status_code == HTTPStatus.OK
    assert list(json_response.keys()) == ['response', 'status']

    get_user_lists_response = json_response['response']
    assert type(get_user_lists_response) == list
    assert len(get_user_lists_response) == 2  # 2 default lists

    # check the contents of the phrases
    # hack - can't guarantee same order every time
    if get_user_lists_response[0]['name'] != \
            configuration.DEFAULT_PAVA_LIST_NAME:
        temp = get_user_lists_response[0]
        get_user_lists_response[0] = get_user_lists_response[1]
        get_user_lists_response[1] = temp

    for default_list, default_phrases in zip(get_user_lists_response, [
        read_json_file(configuration.PHRASES_PATH)['PAVA'],
        read_json_file(configuration.PHRASES_PATH)['PAVA-SUB-DEFAULT']
    ]):
        assert len(default_list['phrases']) == len(default_phrases)
        list_phrase_contents = [phrase['content']
                                for phrase in default_list['phrases']]
        for phrase_content in default_phrases.values():
            assert phrase_content in list_phrase_contents

    global list_id
    list_id = get_user_lists_response[0]['id']

    status_response = json_response['status']
    assert type(status_response) == dict
    assert list(status_response.keys()) == ['message', 'code']
    assert status_response['message'] == HTTPStatus.OK.phrase
    assert status_response['code'] == HTTPStatus.OK


def test_get_user_list(client):
    actual_response = client.get(
        GET_USER_LIST_ENDPOINT.format(user_id=user_id, list_id=list_id)
    )
    json_response = actual_response.get_json()

    assert actual_response.status_code == HTTPStatus.OK
    assert list(json_response.keys()) == ['response', 'status']

    get_user_list_response = json_response['response']
    default_lst = get_user_list_response

    assert type(default_lst) == dict
    assert default_lst['name'] == configuration.DEFAULT_PAVA_LIST_NAME
    assert len(default_lst['phrases']) == 23  # 23 default phrases

    # check the contents of the phrases
    default_phrases = \
        read_json_file(configuration.PHRASES_PATH)['PAVA']

    list_phrase_contents = []
    for phrase in default_lst['phrases']:
        list_phrase_contents.append(phrase['content'])

    for phrase_content in default_phrases.values():
        assert phrase_content in list_phrase_contents

    status_response = json_response['status']
    assert type(status_response) == dict
    assert list(status_response.keys()) == ['message', 'code']
    assert status_response['message'] == HTTPStatus.OK.phrase
    assert status_response['code'] == HTTPStatus.OK
