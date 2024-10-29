"""Runs the process of creating a list, phrases, recording sessions and building models
Uses a webcam for the process
"""
import argparse
import json
import requests
from http import HTTPStatus

from main.utils.cam import Cam

HOST = None

USERS_URL = '/users'
CREATE_USER_URL = USERS_URL
GET_USER_URL = USERS_URL + '/{user_id}'

LISTS_URL = USERS_URL + '/{user_id}/lists'
CREATE_LIST_URL = LISTS_URL
GET_LIST_URL = LISTS_URL + '/{list_id}'
GET_LISTS_URL = LISTS_URL
DELETE_LIST_URL = GET_LIST_URL
BUILD_MODEL_URL = LISTS_URL + '/{list_id}/model/build'

PHRASES_URL = '/lists/{list_id}/phrases'
CREATE_PHRASE_URL = PHRASES_URL
GET_PHRASES_URL = PHRASES_URL
DELETE_PHRASE_URL = PHRASES_URL + '/{phrase_id}'

SESSIONS_URL = '/lists/{list_id}/sessions'
CREATE_SESSION_URL = SESSIONS_URL
GET_SESSION_URL = SESSIONS_URL + '/{session_id}'
GET_SESSIONS_URL = SESSIONS_URL
RECORD_SESSION_PHRASE_URL = '/sessions/{session_id}/phrases/{phrase_id}/record'

TRANSCRIBE_VIDEO_URL = '/lists/{list_id}/transcribe/video'
UPDATE_TEMPLATE_URL = '/phrases/{phrase_id}/templates/{template_id}'

NUM_SESSIONS = 5


class LoopBreak(Exception):
    pass


def construct_options(options_list, include_back=True):
    s = f'\n********************************\n' \
        f'Select from:\n'
    for i, option in enumerate(options_list):
        s += f'[{i+1}] {option}\n'

    if include_back:
        s += f'[{len(options_list)+1}] Back\n'

    return s


def get_option(s, num_options):
    print(s)
    option = input('Selection: ')
    option = int(option)
    if not 1 <= option <= num_options:
        raise ValueError

    return option


def loop_break():
    raise LoopBreak


def pprint_json(j):
    print(json.dumps(j, indent=4, sort_keys=True))


def request(f, url, _json=None, files=None):
    url = f'{HOST}/pava/api/v1{url}'
    response = f(url, json=_json, files=files)
    if response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR:
        print(f'Something went wrong: {response.status_code} - {response.content}')
        exit()

    response = response.json()
    pprint_json(response)

    return response


def option_loop(s, f):
    while True:
        try:
            option = get_option(s, num_options=len(f))
        except ValueError:
            print('Invalid Option')
            return
        f[option]()


def user():
    s = construct_options(['Create User', 'Get User'])

    def _create_user():
        request(requests.post, CREATE_USER_URL)

    def _get_user():
        user_id = input('User ID: ')
        request(requests.get, GET_USER_URL.format(user_id=user_id))

    try:
        option_loop(s, {
            1: _create_user,
            2: _get_user,
            3: loop_break
        })
    except (LoopBreak, KeyboardInterrupt):
        return


def lst():
    s = construct_options(['Create List', 'Get List', 'Get User Lists', 'Delete List', 'Build List Model'])

    def _create_list():
        user_id = input('User ID: ')
        list_name = input('List Name: ')
        request(requests.post, CREATE_LIST_URL.format(user_id=user_id), _json={'name': list_name})

    def _get_list():
        user_id = input('User ID: ')
        list_id = input('List ID: ')
        request(requests.get, GET_LIST_URL.format(user_id=user_id, list_id=list_id))

    def _get_user_lists():
        user_id = input('User ID: ')
        request(requests.get, GET_LISTS_URL.format(user_id=user_id))

    def _delete_list():
        user_id = input('User ID: ')
        list_id = input('List ID: ')
        request(requests.delete, DELETE_LIST_URL.format(user_id=user_id, list_id=list_id))

    def _build_list_model():
        user_id = input('User ID: ')
        list_id = input('List ID: ')
        request(requests.post, BUILD_MODEL_URL.format(user_id=user_id, list_id=list_id))

    try:
        option_loop(s, {
            1: _create_list,
            2: _get_list,
            3: _get_user_lists,
            4: _delete_list,
            5: _build_list_model,
            6: loop_break
        })
    except (LoopBreak, KeyboardInterrupt):
        return


def phrase():
    s = construct_options(['Create Phrase', 'Get List Phrases', 'Delete Phrase'])

    def _create_phrase():
        list_id = input('List ID: ')
        _phrase = input('Phrase: ')
        request(requests.post, CREATE_PHRASE_URL.format(list_id=list_id), _json={'content': _phrase})

    def _get_list_phrases():
        list_id = input('List ID: ')
        request(requests.get, GET_PHRASES_URL.format(list_id=list_id))

    def _delete_phrase():
        list_id = input('List ID: ')
        phrase_id = input('Phrase ID: ')
        request(requests.delete, DELETE_PHRASE_URL.format(list_id=list_id, phrase_id=phrase_id))

    try:
        option_loop(s, {
            1: _create_phrase,
            2: _get_list_phrases,
            3: _delete_phrase,
            4: loop_break
        })
    except (LoopBreak, KeyboardInterrupt):
        return


def session():
    s = construct_options(['Create Session', 'Get Session', 'Get List Sessions', 'Record Session Phrase'])

    def _create_session():
        list_id = input('List ID: ')
        request(requests.post, CREATE_SESSION_URL.format(list_id=list_id))

    def _get_session():
        list_id = input('List ID: ')
        session_id = input('Session ID: ')
        request(requests.get, GET_SESSION_URL.format(list_id=list_id, session_id=session_id))

    def _get_list_sessions():
        list_id = input('List ID: ')
        request(requests.get, GET_SESSIONS_URL.format(list_id=list_id))

    def _record_session_phrase():
        session_id = input('Session ID: ')
        phrase_id = input('Phrase ID: ')
        web_cam = Cam(countdown=True, debug=True)
        save_path = web_cam.record()
        with open(save_path, 'rb') as f:
            request(requests.post, RECORD_SESSION_PHRASE_URL.format(session_id=session_id, phrase_id=phrase_id),
                    files={'file': (save_path, f.read())})

    try:
        option_loop(s, {
            1: _create_session,
            2: _get_session,
            3: _get_list_sessions,
            4: _record_session_phrase,
            5: loop_break
        })
    except (LoopBreak, KeyboardInterrupt):
        return


def transcribe():
    s = construct_options(['Transcribe Video', 'Confirm Response'])

    def _transcribe_video():
        list_id = input('List ID: ')
        web_cam = Cam(countdown=True, debug=True)
        save_path = web_cam.record()
        with open(save_path, 'rb') as f:
            request(requests.post, TRANSCRIBE_VIDEO_URL.format(list_id=list_id),
                    files={'file': (save_path, f.read())})

    def _confirm_response():
        phrase_id = input('Phrase ID: ')
        template_id = input('Template ID: ')
        request(requests.put, UPDATE_TEMPLATE_URL.format(phrase_id=phrase_id, template_id=template_id))

    try:
        option_loop(s, {
            1: _transcribe_video,
            2: _confirm_response,
            3: loop_break
        })
    except (LoopBreak, KeyboardInterrupt):
        return


def main(args):
    global HOST
    HOST = args.host

    s = construct_options(['User', 'List', 'Phrase', 'Session', 'Transcribe', 'Quit'],
                          include_back=False)
    try:
        option_loop(s, {
            1: user,
            2: lst,
            3: phrase,
            4: session,
            5: transcribe,
            6: loop_break
        })
    except (LoopBreak, KeyboardInterrupt):
        pass

    print('\nBye!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', default='http://127.0.0.1:5000')

    main(parser.parse_args())
