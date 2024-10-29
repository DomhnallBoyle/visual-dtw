import io
import os
from datetime import datetime
from http import HTTPStatus

import pytest
from main import cache
from main import configuration as config
from main.models import PAVAList, PAVAModel, PAVAPhrase
from main.utils.db import db_session
from main.utils.enums import ListStatus
from tests.integration.resources.pava.test_record import RECORD_ENDPOINT
from tests.integration.utils import are_signals_cached, \
    construct_request_video_data, create_list_model, generate_response

BASE_ENDPOINT = '/pava/api/v1/lists/{list_id}/transcribe'
VIDEO_ENDPOINT = f'{BASE_ENDPOINT}/video'

USER_ID = '49540448-e1c2-46e9-9bc8-247eb01aac9b'
MOVE_ME_VIDEO_PATH = os.path.join(config.VIDEOS_PATH, 'move_me.mp4')
I_NEED_SUCTION_VIDEO_PATH = os.path.join(config.VIDEOS_PATH,
                                         'i_need_suction.mp4')
I_DONT_WANT_THAT_TREATMENT_VIDEO_PATH = \
    os.path.join(config.VIDEOS_PATH, 'i_dont_want_that_treatment.mp4')


def get_list_id():
    with db_session() as s:
        return PAVAList.get(
            s,
            query=(PAVAList.id,),
            filter=((PAVAList.user_id == USER_ID)
                    & (PAVAList.name == config.DEFAULT_PAVA_LIST_NAME)),
            first=True
        )[0]


class TestTranscribe:

    @pytest.mark.parametrize('expected_response', [
        (generate_response({
             'predictions': [
                 {'label': 'Move me', 'accuracy': 0.78},
                 {'label': 'I\'m hungry', 'accuracy': 0.12},
                 {'label': 'Call my family', 'accuracy': 0.05},
                 {'label': 'None of the above', 'accuracy': 0.0}
             ]
         }))
    ])
    def test_transcribe_video_200_default_list(self, client,
                                               expected_response):
        # ignore any ids we don't know beforehand
        headers = {
            'X-Fields': 'response{predictions{label,accuracy}},status'
        }

        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=get_list_id()),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH),
            headers=headers
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

    @pytest.mark.parametrize(
        'num_default_sessions, cached, expected_response', [
            (20, False,  # should be same results as above (460 / 23 = 20)
             generate_response({
                 'predictions': [
                     {'label': 'Move me', 'accuracy': 0.78},
                     {'label': 'I\'m hungry', 'accuracy': 0.12},
                     {'label': 'Call my family', 'accuracy': 0.05},
                     {'label': 'None of the above', 'accuracy': 0.0}
                 ]
             })),
            (14, True,
             generate_response({
                 'predictions': [
                     {'label': 'Move me', 'accuracy': 0.7},
                     {'label': 'I\'m hungry', 'accuracy': 0.13},
                     {'label': 'Call my family', 'accuracy': 0.11},
                     {'label': 'None of the above', 'accuracy': 0.0}
                 ]
             }))
        ])
    def test_transcribe_video_200_custom_model(self, client,
                                               num_default_sessions, cached,
                                               expected_response):
        list_id = get_list_id()
        create_list_model(list_id=list_id,
                          num_default_sessions=num_default_sessions,
                          to_cache=cached)

        # ignore any ids we don't know beforehand
        headers = {
            'X-Fields': 'response{predictions{label,accuracy}},status'
        }

        assert are_signals_cached(USER_ID, list_id) is cached

        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH),
            headers=headers
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK
        assert are_signals_cached(USER_ID, list_id) is True

        # unset current model so it reverts back to using default list
        PAVAList.update(id=list_id, current_model_id=None)

        cache.clear()

    def test_transcribe_video_200_custom_model_new_phrases_after(self, client):
        list_id = get_list_id()

        # first create new model - simulate algorithm
        model_id, session_ids = \
            create_list_model(list_id=list_id, num_default_sessions=10,
                              to_cache=True)

        # create new phrase after and record phrase for each session
        new_phrase = PAVAPhrase.create(list_id=list_id,
                                       content='I don\'t want that treatment')
        with db_session() as s:
            lst = PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)
            assert lst.has_added_phrases

        for session_id in session_ids:
            record_response = client.post(
                RECORD_ENDPOINT.format(session_id=session_id,
                                       phrase_id=new_phrase.id),
                data=construct_request_video_data(
                    I_DONT_WANT_THAT_TREATMENT_VIDEO_PATH
                )
            )
            assert record_response.status_code == HTTPStatus.CREATED

        # "I don't want that treatment" should not appear in results
        expected_response = generate_response({
            'predictions': [
                {'label': 'I don\'t want that', 'accuracy': 0.51},
                {'label': 'I\'m cold', 'accuracy': 0.2},
                {'label': 'What time is it?', 'accuracy': 0.15},
                {'label': 'None of the above', 'accuracy': 0.0}
            ]
        })

        # ignore any ids we don't know beforehand
        headers = {
            'X-Fields': 'response{predictions{label,accuracy}},status'
        }
        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(
                I_DONT_WANT_THAT_TREATMENT_VIDEO_PATH
            ),
            headers=headers
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        cache.clear()

        # updating date created of old model; simulates new model being created
        PAVAModel.update(id=model_id, date_created=datetime.now())

        # "I don't want that treatment" should now be in the results
        expected_response = generate_response({
            'predictions': [
                {'label': 'I don\'t want that treatment', 'accuracy': 0.95},
                {'label': 'I don\'t want that', 'accuracy': 0.02},
                {'label': 'I\'m cold', 'accuracy': 0.01},
                {'label': 'None of the above', 'accuracy': 0.0}
            ]
        })

        # ignore any ids we don't know beforehand
        headers = {
            'X-Fields': 'response{predictions{label,accuracy}},status'
        }
        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(
                I_DONT_WANT_THAT_TREATMENT_VIDEO_PATH
            ),
            headers=headers
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

        # unset current model so it reverts back to using default list
        PAVAList.update(id=list_id, current_model_id=None)

        cache.clear()

    @pytest.mark.parametrize('data, expected_response, status_code', [
        (None, generate_response(
            status_message='Input payload validation failed: '
                           'A video file is required',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        ({'file': None}, generate_response(
            status_message='Input payload validation failed: '
                           'A video file is required',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        ({'file': (None, 'video.mp4')}, generate_response(
            status_message='The file is empty',
            status_code=HTTPStatus.BAD_REQUEST
        ), HTTPStatus.BAD_REQUEST),
        ({'file': io.BytesIO(open(MOVE_ME_VIDEO_PATH, 'rb').read())},
         generate_response(
             status_message='Input payload validation failed: '
                            'A video file is required',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        ({'file': (io.BytesIO(open(MOVE_ME_VIDEO_PATH, 'rb').read()),
                   'video.txt')},
         generate_response(
             status_message="Invalid video format - use ['.mp4']",
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST),
        ({'video': (io.BytesIO(open(MOVE_ME_VIDEO_PATH, 'rb').read()),
                    'video.mp4')},
         generate_response(
             status_message='Input payload validation failed: '
                            'A video file is required',
             status_code=HTTPStatus.BAD_REQUEST),
         HTTPStatus.BAD_REQUEST)
    ])
    def test_transcribe_video_400_failed(self, client, data, expected_response,
                                         status_code):
        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=get_list_id()),
            data=data
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == status_code

    def test_transcribe_video_400_archive_phrase(self, client):
        # "Move me" no longer in returned phrases after archive
        expected_response = generate_response({
            'predictions': [
                {'label': 'Call my family', 'accuracy': 0.47},
                {'label': "I'm hungry", 'accuracy': 0.41},
                {'label': "I'm in pain", 'accuracy': 0.03},
                {'label': 'None of the above', 'accuracy': 0.0}
            ]
        })
        # ignore any ids we don't know beforehand
        headers = {
            'X-Fields': 'response{predictions{label,accuracy}},status'
        }

        list_id = get_list_id()

        # first archive the top phrase from previous predictions
        # it should no longer be in the predictions result
        with db_session() as s:
            phrase_id = PAVAPhrase.get(
                s, query=(PAVAPhrase.id,),
                filter=((PAVAPhrase.list_id == list_id)
                        & (PAVAPhrase.content == 'Move me')), first=True)[0]
        PAVAPhrase.update(phrase_id, archived=True)

        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH),
            headers=headers
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.OK

    def test_transcribe_video_400_archive_all_default_phrases(self, client):
        expected_response = generate_response(
            status_message='No reference templates found',
            status_code=HTTPStatus.BAD_REQUEST
        )

        list_id = get_list_id()

        # update all phrases to archived
        with db_session() as s:
            phrase_ids = PAVAPhrase.get(s, query=(PAVAPhrase.id,),
                                        filter=(PAVAPhrase.list_id == list_id))
        for phrase_id in phrase_ids:
            PAVAPhrase.update(phrase_id, archived=True)

        # all phrases are archived, no reference templates found
        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH)
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.BAD_REQUEST

    def test_transcribe_video_404_archive_list(self, client):
        expected_response = generate_response(
            status_code=HTTPStatus.NOT_FOUND,
            status_message='List not found'
        )

        list_id = get_list_id()

        # first update the list to be archived
        PAVAList.update(list_id, status=ListStatus.ARCHIVED)

        # the result should return a not found - list is archived
        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH)
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.NOT_FOUND

    def test_transcribe_400_custom_list_no_model(self, client):
        expected_response = generate_response(
            status_code=HTTPStatus.BAD_REQUEST,
            status_message='No reference templates found'
        )

        # create custom list
        lst = PAVAList.create(name='New custom list', user_id=USER_ID)

        actual_response = client.post(
            VIDEO_ENDPOINT.format(list_id=lst.id),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH)
        )

        assert actual_response.get_json() == expected_response
        assert actual_response.status_code == HTTPStatus.BAD_REQUEST
