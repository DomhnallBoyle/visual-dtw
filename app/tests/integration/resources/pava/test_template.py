from http import HTTPStatus

import pytest
from main.models import PAVAPhrase, PAVATemplate
from main.utils.db import db_session
from tests.integration.resources.pava.test_transcribe import \
    MOVE_ME_VIDEO_PATH, VIDEO_ENDPOINT, get_list_id
from tests.integration.utils import construct_request_video_data, \
    generate_response

BASE_ENDPOINT = '/pava/api/v1/phrases/{phrase_id}/templates/{template_id}'


class TestTemplate:

    def test_label_template_200(self, client):
        list_id = get_list_id()

        response = client.post(
            VIDEO_ENDPOINT.format(list_id=list_id),
            data=construct_request_video_data(MOVE_ME_VIDEO_PATH)
        )
        assert response.status_code == HTTPStatus.OK
        response = response.get_json()['response']
        template_id = response['template_id']
        phrase_id_1 = response['predictions'][0]['id']
        phrase_id_2 = response['predictions'][1]['id']

        # check template before
        with db_session() as s:
            template = PAVATemplate.get(
                s, filter=(PAVATemplate.id == template_id), first=True
            )
            assert template.phrase_id is None

        expected_response = generate_response({
            'id': template_id
        })

        response = client.put(
            BASE_ENDPOINT.format(
                phrase_id=phrase_id_1,
                template_id=template_id
            )
        )

        assert response.get_json() == expected_response
        assert response.status_code == HTTPStatus.OK

        # check template after
        with db_session() as s:
            template = PAVATemplate.get(
                s, filter=(PAVATemplate.id == template_id), first=True
            )
            assert str(template.phrase_id) == phrase_id_1

        # check number of templates
        with db_session() as s:
            phrase = PAVAPhrase.get(
                s, filter=(PAVAPhrase.id == phrase_id_1), first=True
            )
            assert len(phrase.templates) == 1

        # testing labelling again with a different phrase
        response = client.put(
            BASE_ENDPOINT.format(
                phrase_id=phrase_id_2,
                template_id=template_id
            )
        )

        assert response.get_json() == expected_response
        assert response.status_code == HTTPStatus.OK

        # check template after
        with db_session() as s:
            template = PAVATemplate.get(
                s, filter=(PAVATemplate.id == template_id), first=True
            )
            assert str(template.phrase_id) == phrase_id_2

    @pytest.mark.parametrize(
        'phrase_id, template_id, expected_response, status_code', [
            ('invalid_phrase_id', 'invalid_template_id',
             generate_response(
                 status_message='Invalid UUID given: invalid_phrase_id',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('2b1e4fa2-3da0-4bf0-b729-9c957ec189c6', 'invalid_template_id',
             generate_response(
                 status_message='Invalid UUID given: invalid_template_id',
                 status_code=HTTPStatus.BAD_REQUEST
             ), HTTPStatus.BAD_REQUEST),
            ('2b1e4fa2-3da0-4bf0-b729-9c957ec189c6',
             'f8b7c167-8cc2-44a8-bc7d-86cb557e33cc',
             generate_response(
                 status_message='Phrase not found',
                 status_code=HTTPStatus.NOT_FOUND
             ), HTTPStatus.NOT_FOUND),
            ('146a75a8-ac64-4cb7-9642-772edc4b1fcb',
             'f8b7c167-8cc2-44a8-bc7d-86cb557e33cc',
             generate_response(
                 status_message='Template not found',
                 status_code=HTTPStatus.NOT_FOUND
             ), HTTPStatus.NOT_FOUND),
        ])
    def test_label_template_failed(self, phrase_id, template_id,
                                   expected_response, status_code, client):
        response = client.put(
            BASE_ENDPOINT.format(
                phrase_id=phrase_id,
                template_id=template_id
            )
        )

        assert response.get_json() == expected_response
        assert response.status_code == status_code
