"""Phrase Resource.

Contains API endpoints for /lists/<list_id>/phrases
"""
from http import HTTPStatus

from flask_restx import Namespace
from main.models import PAVAList, PAVAPhrase
from main.resources import Base
from main.utils.db import db_session
from main.utils.exceptions import InvalidOperationException
from main.utils.schemas import empty_response_schema, \
    list_phrases_response_schema, phrase_response_schema, phrase_schema
from main.utils.validators import validate_uuid

phrase_namespace = Namespace('Phrase', description='Phrase endpoints',
                             path='/lists/<list_id>/phrases')


@phrase_namespace.route('')
class Phrases(Base):
    """Resource that doesn't conform to a specific PAVA Phrase."""

    @phrase_namespace.doc(
        params={
            'list_id': {'description': 'List UUID'}
        },
        responses={
            400: 'Invalid List UUID',
            404: 'List not found'
        }
    )
    @phrase_namespace.marshal_with(phrase_response_schema, code=201,
                                   description=HTTPStatus.CREATED.phrase)
    @phrase_namespace.expect(phrase_schema, validate=True)
    def post(self, list_id):
        """Create Phrase.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            ListNotFoundException: if the list is not found by ID

        Returns:
            json: serialized created phrase object
        """
        validate_uuid(list_id)
        content = self.api.payload['content']

        with db_session() as s:
            lst = PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)
            if not lst.ready:
                raise InvalidOperationException('List not ready')

        phrase = PAVAPhrase.create(content=content, list_id=list_id)

        return self.generate_response(
            phrase, HTTPStatus.CREATED, HTTPStatus.CREATED.phrase
        ), HTTPStatus.CREATED

    @phrase_namespace.doc(
        params={
            'list_id': {'description': 'List UUID'}
        },
        responses={
            400: 'Invalid List UUID',
            404: 'List not found'
        }
    )
    @phrase_namespace.marshal_with(list_phrases_response_schema)
    def get(self, list_id):
        """Retrieve List Phrases.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID

        Returns:
            json: list containing serialized phrases
        """
        validate_uuid(list_id)

        with db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)

            phrases = PAVAPhrase.get(
                s, filter=((PAVAPhrase.list_id == list_id)
                           & (PAVAPhrase.content != 'None of the above')
                           & (PAVAPhrase.archived == False))
            )

        return self.generate_response(phrases)


@phrase_namespace.route('/<phrase_id>')
class Phrase(Base):

    @phrase_namespace.doc(
        params={
            'list_id': {'description': 'List UUID'},
            'phrase_id': {'description': 'Phrase UUID'}
        },
        responses={
            400: 'Invalid List or Phrase UUID',
            404: 'List/Phrase not found'
        }
    )
    @phrase_namespace.marshal_with(empty_response_schema)
    def delete(self, list_id, phrase_id):
        """Delete Phrase.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            ListNotFoundException: if the list does not exist
            PhraseNotFoundException: if phrase does not exist

        Returns:
            json: empty response with success code indicating deletion
        """
        validate_uuid(list_id, phrase_id)

        with db_session() as s:
            lst = PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)
            if not lst.ready:
                raise InvalidOperationException('List not ready')

        PAVAPhrase.update(id=phrase_id, archived=True)

        return self.generate_response()
