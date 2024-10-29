"""List Resources.

Contains endpoints for /users/<user_id/lists
"""
from http import HTTPStatus

from flask_restx import Namespace
from main.models import PAVAList, PAVAUser
from main.resources import Base
from main.utils.db import db_session
from main.utils.enums import ListStatus
from main.utils.schemas import empty_response_schema, \
    list_response_schema, list_schema, user_lists_response_schema
from main.utils.validators import validate_uuid

list_namespace = Namespace('List',
                           description='List endpoints',
                           path='/users/<user_id>/lists')


@list_namespace.route('')
class Lists(Base):
    """Resource for a non-specific list."""

    @list_namespace.doc(
        params={
            'user_id': {'description': 'User UUID'}
        },
        responses={
            400: 'Invalid User UUID',
            404: 'User not found'
        }
    )
    @list_namespace.marshal_with(list_response_schema, code=201,
                                 description=HTTPStatus.CREATED.phrase)
    @list_namespace.expect(list_schema, validate=True)
    def post(self, user_id):
        """Create List.

        Check if user exists first

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            UserNotFoundException: if user not found by ID

        Returns:
            json: created PAVA list json properties
        """
        validate_uuid(user_id)
        name = self.api.payload['name']

        with db_session() as s:
            PAVAUser.get(s, filter=(PAVAUser.id == user_id), first=True)

        lst = PAVAList.create(name=name, user_id=user_id)

        return \
            self.generate_response(lst, HTTPStatus.CREATED,
                                   HTTPStatus.CREATED.phrase), \
            HTTPStatus.CREATED

    @list_namespace.doc(
        params={
            'user_id': {'description': 'User UUID'}
        },
        responses={
            400: 'Invalid User UUID',
            404: 'User not found'
        }
    )
    @list_namespace.marshal_with(user_lists_response_schema)
    def get(self, user_id):
        """Retrieve User Lists.

        Validate UUID first

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            UserNotFoundException: if user not found in db

        Returns:
            json: containing list of PAVA List json objects
        """
        validate_uuid(user_id)

        # check if user exists
        with db_session() as s:
            PAVAUser.get(s, filter=(PAVAUser.id == user_id), first=True)

            lsts = PAVAList.get(s, filter=(PAVAList.user_id == user_id))

        # we don't want to show NOTA or archived phrases in results
        # length updated accordingly
        for lst in lsts:
            lst.phrases = [phrase for phrase in lst.phrases
                           if not phrase.is_nota and not phrase.archived]

        return self.generate_response(lsts)


@list_namespace.route('/<list_id>')
class List(Base):
    """Resource for operations on a specific PAVA List."""

    @list_namespace.doc(
        params={
            'user_id': {'description': 'User UUID'},
            'list_id': {'description': 'List UUID'}
        },
        responses={
            400: 'Invalid User or List UUID',
            404: 'List not found'
        }
    )
    @list_namespace.marshal_with(list_response_schema)
    def get(self, user_id, list_id):
        """Retrieve User List.

        Validate User and List UUID

        Raises:
            ListNotFoundException: if list not found by UUID

        Returns:
            json: Retrieved PAVA List object
        """
        validate_uuid(user_id, list_id)

        with db_session() as s:
            lst = PAVAList.get(s, filter=((PAVAList.id == list_id)
                                          & (PAVAList.user_id == user_id)),
                               first=True)

        # remove NOTA phrase
        lst.phrases = [phrase for phrase in lst.phrases
                       if not phrase.is_nota and not phrase.archived]

        return self.generate_response(lst)

    @list_namespace.doc(
        params={
            'user_id': {'description': 'User UUID'},
            'list_id': {'description': 'List UUID'}
        },
        responses={
            400: 'Invalid User or List UUID',
            404: 'User/List not found'
        }
    )
    @list_namespace.marshal_with(empty_response_schema)
    def delete(self, user_id, list_id):
        """Delete List.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            UserNotFoundException: if the user does not exist
            ListNotFoundException: if list does not exist

        Returns:
            json: empty response with success code indicating deletion
        """
        validate_uuid(user_id, list_id)

        with db_session() as s:
            PAVAUser.get(s, filter=(PAVAUser.id == user_id), first=True)

        PAVAList.update(id=list_id, status=ListStatus.ARCHIVED)

        return self.generate_response()
