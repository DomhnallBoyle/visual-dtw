"""User Resources.

Contains endpoints for /users
"""
from http import HTTPStatus

from flask_restx import Namespace
from main.models import PAVAUser
from main.resources import Base
from main.utils.db import db_session
from main.utils.schemas import user_response_schema
from main.utils.validators import validate_uuid

user_namespace = Namespace('User', description='User endpoints',
                           path='/users')


@user_namespace.route('')
class Users(Base):
    """Resource that doesn't conform to a specific PAVA User."""

    @user_namespace.marshal_with(user_response_schema, code=201,
                                 description=HTTPStatus.CREATED.phrase)
    def post(self):
        """Create User.

        Give user the default phrase list on creation

        Returns:
            json: Created PAVA User object
        """
        # create a new user and add default phrase list
        user = PAVAUser.create(default_list=True)

        return \
            self.generate_response(user, HTTPStatus.CREATED,
                                   HTTPStatus.CREATED.phrase), \
            HTTPStatus.CREATED


@user_namespace.route('/<user_id>')
class User(Base):
    """Resource for operations on a specific PAVA User."""

    @user_namespace.marshal_with(user_response_schema, code=200,
                                 description=HTTPStatus.OK.phrase)
    @user_namespace.doc(
        params={
            'user_id': {'description': 'User UUID'}
        },
        responses={
            400: 'Invalid User UUID',
            404: 'User not found'
        }
    )
    def get(self, user_id):
        """Retrieve User.

        Retrieve PAVA User by ID

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            UserNotFoundException: if user not found by ID

        Returns:
            json: Retrieved PAVA User object
        """
        validate_uuid(user_id)

        with db_session() as s:
            user = PAVAUser.get(s, filter=(PAVAUser.id == user_id),
                                first=True)

        return self.generate_response(user)
