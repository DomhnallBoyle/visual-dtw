from http import HTTPStatus

from flask_restx import Namespace, inputs
from main.models import PAVAList, PAVAPhrase, PAVASession
from main.resources import Base
from main.utils.db import db_session
from main.utils.exceptions import NoListPhrasesException
from main.utils.schemas import list_sessions_response_schema, \
    session_response_schema
from main.utils.validators import validate_uuid

session_namespace = Namespace('Session', description='Session endpoints',
                              path='/lists/<list_id>/sessions')

session_parser = session_namespace.parser()
session_parser.add_argument('completed', type=inputs.boolean)


@session_namespace.route('')
class Sessions(Base):

    @session_namespace.doc(
        params={
            'list_id': {'description': 'List UUID'}
        },
        responses={
            400: 'Invalid List UUID',
            404: 'List not found'
        }
    )
    @session_namespace.marshal_with(session_response_schema, code=201,
                                    description=HTTPStatus.CREATED.phrase)
    def post(self, list_id):
        """Create Session.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            ListNotFoundException: if the list is not found by ID

        Returns:
            json: serialized created session object
        """
        validate_uuid(list_id)

        with db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)
            phrases = PAVAPhrase.get(
                s, filter=(
                    (PAVAPhrase.list_id == list_id)
                    & (PAVAPhrase.content != 'None of the above')
                    & (PAVAPhrase.archived == False)
                )
            )

        if not phrases:
            raise NoListPhrasesException

        session = PAVASession.create(list_id=list_id, new=True)

        return self.generate_response(
            response=session,
            status_code=HTTPStatus.CREATED,
            status_message=HTTPStatus.CREATED.phrase
        ), HTTPStatus.CREATED

    @session_namespace.doc(
        params={
            'list_id': {'description': 'List UUID'}
        },
        responses={
            400: 'Invalid List UUID',
            404: 'List not found'
        }
    )
    @session_namespace.marshal_with(list_sessions_response_schema)
    @session_namespace.expect(session_parser)
    def get(self, list_id):
        """Get List Sessions.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            ListNotFoundException: if the list is not found by ID

        Returns:
            json: all the sessions associated with a list
        """
        validate_uuid(list_id)
        args = session_parser.parse_args()

        with db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)

            sessions = \
                PAVASession.get(s, filter=(PAVASession.list_id == list_id))

        # filter by completed status
        if args.completed is not None:
            sessions = [session for session in sessions
                        if session.completed == args.completed]

        return self.generate_response(response=sessions)


@session_namespace.route('/<session_id>')
class Session(Base):

    @session_namespace.marshal_with(session_response_schema)
    def get(self, list_id, session_id):
        """Get Session"""
        validate_uuid(list_id, session_id)

        with db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)

            session = PAVASession.get(s, filter=(PAVASession.id == session_id),
                                      first=True)

        return self.generate_response(session)
