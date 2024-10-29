from http import HTTPStatus

from flask_restx import Namespace, reqparse
from main.models import Config, PAVASession, PAVATemplate
from main.resources import Base
from main.utils.cfe import run_cfe
from main.utils.db import db_session
from main.utils.exceptions import InvalidPayloadException, \
    TemplateNotFoundException
from main.utils.pre_process import pre_process_signals
from main.utils.schemas import template_response_schema
from main.utils.tasks import test_models
from main.utils.validators import validate_uuid, validate_video
from werkzeug.datastructures import FileStorage

record_namespace = Namespace('Record',
                             description='Record session',
                             path='/sessions/<session_id>')

parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument('file', location='files', type=FileStorage, required=True)


@record_namespace.route('/phrases/<phrase_id>/record')
class Phrase(Base):

    @record_namespace.doc(
        params={
            'session_id': {'description': 'Session UUID'},
            'phrase_id': {'description': 'Phrase UUID'},
            'file': {'description': 'Video File'}
        },
        responses={
            400: 'Invalid video format or session UUID',
            404: 'Session not found'
        }
    )
    @record_namespace.marshal_with(template_response_schema, code=201,
                                   description=HTTPStatus.CREATED.phrase)
    @record_namespace.expect(parser)
    def post(self, session_id, phrase_id):
        """Record Session Phrase.

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            InvalidVideoException: if the video format is invalid
            SessionNotFoundException: if session does not exist
            CFEException: if CFE can't process video

        Returns:
            json: containing the created template response
        """
        try:
            args = parser.parse_args()
        except Exception:
            raise InvalidPayloadException('A video file is required')

        video_file = args['file']

        validate_uuid(session_id, phrase_id)
        validate_video(video_file=video_file)

        with db_session() as s:
            session = PAVASession.get(
                s, filter=(PAVASession.id == session_id), first=True
            )
        completed_before = session.completed

        signal_blob = run_cfe(video_file=video_file)
        signal_blob = pre_process_signals(signals=[signal_blob],
                                          **Config().__dict__)[0]

        try:
            # for over-writing purposes if necessary e.g. re-record
            with db_session() as s:
                template = PAVATemplate.get(
                    s, filter=((PAVATemplate.session_id == session_id)
                               & (PAVATemplate.phrase_id == phrase_id)),
                    first=True
                )
            PAVATemplate.delete(id=template.id)
        except TemplateNotFoundException:
            pass

        template = PAVATemplate.create(
            blob=signal_blob,
            session_id=session_id,
            phrase_id=phrase_id
        )

        completed_after = session.completed
        if not completed_before and completed_after:
            # set session to new if completed again to kick off algorithm
            PAVASession.update(id=session_id, new=True)

            # TODO: Need to be careful with this
            #  it was designed for testing non model sessions that are recorded in 1 go
            #  e.g. if I have a model w/ 5 sessions (complete and not new) and I decide I want to add & record a session
            #  then it's fine to test this new completed session
            #  e.g. if I have a model w/ 5 sessions (complete and not new) and I add a phrase, my sessions become
            #  incomplete. Therefore, recording the new phrase 5 times results in 5 completed session which kicks off
            #  the model testing 5 times. It shouldn't be done this way
            # # kick off model testing
            # test_models.delay(session.list_id)

        return self.generate_response(
            response=template,
            status_code=HTTPStatus.CREATED,
            status_message=HTTPStatus.CREATED.phrase
        ), HTTPStatus.CREATED
