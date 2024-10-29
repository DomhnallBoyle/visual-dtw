"""Transcribe Resource.

Contains API endpoints for /users/<user_id>/transcribe
"""
import argparse
from http import HTTPStatus

from flask_restx import Namespace, reqparse
from main.models import PAVAList, PAVATemplate, PAVAUser
from main.resources import Base
from main.services.transcribe import transcribe_signal
from main.utils.cfe import run_cfe
from main.utils.db import db_session
from main.utils.exceptions import InvalidPayloadException, \
    ReferenceTemplatesNotFoundException
from main.utils.pre_process import pre_process_signals
from main.utils.schemas import transcribe_response_schema
from main.utils.validators import validate_uuid, validate_video
from werkzeug.datastructures import FileStorage

transcribe_namespace = Namespace('Transcribe',
                                 description='Transcribe phrases',
                                 path='/lists/<list_id>/transcribe')

parser = reqparse.RequestParser(bundle_errors=True)
parser.add_argument('file', location='files', type=FileStorage, required=True)


def make_predictions(list_id, video_file):
    with db_session() as s:
        lst = PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)
        dtw_params = lst.user.config.__dict__

    # extract feature matrix from cfe using video file
    test_signal = run_cfe(video_file=video_file)

    ref_signals = lst.get_reference_signals()
    if not ref_signals:
        raise ReferenceTemplatesNotFoundException

    # create phrase lookup to get correct ids for the predictions
    phrase_lookup = {
        phrase.content: phrase.id
        for phrase in lst.phrases
    }

    test_signal = pre_process_signals(signals=[test_signal],
                                      **dtw_params)[0]

    predictions = transcribe_signal(ref_signals=ref_signals,
                                    test_signal=test_signal,
                                    **dtw_params)

    # add phrase ids to the predictions
    for prediction in predictions:
        prediction['id'] = phrase_lookup[prediction['label']]

    # add NOTA phrase to the predictions
    predictions.append({
        'id': phrase_lookup['None of the above'],
        'label': 'None of the above',
        'accuracy': 0
    })

    return predictions, test_signal


@transcribe_namespace.route('/video')
class TranscribeVideo(Base):
    """Transcribe video class.

    Contains methods for /lists/<list_id>/transcribe/video
    """

    @transcribe_namespace.doc(
        params={
            'list_id': {'description': 'List UUID'},
            'file': {'description': 'Video File'}
        },
        responses={
            400: 'Invalid video format, list UUID or no reference templates',
            404: 'List not found'
        }
    )
    @transcribe_namespace.marshal_with(transcribe_response_schema, code=200,
                                       description=HTTPStatus.OK.phrase)
    @transcribe_namespace.expect(parser)
    def post(self, list_id):
        """Transcribe video.

        Take as input a user ID and video file, return predictions

        Args:
            list_id (int): unique user identifier for lookup

        Returns:
            json: containing the predictions from DTW/KNN
        """
        try:
            args = parser.parse_args()
        except Exception:
            raise InvalidPayloadException('A video file is required')

        video_file = args['file']

        validate_uuid(list_id)
        validate_video(video_file=video_file)

        predictions, test_signal = make_predictions(list_id=list_id,
                                                    video_file=video_file)

        template = PAVATemplate.create(blob=test_signal)

        return self.generate_response({
            'template_id': template.str_id,
            'predictions': predictions
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file_path', type=str)
    parser.add_argument('--num_hits', type=int, default=10)

    args = parser.parse_args()
    video_file_path = args.video_file_path
    num_hits = args.num_hits

    # create user and list
    user = PAVAUser.create(default_list=True)
    with db_session() as s:
        lst = PAVAList.get(s, filter=(PAVAList.user_id == user.id), first=True)

    for i in range(num_hits):
        with open(video_file_path, 'rb') as f:
            make_predictions(lst.id, f)

    # clean-up
    PAVAUser.delete(id=user.id)
