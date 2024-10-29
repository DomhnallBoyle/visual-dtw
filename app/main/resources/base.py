from http import HTTPStatus

from flask_restx import Resource


class Base(Resource):

    def generate_response(self, response=None, status_code=HTTPStatus.OK,
                          status_message=HTTPStatus.OK.phrase):
        if response is None:
            response = {}

        return {
            'response': response,
            'status': {
                'code': status_code,
                'message': status_message
            }
        }
