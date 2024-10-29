"""API handling.

Contains functionality for creating API blueprints for PAVA/SRAVI
"""
from flask import Blueprint
from flask_restx import Api
from main.utils.exceptions import CUSTOM_EXCEPTIONS

TITLE = 'Visual Dynamic Time Warping - {}'
VERSION = '1.0'
DESCRIPTION = 'An implementation of Dynamic Time Warping using visual '\
              'features for visual speech recognition'


class CustomAPI(Api):
    """Custom API extends Flask-RestX API.

    Handles custom exceptions with a function
    """

    def __init__(self, *args, **kwargs):
        """Constructor.

        Args:
            *args: arguments
            **kwargs (dict): keyword arguments
        """
        super().__init__(*args, **kwargs)
        self._handle_exceptions()

    def _handle_exceptions(self):
        """Custom exception handler.

        When an exception is raised, run the wrapper function

        Returns:
            None
        """
        def wrapper(e):
            response = {
                'status': {
                    'code': e.status_code,
                    'message': e.status_message
                },
                'response': {}
            }

            return response, e.response_code

        for exception in CUSTOM_EXCEPTIONS:
            self.error_handlers[exception] = wrapper


def create_pava_api():
    from main.utils.schemas import generate_pava_schemas

    blueprint = Blueprint('pava', __name__)
    api = CustomAPI(blueprint,
                    title=TITLE.format('PAVA'),
                    version=VERSION,
                    description=DESCRIPTION)

    generate_pava_schemas(api)

    from main.resources import pava_namespaces

    for namespace in pava_namespaces:
        api.add_namespace(namespace)

    return api


pava_api = create_pava_api()
