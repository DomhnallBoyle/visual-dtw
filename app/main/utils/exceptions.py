"""Exception handling.

Contains custom exceptions and error handling
"""
import inspect
import sys
from http import HTTPStatus

from main import configuration


class BaseCustomException(Exception):
    """Base class for custom exceptions.

    Contains default message and code

    message (string): exception error message
    code (int): corresponding HTTP status code
    """

    status_message = 'Base error message'
    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    _response_code = None

    def __str__(self):
        return f'{self.status_code} - {self.status_message}'

    @property
    def response_code(self):
        if not self._response_code:
            return self.status_code

        return self._response_code


class CFEException(BaseCustomException):
    """Raised when the CFE microservice returns code != 200."""

    def __init__(self, status_message, status_code,
                 response_code=HTTPStatus.OK):
        self.status_message = status_message
        self.status_code = status_code
        self._response_code = response_code


class ObjectNotFoundException(BaseCustomException):
    """Raised when an generic object is not found in the database"""

    status_code = HTTPStatus.NOT_FOUND

    def __init__(self, obj_name='Object'):
        self.status_message = f'{obj_name} not found'


class UserNotFoundException(ObjectNotFoundException):
    """Raised when no user is found in the database with a particular id."""

    def __init__(self):
        super().__init__(obj_name='User')


class ListNotFoundException(ObjectNotFoundException):
    """Raised when no list of found by ID."""

    def __init__(self):
        super().__init__(obj_name='List')


class PhraseNotFoundException(ObjectNotFoundException):
    """Raised when no phrase is found by ID."""

    def __init__(self):
        super().__init__(obj_name='Phrase')


class TemplateNotFoundException(ObjectNotFoundException):
    """Raised when no template is found by ID"""

    def __init__(self):
        super().__init__(obj_name='Template')


class SessionNotFoundException(ObjectNotFoundException):
    """Raised when no session is found by ID"""

    def __init__(self):
        super().__init__(obj_name='Session')


class ModelNotFoundException(ObjectNotFoundException):
    """Raised when no model is found by ID"""

    def __init__(self):
        super().__init__(obj_name='Model')


class ReferenceTemplatesNotFoundException(BaseCustomException):
    """Raised when no reference templates are found to compare with."""

    status_message = 'No reference templates found'
    status_code = HTTPStatus.BAD_REQUEST


class InvalidVideoException(BaseCustomException):
    """Raised when an invalid video format is given for processing."""

    status_message = f'Invalid video format - ' \
                     f'use {configuration.VALID_VIDEO_FORMATS}'
    status_code = HTTPStatus.BAD_REQUEST


class InvalidSignalException(BaseCustomException):
    """Raised when an invalid signal format is given for processing."""

    status_message = f'Invalid template format - ' \
                     f'use {configuration.VALID_SIGNAL_FORMAT}'
    status_code = HTTPStatus.BAD_REQUEST


class InvalidUUIDException(BaseCustomException):
    """Raised when an invalid UUID is given."""

    status_message = 'Invalid UUID given: '
    status_code = HTTPStatus.BAD_REQUEST

    def __init__(self, uuid):
        """Constructor.

        Args:
            uuid (obj): the invalid UUID passed in to be raised
        """
        self.status_message += f'{uuid}'


class PhraseTemplateLimitException(BaseCustomException):
    """Raised when a phrase template limit has been met."""

    status_message = 'Phrase templates limit has been met'
    status_code = HTTPStatus.FORBIDDEN


class InvalidNameException(BaseCustomException):
    """Raised when a name is invalid e.g. already used or blacklisted"""

    status_message = 'Invalid name: '
    status_code = HTTPStatus.BAD_REQUEST

    def __init__(self, msg):
        self.status_message += f'{msg}'


class NoListPhrasesException(BaseCustomException):
    """Raised when operations on a session are attempted without there
    being any list phrases"""

    status_message = 'There are no phrases attached to this list'
    status_code = HTTPStatus.FORBIDDEN


class EmptyFileException(BaseCustomException):
    """Raised when operations on an empty file are attempted"""

    status_message = 'The file is empty'
    status_code = HTTPStatus.BAD_REQUEST


class InvalidPayloadException(BaseCustomException):
    """Raised when an invalid payload is sent to the API"""

    status_message = 'Input payload validation failed: '
    status_code = HTTPStatus.BAD_REQUEST

    def __init__(self, msg):
        self.status_message += msg


class IncorrectPhraseException(BaseCustomException):

    status_message = 'The closest phrase does not match what was uttered'
    status_code = HTTPStatus.BAD_REQUEST


class WeakTemplateException(BaseCustomException):

    status_message = 'Recording is not strong enough to be added'
    status_code = HTTPStatus.BAD_REQUEST


class InaccuratePredictionException(BaseCustomException):

    status_message = 'Unable to accurately predict the phrase'
    status_code = HTTPStatus.BAD_REQUEST


class InvalidOperationException(BaseCustomException):

    status_message = 'Invalid operation: '
    status_code = HTTPStatus.FORBIDDEN

    def __init__(self, msg):
        self.status_message += msg


CUSTOM_EXCEPTIONS = \
    [klass[1]
     for klass in inspect.getmembers(sys.modules[__name__], inspect.isclass)]

