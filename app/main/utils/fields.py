"""Custom model fields.

Contains Custom model fields that aren't supported by Flask-Restplus
"""

from flask_restx import fields
from main.utils.validators import validate_uuid


class EmptyDict(fields.Raw):
    __schema_example__ = {}


class UUID(fields.Raw):
    """Custom UUID field."""

    __schema_example__ = '971ab1c2-219d-4a18-adc5-f446adc84d76'

    def format(self, value):
        """Format value for response.

        Convert UUID object to string

        Args:
            value (UUID): uuid object for formatting

        Returns:
            string: formatted UUID
        """
        return str(value)

    def validate(self, value):
        """Method for validating the field during request.

        Validate the UUID by trying to convert string to UUID object

        Args:
            value (string): value for validating

        Returns:
            None
        """
        validate_uuid(value)


class IntEnum(fields.Raw):

    __schema_example__ = 'READY'

    def __init__(self, enum_type, **kwargs):
        super().__init__(**kwargs)
        self.enum_type = enum_type

    def format(self, value):
        return self.enum_type(value).name
