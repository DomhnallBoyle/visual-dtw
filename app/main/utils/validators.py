"""Validation handling.

Contains custom logic for handling request validation.
"""
import os
from uuid import UUID

from main import configuration
from main.utils.exceptions import EmptyFileException, \
    InvalidUUIDException, InvalidVideoException


def validate_uuid(*args):
    """Validate a UUID.

    Try to convert string to UUID object

    Args:
        uuid (string): uuid string to validate

    Raises:
        InvalidUUIDException: if UUID is invalid
    """
    for uuid in args:
        try:
            UUID(uuid)
        except ValueError:
            raise InvalidUUIDException(uuid)


def validate_video(video_file):
    """Validate a video filename.

    Check the video files extension and make sure it is in the list of valid
    video extensions.

    Args:
        video_file (werkzeug.FileStorage): FileStorage video object

    Raises:
        InvalidVideoException: if video format is invalid
    """
    if not any(os.path.splitext(video_file.filename)[1] == _format
               for _format in configuration.VALID_VIDEO_FORMATS):
        raise InvalidVideoException

    # check if empty video or not
    _f = video_file.stream._file
    old_file_position = _f.tell()
    _f.seek(0, os.SEEK_END)
    size = _f.tell()
    _f.seek(old_file_position, os.SEEK_SET)

    if size == 0:
        raise EmptyFileException
