"""CFE microservice functionality.

Contains logic for interacting with the CFE microservice.
"""
import io
import socket
import tempfile
import time
from contextlib import closing
from http import HTTPStatus

import requests
from main import configuration
from main.utils.exceptions import CFEException
from main.utils.io import read_matrix_ark

MAX_RETRY_ATTEMPTS = 100


def wait_until_up():
    num_attempts = 0
    print('Attempting to connect to CFE', flush=True)
    while True:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            num_attempts += 1
            if s.connect_ex((configuration.CFE_HOST,
                             configuration.CFE_PORT)) == 0:
                print('CFE Connected!', flush=True)
                break
            else:
                if num_attempts == MAX_RETRY_ATTEMPTS:
                    print(f'Failed to connect to CFE '
                          f'after {MAX_RETRY_ATTEMPTS} attempts', flush=True)
                    exit()
                time.sleep(5)


def run_cfe(video_file):
    """Run Cropping Feature Extraction on a video file.

    Uses the cfe microservice, makes POST request and returns .ark file

    Args:
        video_file (FileStorage): represents uploaded file

    Raises:
        CFEException: If the CFE microservice comes back with code != 200

    Returns:
        numpy array: video feature matrix
    """
    # extract feature matrix from cfe using video file
    response = requests.post(
        url=configuration.CFE_URL,
        files={'video': io.BufferedReader(video_file)},
        verify=configuration.CFE_VERIFY
    )

    video_file.close()

    # CFE broke - should always return 200
    if response.status_code != HTTPStatus.OK:
        raise CFEException(
            status_message='CFE: Something went wrong on our end',
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            response_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )

    # if json returned, we know CFE failed to extract features
    if response.headers['Content-Type'] == 'application/json':
        response = response.json()
        raise CFEException(
            status_message=response['message'],
            status_code=response['code']
        )

    # temporary file deletes itself after context manager closes
    with tempfile.TemporaryFile() as f:
        f.write(response.content)

        # point to beginning of file after write
        f.seek(0)

        test_signal = read_matrix_ark(f)

        return test_signal
