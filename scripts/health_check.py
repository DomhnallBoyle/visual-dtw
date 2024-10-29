import os
import requests
from http import HTTPStatus

APP_HOST = os.environ.get('APP_HOST', '127.0.0.1')
APP_PORT = os.environ.get('APP_PORT', 5000)


def main():
    """Run GET health-check on API
    Docker has the following exit codes:
        0: Success
        1: Failure
        2: Unknown
    """
    try:
        # skip ssl certificate verification
        response = requests.get(f'http://{APP_HOST}:{APP_PORT}/pava/api/v1',
                                verify=False)

        if response.status_code != HTTPStatus.OK:
            exit(1)

    except Exception as e:
        exit(1)


if __name__ == '__main__':
    main()
