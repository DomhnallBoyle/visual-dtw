import time

from flask_sqlalchemy import BaseQuery
from sqlalchemy.exc import DBAPIError


class RetryingQuery(BaseQuery):
    """Used to handle stale/closed database connections

    docs.sqlalchemy.org/en/14/core/pooling.html#disconnect-handling-optimistic
    """

    __retry_count__ = 3
    __retry_sleep_interval_sec__ = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        from main import db, logging

        attempts = 0
        while True:
            try:
                return super().__iter__()
            except DBAPIError as e:
                attempts += 1
                # check if connection is disconnected for some reason
                if e.connection_invalidated \
                        and attempts < self.__retry_count__:
                    # invalidate all connections and
                    # try to setup new connection before retrying
                    logging.info(f'Connection invalidated, '
                                 f'attempt {attempts}/{self.__retry_count__}, '
                                 f'retrying...')
                    self.session.invalidate()
                    self.session = db.session()
                    time.sleep(self.__retry_sleep_interval_sec__)
                    continue

                raise
