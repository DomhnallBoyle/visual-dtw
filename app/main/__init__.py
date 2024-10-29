"""Flask API setup.

This module contains logic to setup the API, create the db, setup the
configuration and import namespaces
"""
import logging
import os
import traceback
from logging.handlers import RotatingFileHandler

import redis
from celery import Celery
from flask import Flask, request
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
from main.config import CONFIGS
from main.utils.query import RetryingQuery

app = Flask('Visual Dynamic Time Warping')

configuration = CONFIGS[os.getenv('FLASK_ENV', 'development')]()
app.config.from_object(configuration)

# https://docs.sqlalchemy.org/en/14/core/pooling.html#setting-pool-recycle
db = SQLAlchemy(app, query_class=RetryingQuery, engine_options={
    'pool_recycle': 900  # every 15 minutes
})

redis_cache = redis.Redis(host=configuration.REDIS_HOST, charset='utf-8', decode_responses=True)

cache = Cache(app, config={
    'CACHE_TYPE': configuration.CACHE_TYPE,
    'CACHE_REDIS_HOST': configuration.CACHE_REDIS_HOST,
    'CACHE_DEFAULT_TIMEOUT': configuration.CACHE_DEFAULT_TIMEOUT
})

celery = Celery('app', broker=configuration.CELERY_BROKER_URL)
celery.conf.update(app.config)


def create_app(drop_db=False):
    """Creates the Flask API app.

    Args:
        drop_db (boolean): whether to force recreate the db

    Returns:
        FlaskApp: created Flask app to be started
    """
    from main.api import pava_api
    from main.models import AlchemyEncoder
    from main.utils import cfe, db
    from main.utils.enums import Environment

    # do not require trailing slashes on URLs
    app.url_map.strict_slashes = False

    # register api blueprints
    app.register_blueprint(pava_api.app, url_prefix='/pava/api/v1')

    app.json_encoder = AlchemyEncoder

    db.setup(drop=drop_db)
    cfe.wait_until_up()

    # setup request logger with timestamps
    logging.getLogger('werkzeug').disabled = True
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        handlers=[
            RotatingFileHandler(
                filename=configuration.LOGGER_PATH,
                maxBytes=configuration.MAX_LOGGER_SIZE,
                backupCount=5
            )
        ]
    )

    # make sure we log requests and responses
    @app.after_request
    def after_request(response):
        logging.info(f'{request.remote_addr} {request.method} '
                     f'{request.scheme} {request.full_path} '
                     f'{response.status}')

        return response

    # method runs when exception thrown, make sure we log error
    @app.errorhandler(Exception)
    def exception_raised(e):
        tb = traceback.format_exc()
        logging.info(f'{request.remote_addr} {request.method} '
                     f'{request.scheme} {request.full_path} '
                     f'{tb}')

        return e.status_code

    # only run in non-testing environments
    if configuration.ENVIRONMENT != Environment.TESTING:
        exempt_update_endpoints = ['transcribe', 'record']

        @app.before_request
        def before_request():
            from main.utils.tasks import refresh_cache

            # don't run when transcribing or recording
            if not any([endpoint in request.url
                        for endpoint in exempt_update_endpoints]):
                refresh_cache.delay(request.view_args)

    return app
