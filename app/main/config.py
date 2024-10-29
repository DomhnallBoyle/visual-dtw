"""Configuration handling.

This module contains configuration values relating to different types of
environments e.g. production, development or testing
"""
import os
from os.path import dirname as up

from main.utils.enums import Environment


class Config:
    """Base Configuration.

    Contains values to be used by all configuration types
    """
    PROJECT_PATH = up(up(up(__file__)))
    DATA_PATH = os.path.join(PROJECT_PATH, 'data')
    VIDEOS_PATH = os.path.join(DATA_PATH, 'videos')
    PHRASES_PATH = os.path.join(DATA_PATH, 'phrases.json')
    FIXTURES_PATH = os.path.join(DATA_PATH, 'fixtures.json')
    DEFAULT_TEMPLATES_PATH = os.path.join(DATA_PATH, 'default_templates.np')

    # database settings
    DATABASE_HOST = os.getenv('DATABASE_HOST', 'db')
    DATABASE_NAME = os.getenv('DATABASE_NAME', 'database')
    DATABASE_PORT = os.getenv('DATABASE_PORT', 5432)
    DATABASE_USER = os.getenv('DATABASE_USER', 'admin')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD', 'password')
    DATABASE_VERIFY = os.getenv('DATABASE_VERIFY', None)  # ssl certificate
    SQLALCHEMY_DATABASE_URI = \
        f'postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}' \
        f'@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}'
    if DATABASE_VERIFY:
        SQLALCHEMY_DATABASE_URI += \
            f'?sslmode=verify-full&sslrootcert={DATABASE_VERIFY}'

    # cfe settings
    CFE_HOST = os.getenv('CFE_HOST', 'cfe')
    CFE_PORT = int(os.getenv('CFE_PORT', 5001))
    CFE_VERIFY = os.getenv('CFE_VERIFY', None)  # ssl certificate
    CFE_URL = f'http://{CFE_HOST}:{CFE_PORT}/api/v1/extract/'

    # redis settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    REDIS_DEFAULT_TIMEOUT = os.getenv('REDIS_DEFAULT_TIMEOUT', 300)  # 5 mins
    REDIS_SESSION_SELECTION_QUEUE_NAME = 'session_selection_queue'
    REDIS_MAX_TIMEOUT = 86400  # 1 day

    # celery settings
    CELERY_BROKER_URL = f'redis://{REDIS_HOST}:6379/0'
    CELERY_RESULT_BACKEND = f'redis://{REDIS_HOST}:6379/0'

    # flask cache settings
    CACHE_TYPE = 'redis'
    CACHE_REDIS_HOST = REDIS_HOST
    CACHE_DEFAULT_TIMEOUT = REDIS_DEFAULT_TIMEOUT

    # flask/flask rest x configuration settings
    DEBUG = False
    ERROR_404_HELP = False
    PROPAGATE_EXCEPTIONS = False
    BUNDLE_ERRORS = True
    ERROR_INCLUDE_MESSAGE = False

    # sqlalchemy settings
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = os.getenv('SQLALCHEMY_ECHO', 'False') == 'True'

    SRAVI_PHRASE_SETS = ['SR', 'SRF', 'S2F', 'S2R', 'SB']
    FEATURE_TYPES = ['AE_norm', 'AE_norm_2']
    VALID_VIDEO_FORMATS = ['.mp4']
    VALID_SIGNAL_FORMAT = '.ark'

    # logger settings
    MAX_LOGGER_SIZE = 10485760  # 10 MB
    LOGGER_PATH = '/shared/app.log'

    # default list settings
    DEFAULT_PAVA_LIST_ID = 'af1fa0d7-9ead-4976-9f22-a81f420ac589'
    DEFAULT_PAVA_LIST_NAME = 'Default'
    DEFAULT_LIST_PATH = os.path.join(DATA_PATH, 'default_list.pkl')
    NUM_DEFAULT_SESSIONS = 13
    NUM_DEFAULT_PHRASES = 20

    # default sub-list settings
    DEFAULT_PAVA_SUB_LIST_ID = '82d15cea-76f3-4cdd-9343-49ddb04c15bb'
    DEFAULT_PAVA_SUB_LIST_NAME = 'Default sub-list'
    DEFAULT_SUB_LIST_PATH = os.path.join(DATA_PATH, 'default_sub_list.pkl')
    NUM_DEFAULT_SUB_PHRASES = 5


class DevelopmentConfig(Config):
    """Development Configuration.

    Contains values to be used by development mode only
    """
    ENVIRONMENT = Environment.DEVELOPMENT
    DEBUG = True
    SQLALCHEMY_ECHO = os.getenv('SQLALCHEMY_ECHO', 'True') == 'True'


class ProductionConfig(Config):
    """Production Configuration.

    Contains values to be used by production mode only
    """
    ENVIRONMENT = Environment.PRODUCTION
    CFE_URL = f'https://{Config.CFE_HOST}/api/v1/extract/'


class TestingConfig(Config):
    """Test Configuration.

    Contains values to be used by testing mode only
    """
    ENVIRONMENT = Environment.TESTING
    DEFAULT_TEMPLATES_PATH = os.path.join(Config.DATA_PATH,
                                          'test_templates.np')
    DEFAULT_LIST_PATH = os.path.join(Config.DATA_PATH, 'test_list.pkl')
    NUM_DEFAULT_SESSIONS = 20
    NUM_DEFAULT_PHRASES = 23


class ResearchConfig(Config):
    ENVIRONMENT = Environment.RESEARCH


CONFIGS = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'research': ResearchConfig
}
