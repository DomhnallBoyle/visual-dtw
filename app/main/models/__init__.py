from flask import json
from sqlalchemy.ext.declarative import DeclarativeMeta

from .base import Base
from .config import Config
from .user import User
from .pava import PAVAList, PAVAModel, PAVAModelAccuracy, PAVAModelSession, \
    PAVAPhrase, PAVATemplate, PAVASession, PAVAUser
from .sravi import SRAVIPhrase, SRAVITemplate, SRAVIUser

__all__ = ['Base', 'Config', 'PAVAList', 'PAVAModel', 'PAVAModelAccuracy',
           'PAVAModelSession', 'PAVAPhrase', 'PAVATemplate', 'PAVASession',
           'PAVAUser', 'SRAVIPhrase', 'SRAVITemplate', 'SRAVIUser', 'User']


class AlchemyEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o.__class__, DeclarativeMeta):
            data = {}
            fields = o.__json__ if hasattr(o, '__json__') else dir(o)

            for field in [f for f in fields
                          if not f.startswith('_')
                          and f not in ['metadata', 'query', 'query_class']]:
                value = o.__getattribute__(field)

                try:
                    json.dumps(value)
                    data[field] = value
                except TypeError:
                    data[field] = None

            return data

        return json.JSONEncoder.default(self, o)
