from .list import list_namespace
from .phrase import phrase_namespace
from .model import model_namespace
from .record import record_namespace
from .session import session_namespace
from .template import template_namespace
from .transcribe import transcribe_namespace
from .user import user_namespace

__all__ = ['list_namespace', 'record_namespace', 'session_namespace',
           'phrase_namespace', 'model_namespace', 'template_namespace',
           'transcribe_namespace', 'user_namespace']
