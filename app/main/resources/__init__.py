from .base import Base
from .about import about_namespace
from .pava import list_namespace, phrase_namespace, model_namespace, record_namespace, \
    session_namespace, template_namespace, transcribe_namespace, user_namespace

__all__ = ['Base']

pava_namespaces = [about_namespace, list_namespace, phrase_namespace, model_namespace,
                   record_namespace, session_namespace, template_namespace,
                   transcribe_namespace, user_namespace]
