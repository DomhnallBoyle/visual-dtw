"""Model schemas.

Contains custom model schemas for serialization/validation
"""
from flask_restx import fields
from main.utils.enums import ListStatus
from main.utils.fields import EmptyDict, IntEnum, UUID


about_response_schema = None
list_phrases_response_schema = None
list_response_schema = None
phrase_response_schema = None
list_sessions_response_schema = None
session_response_schema = None
transcribe_response_schema = None
user_lists_response_schema = None
user_response_schema = None
template_response_schema = None
empty_response_schema = None

list_schema = None
phrase_schema = None


def generate_pava_schemas(api):
    status_schema = api.model('Status Response', {
        'status': fields.Nested(api.model('Status', {
            'message': fields.String(description='Status message',
                                     example='Example message'),
            'code': fields.Integer(description='Status code',
                                   example=123)
        }))
    })

    user_schema = api.model('User', {
        'id': UUID(description='User UUID'),
        'config_id': UUID(description='User config UUID')
    })

    global user_response_schema
    user_response_schema = api.inherit(
        'User Response', status_schema, {
            'response': fields.Nested(user_schema),
        })

    global phrase_schema
    phrase_schema = api.model('Phrase', {
        'id': UUID(description='Phrase UUID', readonly=True),
        'content': fields.String(example='Turn on the light',
                                 description='Phrase text', min_length=1,
                                 max_length=50, required=True),
        'in_model': fields.Boolean(description='Phrase In Model?', readonly=True),
    })

    global phrase_response_schema
    phrase_response_schema = api.inherit(
        'Phrase Response', status_schema, {
            'response': fields.Nested(phrase_schema)
        }
    )

    session_phrase_schema = api.inherit(
        'Session Phrase', phrase_schema, {
            'template_id': UUID()
        }
    )

    session_schema = api.model('Session (Multi)', {
        'id': UUID(description='Session UUID', readonly=True),
        'completed': fields.Boolean(description='Session recording completed',
                                    readonly=True)
    })

    global list_sessions_response_schema
    list_sessions_response_schema = api.inherit(
        'List Sessions Response', status_schema, {
            'response': fields.Nested(session_schema, as_list=True)
        }
    )

    single_session_schema = api.inherit(
        'Session (Single)', session_schema, {
            'phrases': fields.Nested(session_phrase_schema, as_list=True)
        }
    )

    global session_response_schema
    session_response_schema = api.inherit(
        'Session Response', status_schema, {
            'response': fields.Nested(single_session_schema)
        }
    )

    global list_phrases_response_schema
    list_phrases_response_schema = api.inherit(
        'List Phrases Response', phrase_response_schema, {
            'response': fields.Nested(phrase_schema, as_list=True)
        }
    )

    current_model_schema = api.model('Current Model', {
        'id': UUID(description='List UUID', readonly=True),
        'num_sessions': fields.Integer(readonly=True),
        'phrase_coverage': fields.Float(readonly=True)
    })

    model_accuracy_schema = api.model('Model Accuracies', {
        'accuracy': fields.Float(readonly=True),
        'num_test_templates': fields.Integer(readonly=True),
        'date_created': fields.DateTime(readonly=True)
    })

    models_schema = api.model('Models', {
        'id': UUID(description='List UUID', readonly=True),
        'date_created': fields.DateTime(readonly=True),
        'accuracies': fields.Nested(model_accuracy_schema, allow_null=True,
                                    readonly=True, as_list=True)
    })

    global list_schema
    list_schema = api.model('List', {
        'id': UUID(description='List UUID', readonly=True),
        'name': fields.String(example='Morning List',
                              description='List name', min_length=1,
                              max_length=50, required=True),
        'default': fields.Boolean(readonly=True),
        'status': IntEnum(ListStatus, description='List status',
                          readonly=True),
        'num_phrases': fields.Integer(example=1, readonly=True),
        'phrases': fields.Nested(phrase_schema, readonly=True,
                                 as_list=True),
        'num_user_sessions': fields.Integer(readonly=True),
        'current_model': fields.Nested(current_model_schema, allow_null=True,
                                       readonly=True),
        'models': fields.Nested(models_schema, allow_null=True,
                                readonly=True, as_list=True),
        'last_updated': fields.DateTime(readonly=True)
    })

    global list_response_schema
    list_response_schema = api.inherit(
        'List Response', status_schema, {
            'response': fields.Nested(list_schema)
        }
    )

    global user_lists_response_schema
    user_lists_response_schema = api.inherit(
        'User Lists Response', list_response_schema, {
            'response': fields.Nested(list_schema, as_list=True)
        }
    )

    about_schema = api.model('About', {
        'name': fields.String(example='API name',
                              description='API name'),
        'version': fields.String(example='0.1',
                                 description='API version'),
        'description': fields.String(example='API description',
                                     description='API description')
    })

    global about_response_schema
    about_response_schema = api.inherit(
        'About Response', status_schema, {
            'response': fields.Nested(about_schema)
        }
    )

    transcribe_schema = api.model('Transcribe', {
        'template_id': UUID(description='Template UUID',
                            readonly=True),
        'predictions': fields.Nested(api.model('Prediction', {
            'id': UUID(description='Phrase UUID', readonly=True),
            'label': fields.String(example='Turn on the light'),
            'accuracy': fields.Float(example=0.98)
        }), as_list=True)
    })

    global transcribe_response_schema
    transcribe_response_schema = api.inherit(
        'Transcribe Response', status_schema, {
            'response': fields.Nested(transcribe_schema)
        }
    )

    template_schema = api.model('Template', {
        'id': UUID(description='Template UUID', readonly=True),
    })

    global template_response_schema
    template_response_schema = api.inherit(
        'Template Response', status_schema, {
            'response': fields.Nested(template_schema)
        }
    )

    api.inherit('Error Response', status_schema, {
        'response': EmptyDict()
    })

    global empty_response_schema
    empty_response_schema = api.inherit('Empty Response', status_schema, {
        'response': EmptyDict()
    })
