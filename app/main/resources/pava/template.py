"""Template Resource.

Contains API endpoints for /phrases/<phrase_id>/templates
"""
from http import HTTPStatus

from flask_restx import Namespace
from main.models import PAVAList, PAVAPhrase, PAVATemplate
from main.resources import Base
from main.utils.db import db_session
from main.utils.schemas import template_response_schema
from main.utils.tasks import test_models
from main.utils.validators import validate_uuid
from sqlalchemy.orm import defer, noload

NUM_GROUNDTRUTH_TEMPLATES_PER_MODEL_TESTS = 25

template_namespace = Namespace(
    'Template', description='Template endpoints',
    path='/phrases/<phrase_id>/templates/<template_id>'
)


@template_namespace.route('')
class Templates(Base):
    """Resource that doesn't conform to a specific PAVA Template."""

    @template_namespace.doc(
        params={
            'phrase_id': {'description': 'Phrase UUID'},
            'template_id': {'description': 'Template UUID'}
        },
        responses={
            400: 'Invalid Phrase or Template UUID',
            404: 'Phrase or Template not found'
        }
    )
    @template_namespace.marshal_with(template_response_schema, code=200,
                                     description=HTTPStatus.OK.phrase)
    def put(self, phrase_id, template_id):
        """Update Template.

        Update a specific template object

        Raises:
            InvalidUUIDException: if ID is not a valid UUID
            PhraseNotFoundException: if phrase not found by ID
            PhraseTemplateLimitException: if phrase template limit met

        Returns:
            json: updated template instance
        """
        validate_uuid(phrase_id, template_id)

        with db_session() as s:
            phrase = PAVAPhrase.get(s, filter=(PAVAPhrase.id == phrase_id),
                                    first=True)

            template = PAVATemplate.get(
                s, loading_options=(defer('blob'),),
                filter=(PAVATemplate.id == template_id), first=True
            )

        if not template.phrase_id or template.phrase_id != phrase_id:
            # label template with the chosen phrase
            PAVATemplate.update(template.id, phrase_id=phrase.id)

        # find out if we should test models
        with db_session() as s:
            lst = PAVAList.get(s, loading_options=(noload('phrases'),),
                               filter=((PAVAPhrase.id == phrase_id)
                                       & (PAVAPhrase.list_id == PAVAList.id)),
                               first=True)

            num_groundtruth_templates = PAVATemplate.get(
                s, loading_options=(defer('blob'),),
                filter=((PAVATemplate.phrase_id == PAVAPhrase.id)
                        & (PAVAPhrase.content != 'None of the above')
                        & (PAVAPhrase.list_id == lst.str_id)
                        & (PAVATemplate.session_id == None)),
                count=True
            )

            if num_groundtruth_templates % \
                    NUM_GROUNDTRUTH_TEMPLATES_PER_MODEL_TESTS == 0:
                test_models.delay(lst.str_id)

        # TODO: Perform other template updates here from json payload

        return self.generate_response(template)
