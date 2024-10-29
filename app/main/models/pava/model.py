from datetime import datetime

from main import cache
from main.models import Base
from main.utils.db import db_session
from main.utils.exceptions import ModelNotFoundException
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import backref, defer, joinedload, relationship


class PAVAModel(Base):
    __tablename__ = 'pava_model'
    __not_found_exception__ = ModelNotFoundException

    # many to one
    list_id = Column(UUID(as_uuid=True), ForeignKey('pava_list.id'))

    date_created = Column(TIMESTAMP(timezone=False), default=datetime.now)

    # one to many
    model_sessions = relationship('PAVAModelSession', lazy='joined',
                                  backref=backref('model', lazy='select'),
                                  cascade='save-update, delete')

    # one to many
    model_accuracies = relationship(
        'PAVAModelAccuracy', lazy='joined',
        order_by='asc(PAVAModelAccuracy.date_created)',
        backref=backref('model', lazy='select'), cascade='save-update, delete')

    @property
    def num_sessions(self):
        return len(self.model_sessions)

    @property
    def accuracies(self):
        return self.model_accuracies

    @property
    def phrase_coverage(self):
        # completed phrase coverage
        phrase_count_lookup = self.get_phrase_counts()
        num_phrases = len(phrase_count_lookup)

        completed_phrases = \
            sum([1 if count == self.num_sessions else 0
                 for count in phrase_count_lookup.values()])

        return round(completed_phrases / num_phrases, 2)

    @cache.memoize(60)  # args (inc. self) are part of the cache key
    def get_phrase_counts(self):
        from main.models import PAVAPhrase

        with db_session() as s:
            phrases = PAVAPhrase.get(
                s, filter=((PAVAPhrase.list_id == self.list_id)
                           & (PAVAPhrase.content != 'None of the above'))
            )

        phrase_count_lookup = {phrase.content: 0 for phrase in phrases}
        for template in self.get_templates(load_blobs=False):
            phrase_count_lookup[template.phrase.content] += 1

        return phrase_count_lookup

    def get_templates(self, load_blobs=True):
        from main.models import PAVAModelSession, PAVAPhrase, PAVATemplate

        loading_options = (joinedload('phrase'),)
        if not load_blobs:
            loading_options += (defer('blob'),)

        with db_session() as s:
            templates = PAVATemplate.get(
                s, loading_options=loading_options,
                filter=(
                    (PAVATemplate.session_id == PAVAModelSession.session_id)
                    & (PAVAModelSession.model_id == self.id)
                    & (PAVATemplate.phrase_id == PAVAPhrase.id)
                    & (PAVAPhrase.archived == False)
                    & (PAVAPhrase.content != 'None of the above')
                )
            )

        # new phrases should not be included until model is created for them
        # i.e. only include templates that have phrases created before the model
        templates = [template for template in templates
                     if template.phrase.date_created < self.date_created]

        return templates
