"""Phrase model.

Contains a db schema for the phrase table in the database
"""
from datetime import datetime

from main.models import Base
from main.utils.db import db_session
from main.utils.exceptions import InvalidNameException, ModelNotFoundException, PhraseNotFoundException
from main.utils.parsing import clean_string
from sqlalchemy import Boolean, Column, ForeignKey, String
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import backref, relationship


class PAVAPhrase(Base):
    """Phrase class."""
    __tablename__ = 'pava_phrase'
    __not_found_exception__ = PhraseNotFoundException

    content = Column(String)
    archived = Column(Boolean, default=False)
    date_created = Column(TIMESTAMP(timezone=False), default=datetime.now)

    # many to one
    list_id = Column(UUID(as_uuid=True), ForeignKey('pava_list.id'))

    # one to many
    templates = relationship('PAVATemplate',
                             backref=backref('phrase', lazy='select'),
                             cascade='save-update, delete')

    @property
    def is_nota(self):
        return self.content == 'None of the above'

    @property
    def in_model(self):
        from main.models import PAVAModel, PAVAList

        with db_session() as s:
            try:
                model = PAVAModel.get(s, filter=((PAVAModel.id == PAVAList.current_model_id) &
                                                 (PAVAList.id == self.list_id)),
                                      first=True)
            except ModelNotFoundException:
                return False

        model_phrase_counts = model.get_phrase_counts()

        # phrase in model if phrase creation date < model creation date and
        # phrase recordings have been completed
        return self.content in model_phrase_counts and \
            model_phrase_counts[self.content] == model.num_sessions

    @classmethod
    def create(cls, **kwargs):
        content = kwargs['content']
        list_id = kwargs['list_id']

        # check phrase not already created before
        # don't include archived phrases i.e. allows you to recreate a phrase with the same name if
        # it's archived
        with db_session() as s:
            lst_phrases = [clean_string(phrase[0])
                           for phrase in PAVAPhrase.get(s, query=(PAVAPhrase.content,),
                                                        filter=((PAVAPhrase.list_id == list_id)
                                                                & (PAVAPhrase.archived == False)))]
        if clean_string(content) in lst_phrases:
            raise InvalidNameException(f'\'{content}\' is already in the list')

        return super().create(**kwargs)
