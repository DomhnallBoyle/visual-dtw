from datetime import datetime

from main.models import Base
from main.utils.db import db_session
from main.utils.exceptions import SessionNotFoundException
from sqlalchemy import Boolean, Column, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import backref, relationship


class PAVASession(Base):
    """List session model."""
    __tablename__ = 'pava_session'
    __not_found_exception__ = SessionNotFoundException

    # many to one
    list_id = Column(UUID(as_uuid=True), ForeignKey('pava_list.id'))

    # used to determine if picked up by session selection algorithm
    new = Column(Boolean, default=False)

    date_created = Column(TIMESTAMP(timezone=False), default=datetime.now)

    # one to many
    templates = relationship('PAVATemplate',
                             backref=backref('session', lazy='select'),
                             cascade='save-update, delete')

    @property
    def phrases(self):
        from main.models import PAVAPhrase, PAVATemplate

        with db_session() as s:
            phrases = PAVAPhrase.get(
                s, filter=(
                    (PAVAPhrase.list_id == self.list_id)
                    & (PAVAPhrase.archived == False)
                    & (PAVAPhrase.content != 'None of the above')
                )
            )

            phrase_templates = {
                str(phrase_id): template_id
                for phrase_id, template_id in
                PAVATemplate.get(
                    s, query=(PAVATemplate.phrase_id, PAVATemplate.id),
                    filter=(PAVATemplate.session_id == self.id)
                )
            }

        return [{
            'id': phrase.str_id,
            'content': phrase.content,
            'template_id': phrase_templates.get(phrase.str_id),  # default=None
            'in_model': phrase.in_model
        } for phrase in phrases]

    @property
    def completed(self):
        phrases = self.phrases  # returns phrases that aren't NOTA or archived
        num_phrases = len(phrases)
        num_recorded_templates = len([phrase for phrase in phrases
                                     if phrase['template_id']])

        return num_recorded_templates == num_phrases
