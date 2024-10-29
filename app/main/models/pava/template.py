"""Template model.

Contains a db schema for the template table in the database
"""
from main.models import Base
from main.utils.exceptions import TemplateNotFoundException
from sqlalchemy import Column, ForeignKey, PickleType
from sqlalchemy.dialects.postgresql import UUID


class PAVATemplate(Base):
    """Template class."""
    __tablename__ = 'pava_template'
    __not_found_exception__ = TemplateNotFoundException

    blob = Column(PickleType)

    # many to one
    session_id = Column(UUID(as_uuid=True), ForeignKey('pava_session.id'))
    phrase_id = Column(UUID(as_uuid=True), ForeignKey('pava_phrase.id'))
