"""Phrase model.

Contains a db schema for the phrase table in the database
"""
from main.models import Base
from sqlalchemy import Column, String


class SRAVIPhrase(Base):
    """Phrase class."""
    __tablename__ = 'sravi_phrase'

    id = Column(String, primary_key=True)
    phrase_set = Column(String)
    phrase_set_id = Column(String)
    content = Column(String)
