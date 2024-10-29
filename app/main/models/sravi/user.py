"""User model.

Contains a db schema for the user table in the database
"""
from main.models.user import User
from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship


class SRAVIUser(User):
    """User class."""
    __tablename__ = 'sravi_user'

    id = Column(Integer, primary_key=True)

    # one to many
    templates = relationship('SRAVITemplate', lazy='joined')
