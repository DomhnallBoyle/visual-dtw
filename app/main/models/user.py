"""Base User model.

Contains logic to be inherited by child User models.
"""
from main.models import Base
from main.utils.exceptions import UserNotFoundException
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship


class User(Base):
    """Base User class.

    Abstract model
    """
    __abstract__ = True
    __not_found_exception__ = UserNotFoundException

    # one to one
    @declared_attr
    def config_id(self):
        """Config ID foreign key.

        Each User model has a config ID
        Abstract models cannot have relationship Columns - declared attribute

        Returns:
            obj: unique foreign key column field
        """
        return Column(UUID(as_uuid=True), ForeignKey('config.id'))

    @declared_attr
    def config(self):
        """Config relationship instance.

        Lazy join so model is loaded straight away
        Abstract models cannot have relationship Columns - declared attribute

        Returns:
            obj: config instance
        """
        return relationship('Config', cascade='save-update, delete')
