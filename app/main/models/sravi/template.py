"""Template model.

Contains a db schema for the template table in the database
"""
from main.models import Base
from sqlalchemy import Column, ForeignKey, Integer, PickleType, String, \
    UniqueConstraint
from sqlalchemy.orm import relationship


class SRAVITemplate(Base):
    """Template class."""
    __tablename__ = 'sravi_template'

    # apply a unique constraint involving some table attributes
    __table_args__ = (
        UniqueConstraint('feature_type', 'user_id', 'session_id',
                         'phrase_id'),
    )

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer)
    feature_type = Column(String)
    blob = Column(PickleType)

    # many to one
    user_id = Column(Integer, ForeignKey('sravi_user.id'))

    # one to one
    phrase_id = Column(String, ForeignKey('sravi_phrase.id'))
    phrase = relationship('SRAVIPhrase', lazy='joined')

    @property
    def key(self):
        """Template property.

        Constructs a unique identifier for a template object

        Returns:
            string: unique identifier
        """
        return f'{self.feature_type}_{self.user_id}_{self.phrase_id}_' \
               f'{self.session_id}'
