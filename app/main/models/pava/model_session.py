from main.models import Base
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID


class PAVAModelSession(Base):
    __tablename__ = 'pava_model_session'

    # many to one
    model_id = Column(UUID(as_uuid=True), ForeignKey('pava_model.id'))
    session_id = Column(UUID(as_uuid=True), ForeignKey('pava_session.id'))
