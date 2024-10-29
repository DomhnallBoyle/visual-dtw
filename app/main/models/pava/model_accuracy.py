from datetime import datetime

from main.models import Base
from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID


class PAVAModelAccuracy(Base):
    __tablename__ = 'pava_model_accuracy'

    model_id = Column(UUID(as_uuid=True), ForeignKey('pava_model.id'))

    accuracy = Column(Float)
    num_test_templates = Column(Integer)
    date_created = Column(TIMESTAMP(timezone=False), default=datetime.now)
