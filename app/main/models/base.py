"""Base model.

Contains logic to be inherited by other child models.
"""
import uuid

from flask import json
from main import db
from main.utils.query import RetryingQuery
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import UUID


class Base(db.Model):
    """Base class.

    Abstract model.
    """
    __abstract__ = True
    __not_found_exception__ = NotImplementedError
    query_class = RetryingQuery

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    @property
    def str_id(self):
        return str(self.id)

    def as_dict(self):
        """Convert to string format and then to dictionary.

        Uses custom JSON encoder

        Returns:
            dict: containing model properties and values
        """
        return json.loads(json.dumps(self))

    @classmethod
    def get(cls, s, query=None, loading_options=None, filter=None,
            distinct=False, first=False, count=False):
        """Generic get query method.

        Does not require instance of class to use

        Args:
            query (InstrumentedAttribute): attributes to select
            e.g. SELECT name FROM ...
            filter (BinaryExpression): expression to filter by
            distinct (boolean): whether to grab distinct values or not

        Returns:
            obj: result of the query (usually a list of objects)
        """
        if query:
            q = s.query(*query)
        else:
            q = s.query(cls)

        if loading_options:
            q = q.options(loading_options)

        if filter is not None:
            q = q.filter(filter)

        if distinct:
            q = q.distinct()

        if first:
            result = q.first()
            if not result:
                raise cls.__not_found_exception__

            return result

        if count:
            return q.count()

        return q.all()

    @classmethod
    def create(cls, **kwargs):
        """Create a model instance and add to the database.

        Does not require instance of class to use

        Args:
            **kwargs (dict): keyword arguments for the model instance

        Returns:
            obj: created model instance
        """
        from main.utils.db import db_session

        with db_session() as s:
            obj = cls(**kwargs)
            s.add(obj)
            s.commit()

            # need to refresh the object to get it's properties
            s.refresh(obj)

        return obj

    @classmethod
    def update(cls, id, loading_options=None, **kwargs):
        from main.utils.db import db_session

        with db_session() as s:
            q = s.query(cls)
            if loading_options:
                q = q.options(loading_options)

            obj = q.get(id)
            if not obj:
                raise cls.__not_found_exception__

            for k, v in kwargs.items():
                setattr(obj, k, v)

            s.commit()

    @classmethod
    def delete(cls, id):
        from main.utils.db import db_session

        with db_session() as s:
            obj = s.query(cls).get(id)
            if not obj:
                raise cls.__not_found_exception__

            s.delete(obj)
            s.commit()
