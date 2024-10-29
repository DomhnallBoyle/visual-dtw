"""Model Resource.

Contains API endpoints for /lists/<list_id>/model
"""
import json
import uuid

from flask_restx import Namespace
from main import cache, configuration, redis_cache
from main.models import PAVAList
from main.resources import Base
from main.utils.db import db_session
from main.utils.enums import ListStatus
from main.utils.validators import validate_uuid

model_namespace = Namespace('Model', description='Model endpoints',
                            path='/lists/<list_id>/model')


@model_namespace.route('/build')
class Build(Base):

    def post(self, list_id):
        """Build List Model

        Raises:
            ListNotFoundException: if the list does not exist

        Returns:
            json: containing the build_id and # of queued jobs
        """
        validate_uuid(list_id)

        with db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)

        # generate a unique build id
        build_id = str(uuid.uuid4())

        # add the list id to the session selection redis queue
        queue_item = json.dumps({'list_id': str(list_id), 'build_id': build_id})
        redis_cache.rpush(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME, queue_item)

        # update status of list to queued
        PAVAList.update(id=list_id, status=ListStatus.QUEUED)
        cache.set(build_id, ListStatus.QUEUED.name, configuration.REDIS_MAX_TIMEOUT)

        return self.generate_response({
            'build_id': build_id,
            'num_queued_lists': redis_cache.llen(configuration.REDIS_SESSION_SELECTION_QUEUE_NAME)
        })


@model_namespace.route('/status/<build_id>')
class Status(Base):

    def get(self, list_id, build_id):
        """Get status of model build

        Raises:
            ListNotFoundException: if the list does not exist

        Returns:
            json: containing the status of the model build
        """
        validate_uuid(list_id, build_id)

        with db_session() as s:
            PAVAList.get(s, filter=(PAVAList.id == list_id), first=True)

        # get status from cache
        build_status = cache.get(build_id)

        return self.generate_response({
            'status': build_status
        })
