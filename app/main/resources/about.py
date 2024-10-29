"""About Resource.

Contains API endpoints for /about
"""
from flask import request
from flask_restx import Namespace
from main.resources import Base
from main.utils.schemas import about_response_schema

about_namespace = Namespace('About', description='API info', path='/about')


@about_namespace.route('')
class About(Base):
    """About class.

    Contains methods for /about
    """

    @about_namespace.marshal_with(about_response_schema)
    def get(self):
        """Retrieve API info.

        Construct and return API specific information

        Returns:
            json: containing API information
        """
        from main import api as api_manager

        application = str(request.url_rule).split('/')[1]
        api = getattr(api_manager, f'{application}_api')

        return self.generate_response({
            'name': api.title,
            'version': api.version,
            'description': api.description
        })
