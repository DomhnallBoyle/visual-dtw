"""User model.

Contains a db schema for the user table in the database
"""
from main import configuration
from main.models import Config, User
from main.utils.io import read_pickle_file
from sqlalchemy.orm import backref, relationship


def create_default_list(user_id, list_path, list_name):
    from main.models import PAVAList, PAVAPhrase

    # add default pava list - read from pickle file
    default_list = read_pickle_file(list_path)

    # to mimic default "best" list
    lst = PAVAList.create(name=list_name, user_id=user_id, default=True)

    for phrase in default_list.phrases:
        if not phrase.is_nota:
            PAVAPhrase.create(content=phrase.content, list_id=lst.id)


class PAVAUser(User):
    """User class."""
    __tablename__ = 'pava_user'

    # one to many
    # backref loaded on attribute access, doesn't load users list again
    lists = relationship('PAVAList',
                         backref=backref('user', lazy='select'),
                         cascade='save-update, delete')

    @classmethod
    def create(cls, default_list=False, **kwargs):
        """Create a model instance and add to the database.

        Does not require instance of class to use

        Args:
            default_list (boolean): give user copy of the default phrase list
            **kwargs (dict): keyword arguments for the model instance

        Returns:
            obj: created model instance
        """
        user_object = super().create(config=Config(), **kwargs)

        if default_list:
            # create default main and sub-list
            create_default_list(
                user_id=user_object.id,
                list_path=configuration.DEFAULT_LIST_PATH,
                list_name=configuration.DEFAULT_PAVA_LIST_NAME
            )
            create_default_list(
                user_id=user_object.id,
                list_path=configuration.DEFAULT_SUB_LIST_PATH,
                list_name=configuration.DEFAULT_PAVA_SUB_LIST_NAME
            )

        return user_object
