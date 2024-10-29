"""List model.

Contains a db schema for the list table in the database
"""
from main import cache, configuration
from main.models import Base
from main.utils.db import db_session
from main.utils.enums import ListStatus
from main.utils.exceptions import InvalidNameException, \
    ListNotFoundException
from main.utils.io import read_pickle_file
from main.utils.parsing import clean_string
from main.utils.types import IntEnum
from sqlalchemy import Boolean, Column, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import backref, relationship


class PAVAList(Base):
    """User list model."""
    __tablename__ = 'pava_list'
    __not_found_exception__ = ListNotFoundException

    name = Column(String)
    default = Column(Boolean, default=False)
    status = Column(IntEnum(ListStatus), default=ListStatus.READY)

    # many to one
    user_id = Column(UUID(as_uuid=True), ForeignKey('pava_user.id'))

    # one to one
    current_model_id = Column(UUID(as_uuid=True), ForeignKey('pava_model.id'))

    # one to many
    sessions = relationship('PAVASession',
                            backref=backref('list', lazy='select'),
                            cascade='save-update, delete')

    # one to many
    phrases = relationship('PAVAPhrase', lazy='joined',
                           backref=backref('list', lazy='select'),
                           cascade='save-update, delete')

    # one to many
    models = relationship('PAVAModel', lazy='joined',
                          order_by='asc(PAVAModel.date_created)',
                          backref=backref('list', lazy='select'),
                          cascade='save-update, delete',
                          foreign_keys='PAVAModel.list_id')

    @property
    def is_default_sub_list(self):
        return self.default and \
           self.name == configuration.DEFAULT_PAVA_SUB_LIST_NAME

    @property
    def ready(self):
        return self.status == ListStatus.READY

    @property
    def archived(self):
        return self.status == ListStatus.ARCHIVED

    @property
    def updating(self):
        return self.status == ListStatus.UPDATING

    @property
    def current_model(self):
        from main.models import PAVAModel

        if not self.current_model_id:
            return None

        with db_session() as s:
            return PAVAModel.get(
                s, filter=(PAVAModel.id == self.current_model_id), first=True
            )

    @property
    def cache_key(self):
        return f'{str(self.user_id)}_{str(self.str_id)}_signals'

    @property
    def num_phrases(self):
        return len(self.phrases)

    @property
    def num_user_sessions(self):
        from main.models import PAVASession

        with db_session() as s:
            return PAVASession.get(s, filter=(PAVASession.list_id == self.id),
                                   count=True)

    @property
    def has_added_phrases(self):
        num_phrases = self.num_phrases - 1  # NOTE: excludes NOTA phrase but includes all archived phrases

        if self.default:
            num_default_phrases = {
                configuration.DEFAULT_PAVA_LIST_NAME: configuration.NUM_DEFAULT_PHRASES,
                configuration.DEFAULT_PAVA_SUB_LIST_NAME: configuration.NUM_DEFAULT_SUB_PHRASES
            }[self.name]
            if num_phrases > num_default_phrases:
                return True
        elif not self.default and num_phrases > 0:
            return True

        return False

    @property
    def last_updated(self):
        return self.current_model.date_created if self.current_model else None

    @classmethod
    def get(cls, s, **kwargs):
        result = super().get(s, **kwargs)

        if kwargs.get('count'):
            return result

        first = kwargs.get('first')

        if first:
            if isinstance(result, PAVAList) and result.archived:
                raise cls.__not_found_exception__
        else:
            if result and isinstance(result[0], PAVAList):
                return [lst for lst in result if not lst.archived]

        return result

    @classmethod
    def create(cls, **kwargs):
        from main.models import PAVAPhrase

        name = kwargs['name']

        # check if list name is blacklisted
        default = kwargs.get('default', False)
        if not default and clean_string(name) in \
            [clean_string(configuration.DEFAULT_PAVA_LIST_NAME),
             clean_string(configuration.DEFAULT_PAVA_SUB_LIST_NAME)]:
            raise InvalidNameException(f'\'{name}\' cannot be used')

        user_id = kwargs.get('user_id')
        if user_id:
            # check if list name not already being used
            # don't include archived lists i.e. allows recreation of lists if previous is archived
            with db_session() as s:
                other_list_names = [clean_string(list_name[0])
                                    for list_name in PAVAList.get(s, query=(PAVAList.name,),
                                                                  filter=((PAVAList.user_id == user_id)
                                                                          & (PAVAList.status != ListStatus.ARCHIVED)))]
            if clean_string(name) in other_list_names:
                raise InvalidNameException(f'\'{name}\' is already being used')

        lst = super().create(**kwargs)

        # also create NOTA phrase
        PAVAPhrase.create(content='None of the above', list_id=lst.id)

        return lst

    def get_reference_signals(self):
        model = self.current_model
        ref_signals = []

        if self.default and not model:
            # get default list templates from pickle file
            if self.is_default_sub_list:
                default_list = read_pickle_file(
                    configuration.DEFAULT_SUB_LIST_PATH
                )
            else:
                default_list = read_pickle_file(
                    configuration.DEFAULT_LIST_PATH
                )

            default_lookup = {
                phrase.content: phrase.templates
                for phrase in default_list.phrases
                if not phrase.is_nota
            }

            ref_signal_append = ref_signals.append
            for phrase in self.phrases:
                # always check updated database for archived phrases
                if not phrase.archived and not phrase.is_nota:
                    # new phrases can't be shown until model is created
                    for template in default_lookup.get(phrase.content, []):
                        ref_signal_append((phrase.content, template.blob))
        elif model:
            # model/s available - get model templates from cache
            ref_signals = cache.get(self.cache_key)
            if not ref_signals:
                ref_signals = [
                    (template.phrase.content, template.blob)
                    for template in model.get_templates()
                ]
                cache.set(self.cache_key, ref_signals)

        return ref_signals
