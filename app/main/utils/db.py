"""Database utils.

Module for creating and populating db
"""
import glob
import os
from contextlib import contextmanager

import numpy as np
from main import configuration, db
from main.utils.enums import Environment
from main.utils.io import read_ark_file, read_json_file, write_pickle_file
from main.utils.parsing import extract_template_info
from main.utils.pre_process import pre_process_signals
from sqlalchemy import inspect
from sqlalchemy.orm import joinedload
from sqlalchemy_utils import create_database, database_exists, drop_database


@contextmanager
def db_session():
    """Context Manager that yields a session for writing to the db.

    Performs any required db rollbacks, closes the session afterwards.

    Usage:
        with db_session() as s:
            <!-- do something with s here -->

    @contextmanager is similar to:
        with open() as f:
            ...

    Everything before 'yield' is equal to '__enter__'
    Everything after 'yield' is equal to '__exit__'

    See: https://jeffknupp.com/blog/2016/03/07/python-with-context-managers/

    Returns:
        None
    """
    session = db.session()

    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def add_sravi_objects(feature_type):
    """Add SRAVI objects to the database.

    A user has multiple templates
    Templates share the same phrases (can't be changed)

    Returns:
        None
    """
    from main.models import Config, SRAVIPhrase, SRAVITemplate, SRAVIUser
    from main.utils.exceptions import UserNotFoundException

    # add all phrases
    phrases = read_json_file(file_path=configuration.PHRASES_PATH)
    for phrase_set in configuration.SRAVI_PHRASE_SETS:
        for key in phrases[phrase_set]:
            SRAVIPhrase.create(id=phrase_set + str(key),
                               phrase_set=phrase_set,
                               phrase_set_id=str(key),
                               content=phrases[phrase_set][key])

    data_path = os.path.join(configuration.DATA_PATH, feature_type)

    templates = [os.path.join(data_path, template)
                 for template in os.listdir(data_path)
                 if template.endswith('.ark')]

    # add all R&D templates
    for template in templates:
        basename = os.path.basename(template).replace('.ark', '')

        user_id, phrase_set, phrase_id, session_id = \
            extract_template_info(template_id=basename)

        try:
            with db_session() as s:
                SRAVIUser.get(s, filter=(SRAVIUser.id == user_id), first=True)
        except UserNotFoundException:
            SRAVIUser.create(id=user_id, config=Config())

        if phrase_set == 'S3R':
            continue

        # don't do pre-processing here
        ark_matrix = read_ark_file(file_path=template)

        SRAVITemplate.create(user_id=user_id,
                             session_id=session_id,
                             phrase_id=(phrase_set + phrase_id),
                             feature_type=feature_type,
                             blob=ark_matrix)


def find_phrase_mappings(main_phrase_set):
    mappings = {}
    data = read_json_file(configuration.PHRASES_PATH)

    for i, phrase in data[main_phrase_set].items():
        mappings[main_phrase_set + i] = []

        for phrase_set in set(configuration.SRAVI_PHRASE_SETS):
            for k, phrase_2 in data[phrase_set].items():
                if phrase == phrase_2:
                    mappings[main_phrase_set + i].append(phrase_set + k)

    return mappings


def invert_phrase_mappings(phrase_mappings):
    d = {}
    for phrase, related_phrases in phrase_mappings.items():
        for related_phrase in related_phrases:
            d[related_phrase] = phrase

    return d


def save_default_list_to_file(list_id, list_path):
    from main.models import PAVAList

    # pickle the list so we don't need to keep querying for it later
    with db_session() as s:
        lst = PAVAList.get(
            s, loading_options=(joinedload('phrases').joinedload('templates')),
            filter=(PAVAList.id == list_id),
            first=True
        )

    write_pickle_file(lst, list_path)


def add_default_phrase_list(phrase_set_name,
                            list_id=configuration.DEFAULT_PAVA_LIST_ID,
                            list_name=configuration.DEFAULT_PAVA_LIST_NAME,
                            list_path=configuration.DEFAULT_LIST_PATH):
    """Only want to add phrase set with specific name e.g. PAVA-DEFAULT

    Args:
        phrase_set_name (str): name of the phrase set to add
        list_id (str): default list id
        list_name (str): default list name
        list_path (str): pickle save path for list object

    Returns:
        None
    """
    from main.models import Config
    from main.models.pava import PAVAList, PAVAPhrase, PAVATemplate, \
        PAVASession

    feature_type = 'AE_norm_2'

    # create template lookup table
    template_lookup = {}
    data_path = os.path.join(configuration.DATA_PATH, feature_type)

    templates = glob.glob(os.path.join(data_path, '*.ark'))

    for template in templates:
        basename = os.path.basename(template).replace('.ark', '')
        user_id, phrase_set, phrase_id, session_id = \
            extract_template_info(basename)
        key = f'{feature_type}_{user_id}_{phrase_set + phrase_id}_{session_id}'
        template_lookup[key] = template

    # create default list obj in db
    lst = PAVAList.create(id=list_id, name=list_name, default=True)

    # collect default templates from file
    template_list = np.genfromtxt(configuration.DEFAULT_TEMPLATES_PATH,
                                  delimiter=',', dtype='str')

    all_phrases = read_json_file(configuration.PHRASES_PATH)

    # create phrase objects in db and phrase lookup table
    phrase_lookup = {}
    for phrase_key, phrase_content in all_phrases[phrase_set_name].items():
        phrase = PAVAPhrase.create(list_id=lst.id, content=phrase_content)
        phrase_lookup[phrase_content] = phrase

    dtw_params = Config().__dict__

    # pre-process templates and create template db objects
    for template_id in template_list:
        phrase_set, phrase_id = \
            extract_template_info(template_id, from_default_list=True)[1:3]

        phrase_content = all_phrases[phrase_set][phrase_id]

        pava_phrase = phrase_lookup.get(phrase_content)
        if not pava_phrase:
            continue

        template_path = template_lookup[template_id]

        ark_matrix = read_ark_file(file_path=template_path)
        pre_processed_matrix = pre_process_signals([ark_matrix],
                                                   **dtw_params)[0]

        PAVATemplate.create(phrase_id=pava_phrase.id,
                            blob=pre_processed_matrix)

    # update default templates to sessions
    with db_session() as s:
        lst = PAVAList.get(
            s, loading_options=(joinedload('phrases').joinedload('templates')),
            filter=(PAVAList.id == list_id),
            first=True
        )

    for i in range(configuration.NUM_DEFAULT_SESSIONS):
        # create sessions for the default list
        session = PAVASession.create(list_id=list_id)
        # update template to include session id
        for phrase in lst.phrases:
            if not phrase.is_nota:
                PAVATemplate.update(
                    id=phrase.templates[i].id,
                    session_id=session.id
                )

    save_default_list_to_file(list_id=list_id, list_path=list_path)


def tables_exist():
    """Checks if tables exist in the database.

    Returns:
        boolean: result of check
    """
    inspection = inspect(db.engine)

    return len(inspection.get_table_names()) > 0


def create_schema():
    """Creates the database schema and adds objects.

    Creates all db tables and inserts data into them.

    Returns:
        None
    """
    db.create_all()

    config_env = configuration.ENVIRONMENT
    if config_env == Environment.TESTING:
        add_sravi_objects('AE_norm')
        add_default_phrase_list('PAVA')
    else:
        if config_env == Environment.RESEARCH:
            add_sravi_objects('AE_norm_2')
        add_default_phrase_list('PAVA-DEFAULT')
    add_default_phrase_list(
        'PAVA-SUB-DEFAULT',
        list_id=configuration.DEFAULT_PAVA_SUB_LIST_ID,
        list_name=configuration.DEFAULT_PAVA_SUB_LIST_NAME,
        list_path=configuration.DEFAULT_SUB_LIST_PATH
    )


def setup(drop=False):
    """Creates and populates the database.

    Creates database schema and inserts objects from filesystem
    Retry connecting every second if failure

    Args:
        drop (boolean): Whether to recreate the database if it already exists

    Returns:
        None
    """
    try:
        if database_exists(configuration.SQLALCHEMY_DATABASE_URI):
            if drop:
                drop_database(configuration.SQLALCHEMY_DATABASE_URI)
                create_database(configuration.SQLALCHEMY_DATABASE_URI)
                create_schema()

            if not tables_exist():
                create_schema()
        else:
            create_database(configuration.SQLALCHEMY_DATABASE_URI)
            create_schema()
    except Exception as e:
        print(e)
        exit()
