"""
Create sessions manually from data captured videos
This is apart of the migration to using models and model sessions
This script takes captured sessions (from the data capture app)
from internal users and adds them to the database
"""
import argparse

from main import configuration
from main.models import PAVASession, PAVATemplate, PAVAUser, PAVAList
from main.research.test_update_list_3 import get_user_sessions
from main.utils.db import db_session


def main(args):
    user_sessions = get_user_sessions(args.videos_directory)

    # if user already exists in the database, use them
    user_id = args.user_id
    if not user_id:
        user_id = PAVAUser.create(default_list=True).id

    # get default list
    with db_session() as s:
        lst = PAVAList.get(
            s,
            filter=((PAVAList.user_id == user_id)
                    & (PAVAList.name == configuration.DEFAULT_PAVA_LIST_NAME)),
            first=True
        )

    phrase_lookup = {
        phrase.content: phrase.id
        for phrase in lst.phrases
    }

    num_created_sessions = 0

    # create sessions
    for session_name, session_templates in user_sessions:
        # # make sure there are 20 templates
        # if len(session_templates) != 20:
        #     continue

        session = PAVASession.create(
            list_id=lst.id,
            new=True
        )

        for phrase_content, template in session_templates:
            PAVATemplate.create(
                session_id=session.id,
                phrase_id=phrase_lookup[phrase_content],
                blob=template.blob
            )

        num_created_sessions += 1
        if num_created_sessions == args.num_sessions_to_create:
            break

    print('User ID:', user_id)
    print('Num created sessions:', num_created_sessions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')
    parser.add_argument('--user_id')
    parser.add_argument('--num_sessions_to_create', type=int)

    main(parser.parse_args())
