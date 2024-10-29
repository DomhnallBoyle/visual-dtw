"""
Convert the default list in the database to use sessions for it's templates
This is apart of the migration to using models and model sessions
This is tested locally and 13 sessions are created
"""
from main import configuration
from main.models import PAVAList, PAVASession, PAVATemplate
from main.utils.db import db_session
from sqlalchemy.orm import joinedload


def main():
    with db_session() as s:
        default_list = \
            PAVAList.get(
                s, loading_options=(joinedload('phrases.templates')),
                filter=(PAVAList.id == configuration.DEFAULT_PAVA_LIST_ID),
                first=True)

    phrase_templates = {}
    for phrase in default_list.phrases:
        if not phrase.is_nota:
            phrase_templates[phrase.content] = phrase.templates

    num_templates = sum([len(v) for k, v in phrase_templates.items()])
    num_sessions = num_templates // 20
    print('Num default sessions: ', num_sessions)

    for i in range(num_sessions):
        # create sessions for the default list
        session = PAVASession.create(
            list_id=configuration.DEFAULT_PAVA_LIST_ID
        )

        for phrase_content, templates in phrase_templates.items():
            # update the templates session id in the same way we organised
            # them in our update sessions algorithm
            PAVATemplate.update(
                id=templates[i].id,
                session_id=session.id
            )


if __name__ == '__main__':
    main()
