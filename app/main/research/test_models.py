import argparse

from main.models import PAVAModelSession, PAVAList, PAVAPhrase, PAVATemplate, \
    PAVASession
from main.utils.db import db_session
from sqlalchemy.orm import joinedload


def main(args):
    with db_session() as s:
        lst = PAVAList.get(s, filter=(PAVAList.id == args.list_id))
        user_models = lst.models

        # grab all list templates
        test_templates = PAVATemplate.get(
            s, filter=(
                (PAVATemplate.phrase_id == PAVAPhrase.id)
                & (PAVAPhrase.list_id == args.list_id)
            ))

    for model in user_models:
        with db_session() as s:
            model_templates = PAVATemplate.get(
                s, filter=(
                    (PAVATemplate.session_id == PAVAModelSession.session_id)
                    & (PAVAModelSession.model_id == model.id)
                )
            )

            ref_signals = [
                (template.phrase.content, template.blob)
                for template in model_templates
            ]

        print(len(ref_signals))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('list_id')

    main(parser.parse_args())
