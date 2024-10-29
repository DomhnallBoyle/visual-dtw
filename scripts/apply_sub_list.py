"""
Give every user the new default sub-list who don't have it
"""
import argparse
import os

from main import configuration
from main.models import PAVAList, PAVAUser
from main.models.pava.user import create_default_list
from main.utils.db import db_session, add_default_phrase_list


def apply():
    # add sub-list to the system if it doesn't exist
    if not os.path.exists(configuration.DEFAULT_SUB_LIST_PATH):
        add_default_phrase_list(
            'PAVA-SUB-DEFAULT',
            list_id=configuration.DEFAULT_PAVA_SUB_LIST_ID,
            list_name=configuration.DEFAULT_PAVA_SUB_LIST_NAME,
            list_path=configuration.DEFAULT_SUB_LIST_PATH
        )

    num_affected = 0
    with db_session() as s:
        user_ids = PAVAUser.get(s, query=(PAVAUser.id,))
        print(f'Found {len(user_ids)} users')
        for user_id in user_ids:
            list_names = [name[0] for name in
                          PAVAList.get(s, query=(PAVAList.name,),
                                       filter=(PAVAList.user_id == user_id))]
            if configuration.DEFAULT_PAVA_SUB_LIST_NAME not in list_names:
                create_default_list(
                    user_id=user_id,
                    list_path=configuration.DEFAULT_SUB_LIST_PATH,
                    list_name=configuration.DEFAULT_PAVA_SUB_LIST_NAME
                )
                num_affected += 1
    print(f'Num users affected: {num_affected}/{len(user_ids)}')


def undo():
    num_affected = 0
    with db_session() as s:
        user_ids = PAVAUser.get(s, query=(PAVAUser.id,))
        print(f'Found {len(user_ids)} users')
        for user_id in user_ids:
            lists = PAVAList.get(s, query=(PAVAList.id, PAVAList.name,),
                                 filter=(PAVAList.user_id == user_id))
            for list_id, list_name in lists:
                if list_name == configuration.DEFAULT_PAVA_SUB_LIST_NAME:
                    PAVAList.delete(id=list_id)
                    num_affected += 1
                    break
    print(f'Num users affected: {num_affected}/{len(user_ids)}')

    # remove instances of sub-list from the system
    PAVAList.delete(id=configuration.DEFAULT_PAVA_SUB_LIST_ID)
    os.remove(configuration.DEFAULT_SUB_LIST_PATH)


def main(args):
    f = {
        'apply': apply,
        'undo': undo
    }
    if args.run_type in f:
        f[args.run_type]()
    else:
        print('Choose from:', list(f.keys()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')
    parser_1 = sub_parsers.add_parser('apply')
    parser_2 = sub_parsers.add_parser('undo')

    main(parser.parse_args())
