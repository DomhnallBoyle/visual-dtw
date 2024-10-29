"""
Apply a new list to every user
"""
import argparse

from main import configuration
from main.models import PAVAList, PAVAPhrase, PAVAUser
from main.utils.db import db_session
from main.utils.io import read_json_file

# TODO: Need to create table storing names of out-of-bounds list names


def apply(args):
    # create list for each user (not default)

    # get phrases from external file or json phrases
    all_phrases = read_json_file(configuration.PHRASES_PATH)
    phrases = args.phrases if args.phrases else all_phrases.get(args.name)
    if not phrases:
        print('No phrases...')
        return
    if isinstance(phrases, dict):
        phrases = list(phrases.values())

    num_affected = 0
    with db_session() as s:
        user_ids = PAVAUser.get(s, query=(PAVAUser.id,))
        print(f'Found {len(user_ids)} users')
        for user_id in user_ids:
            list_names = [name[0] for name in
                          PAVAList.get(s, query=(PAVAList.name,),
                                       filter=(PAVAList.user_id == user_id))]
            if args.name not in list_names:
                lst = PAVAList.create(name=args.name, user_id=user_id)
                for phrase in phrases:
                    PAVAPhrase.create(content=phrase, list_id=lst.id)
                num_affected += 1
    print(f'Num users affected: {num_affected}/{len(user_ids)}')


def undo(args):
    num_affected = 0
    with db_session() as s:
        user_ids = PAVAUser.get(s, query=(PAVAUser.id,))
        for user_id in user_ids:
            lists = PAVAList.get(s, query=(PAVAList.id, PAVAList.name,),
                                 filter=(PAVAList.user_id == user_id))
            for list_id, list_name in lists:
                if list_name == args.name:
                    PAVAList.delete(id=list_id)
                    num_affected += 1
                    break
    print(f'Num users affected: {num_affected}/{len(user_ids)}')


def main(args):
    f = {
        'apply': apply,
        'undo': undo
    }
    if args.run_type in f:
        f[args.run_type](args)
    else:
        print('Choose from:', list(f.keys()))


def file_list(s):
    with open(s, 'r') as f:
        return f.read().splitlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('apply')
    parser_1.add_argument('name')
    parser_1.add_argument('--phrases', type=file_list, default=[])

    parser_2 = sub_parsers.add_parser('undo')
    parser_2.add_argument('name')

    main(parser.parse_args())
