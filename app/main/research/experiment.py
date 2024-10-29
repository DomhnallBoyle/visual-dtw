"""R&D - Experiment.

This module provides functionality to compare different user templates and
see the effect of accuracy on increasing users/sessions

Usage Example:
    python experiment.py 207 1,2 207 3 --top_n=22 --phrase_set=SR

See Also:
    python experiment.py --help
"""
import argparse

from main.models import SRAVIPhrase, SRAVITemplate, SRAVIUser
from main.research.cmc import CMC
from main.research.confusion_matrix import ConfusionMatrix
from main.research.utils import csl, experiment, generate_dtw_params
from main.utils.db import setup_db


def parse_args(args):
    """Parse the command line arguments.

    Used to look up actual user and session ids from the database

    Args:
        args (Namespace): containing command line arguments

    Returns:
        Namespace: containing updated command line arguments
    """
    ref_users, ref_sessions = args.ref_users, args.ref_sessions
    test_users, test_sessions = args.test_users, args.test_sessions

    def query_session_ids(users, sessions):
        # if all specified, return all distinct template session ids
        if sessions == 'all':
            sessions = SRAVITemplate.get(
                query=SRAVITemplate.session_number,
                filter=SRAVITemplate.user_id.in_(users),
                distinct=True)
            sessions = [session[0] for session in sessions]

        return sessions

    def query_user_ids(_ref_users, _test_user):
        # if all specified, return all user ids
        if _ref_users == 'all':
            _ref_users = SRAVIUser.get()
            _ref_users = [user.id for user in _ref_users]
            _ref_users = list(set(_ref_users) - set(_test_user))

        return _ref_users

    args.ref_users = query_user_ids(_ref_users=ref_users,
                                    _test_user=test_users)
    args.ref_sessions = query_session_ids(users=ref_users,
                                          sessions=ref_sessions)
    args.test_sessions = query_session_ids(users=test_users,
                                           sessions=test_sessions)
    args.top_n = len(SRAVIPhrase.get(
        filter=SRAVIPhrase.phrase_set == args.phrase_set))

    return args


def main(args):
    """Main entry-point.

    Compare different templates by users and sessions ids
    Plot CMC and confusion matrix graph of results

    Args:
        args (Namespace): command line arguments

    Returns:
        None
    """
    setup_db(drop=args.drop_db)

    args = parse_args(args)
    dtw_params = generate_dtw_params(**args.__dict__)

    ref_users, ref_sessions = args.ref_users, args.ref_sessions
    test_users, test_sessions = args.test_users, args.test_sessions
    phrase_set, feature_type = args.phrase_set, args.feature_type

    print('-' * 100)
    print(f'Reference -> Users: {ref_users}, Sessions: {ref_sessions}')
    print(f'Test -> Users: {test_users}, Sessions: {test_sessions}')
    print('-' * 100)

    cmc = CMC(num_ranks=args.top_n)
    cmc_title = f'CMC: Effect of increasing #{args.increasing_by}' \
                f' on the accuracy\n'

    confusion_matrix = ConfusionMatrix()

    if args.increasing_by == 'users':
        for i in range(len(ref_users)):
            experiment(ref_users[:i + 1], ref_sessions, test_users,
                       test_sessions, phrase_set, feature_type, dtw_params,
                       cmc, confusion_matrix)
            cmc.labels.append(f'# users = {len(ref_users[:i + 1])}')
    elif args.increasing_by == 'sessions':
        for i in range(len(ref_sessions)):
            experiment(ref_users, ref_sessions[:i + 1], test_users,
                       test_sessions, phrase_set, feature_type, dtw_params,
                       cmc, confusion_matrix)
            cmc.labels.append(f'# sessions = {len(ref_sessions[:i + 1])}')
    else:
        experiment(ref_users, ref_sessions, test_users, test_sessions,
                   phrase_set, feature_type, dtw_params, cmc,
                   confusion_matrix)
        cmc_title = 'CMC: Rank Accuracy of test phrases vs ref phrases\n'

    cmc_title += \
        f'Ref: User IDs = {ref_users}, Session IDs = {ref_sessions}\n' \
        f'Test: User IDs = {test_users}, Sessions IDs = {test_sessions}\n' \
        f'Phrase Set = {phrase_set}, Feature Type = {feature_type}'
    cmc.title = cmc_title
    cmc.plot()
    confusion_matrix.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('ref_users', type=csl)
    parser.add_argument('ref_sessions', type=csl)
    parser.add_argument('test_users', type=csl)
    parser.add_argument('test_sessions', type=csl)

    parser.add_argument('--feature_type', default='AE_norm', type=str)
    parser.add_argument('--frame_step', default=1, type=int)
    parser.add_argument('--delta_1', default=100, type=int)
    parser.add_argument('--delta_2', default=0, type=int)
    parser.add_argument('--dtw_top_n_tail', default=0, type=int)
    parser.add_argument('--dtw_transition_cost', default=0.1, type=float)
    parser.add_argument('--dtw_beam_width', default=0, type=int)
    parser.add_argument('--dtw_distance_metric', default='euclidean_squared')
    parser.add_argument('--knn_type', default=2, type=int)
    parser.add_argument('--knn_k', default=50, type=int)
    parser.add_argument('--frames_per_second', default=25, type=int)
    parser.add_argument('--phrase_set', default='S2R', type=str)
    parser.add_argument('--increasing_by', default=None, type=str,
                        choices=['users', 'sessions'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--drop_db', action='store_true')

    main(parser.parse_args())
