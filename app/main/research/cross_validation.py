"""R&D - Cross validation.

This module provides functionality to run cross-validation on a particular
users session ids

Usage Example:
    python cross_validation.py 1 --num_folds=3 --top_n=21

See Also:
    python cross_validation.py --help
"""
import argparse

import numpy as np
from main.models import SRAVIPhrase, SRAVITemplate, SRAVIUser
from main.research.cmc import CMC
from main.research.utils import csl, experiment, generate_dtw_params
from main.utils import db
from sklearn.model_selection import KFold


def query_user_ids(_users):
    """Query user ids from the database.

    Args:
        _users (list/string): containing original user ids

    Returns:
        list: containing user ids to run cross validation on
    """
    # if all user specified, get all user ids from the database
    if _users == 'all':
        _users = [_id[0] for _id in SRAVIUser.get(query=SRAVIUser.id)]

    return _users


def query_session_ids(user_id, feature_type, phrase_set):
    """Query session ids from database by user.

    Args:
        user_id (int): user id to query
        feature_type (string): template features to use
        phrase_set (string): template phrase set to use

    Returns:
        list: containing sessions ids for a particular user
    """
    f = (
        (SRAVITemplate.user_id == user_id) &
        (SRAVITemplate.feature_type == feature_type) &
        (SRAVITemplate.phrase.has(phrase_set=phrase_set))
    )

    return np.array([
        t[0]
        for t in SRAVITemplate.get(query=SRAVITemplate.session_number,
                                   filter=f,
                                   distinct=True)])


def main(args):
    """Main entry-point.

    Parse command line arguments, getting user ids
    For each user, query session ids
    Run K-fold cross-validation on user templates split by session ids
    Plot CMC curves for each reference/testing fold pairing
    Plot average CMC curve over all fold pairings

    Args:
        args (Namespace): containing command line arguments

    Returns:
        None
    """
    db.setup(drop=args.drop_db)

    user_ids = query_user_ids(_users=args.users)
    phrase_set, feature_type = args.phrase_set, args.feature_type
    num_folds = args.num_folds

    args.top_n = len(SRAVIPhrase.get(
        filter=SRAVIPhrase.phrase_set == phrase_set))
    dtw_params = generate_dtw_params(**args.__dict__)

    for user_id in user_ids:
        session_ids = query_session_ids(user_id=user_id,
                                        feature_type=feature_type,
                                        phrase_set=phrase_set)

        if len(session_ids) == 0 or len(session_ids) < num_folds:
            print(f'No sessions for user: {user_id}')
            continue
        else:
            print('-' * 100)
            print(f'User: {user_id}\nSessions: {session_ids}')
            print('-' * 100)

        cmc = CMC(num_ranks=args.top_n)

        # extract reference and test session ids from each fold
        k_fold = KFold(n_splits=num_folds)
        for train_index, test_index in k_fold.split(session_ids):
            ref_sessions = session_ids[train_index].tolist()
            test_sessions = session_ids[test_index].tolist()

            # run experiments with this users reference and test sessions
            experiment([user_id], ref_sessions, [user_id], test_sessions,
                       phrase_set, feature_type, dtw_params, cmc)

            cmc.sub_titles.append(f'Ref sessions: {ref_sessions}\n'
                                  f'Test sessions: {test_sessions}')
        cmc.title = f'{num_folds}-fold cross-validation of sessions for ' \
                    f'user {user_id}'
        cmc.sub_plots()

        # calculate average performance for ranks over # folds
        rank_averages = []
        for i in range(args.top_n):
            rank_average = sum(cmc.all_rank_accuracies[j][i]
                               for j in range(num_folds))
            rank_average /= num_folds
            rank_averages.append(rank_average)

        # plot average performance
        cmc = CMC(num_ranks=args.top_n)
        cmc.all_rank_accuracies = [rank_averages]
        cmc.title = f'Average rank accuracy after {num_folds} folds of ' \
                    f'cross-validation for user {user_id}'
        cmc.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('users', type=csl)

    parser.add_argument('--num_folds', default=5, type=int)
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
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--drop_db', action='store_true')

    main(parser.parse_args())
