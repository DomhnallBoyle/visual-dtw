"""Testing the time it takes the session selection process to finish with increasing phrases and sessions
Use the Speech Ocean data for this - 20 sessions, 70 phrases per user
"""
import argparse
import random
import time

from main.research.research_utils import sessions_to_templates
from main.utils.io import read_pickle_file, write_pickle_file
from scripts.session_selection import MINIMUM_NUM_SESSIONS_TO_ADD, \
    session_selection_with_cross_validation, SESSION_SPLIT

NUM_PHRASES = 70
NUM_SESSIONS = 20
MIN_NUM_PHRASES = 5
MIN_NUM_SESSIONS = 5

random.seed(2021)


def main(args):
    sessions = read_pickle_file(args.sessions_path)
    assert len(sessions) == NUM_SESSIONS
    assert all([len(s[1]) == NUM_PHRASES for s in sessions])

    random.shuffle(sessions)

    # sort phrases in the right order for every session
    for i in range(len(sessions)):
        session_templates = sessions[i][1]
        session_templates = sorted(session_templates, key=lambda x: x[0])
        sessions[i] = (sessions[i][0], session_templates)

    session_times = []
    for num_sessions in range(MIN_NUM_SESSIONS, NUM_SESSIONS+1, MIN_NUM_SESSIONS):

        phrase_times = []
        for num_phrases in range(MIN_NUM_PHRASES, NUM_PHRASES+1, MIN_NUM_PHRASES):
            print(f'Num sessions: {num_sessions}, Num Phrases: {num_phrases}...', end='')

            # select the correct number of phrases from the sessions
            sessions_this_round = []
            for session_id, session_templates in sessions[:num_sessions]:
                sessions_this_round.append((
                    session_id,
                    session_templates[:num_phrases]
                ))
            assert len(sessions_this_round) == num_sessions
            assert all([len(s[1]) == num_phrases for s in sessions_this_round])

            random.shuffle(sessions_this_round)
            split = int(len(sessions_this_round) * SESSION_SPLIT)
            sessions_to_add = sessions_this_round[:split]
            training_sessions = sessions_this_round[split:]
            training_templates = sessions_to_templates(training_sessions)

            start_time = time.time()
            session_selection_with_cross_validation(
                _sessions=sessions_to_add,
                _training_templates=training_templates,
                initial_max=MINIMUM_NUM_SESSIONS_TO_ADD
            )
            took = time.time() - start_time  # in seconds
            print(f'Took {took} seconds')

            phrase_times.append(took)

        session_times.append(phrase_times)

    print(session_times)
    write_pickle_file(session_times, 'session_selection_times.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sessions_path')

    main(parser.parse_args())
