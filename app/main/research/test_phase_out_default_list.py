import os
import random
import re
import time

from main import configuration
from main.models import Config, PAVAList, PAVAPhrase, PAVATemplate, PAVAUser
from main.research.cmc import CMC
from main.services.transcribe import transcribe_signal
from main.utils.cfe import run_cfe
from main.utils.dtw import DTW
from main.utils.io import read_json_file, read_pickle_file
from main.utils.pre_process import pre_process_signals

NEW_RECORDINGS_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'

USER = PAVAUser.create(default_list=True, config=Config())
USER_LIST_ID = PAVAList.get(
    query=(PAVAList.id,),
    filter=(PAVAList.user_id == USER.id),
    first=True
)[0]


def get_phrase_videos(videos_path):
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']
    phrase_videos = {k: [] for k in pava_phrases.keys()}

    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        video = os.path.join(videos_path, video)

        videos = phrase_videos.get(phrase_id)
        videos.append((pava_phrases[phrase_id], video))
        phrase_videos[phrase_id] = videos

    return {int(k): v for k, v in phrase_videos.items()}


def test(video_path, lst):
    dtw_params = USER.config.__dict__
    try:
        with open(video_path, 'rb') as f:
            test_feature_matrix = run_cfe(f)

        test_feature_matrix = \
            pre_process_signals([test_feature_matrix], **dtw_params)[0]

        ref_signals = []
        for phrase in lst.phrases:
            for template in phrase.templates:
                ref_signals.append((phrase.content, template.blob))

        return transcribe_signal(ref_signals,
                                 test_feature_matrix,
                                 None, **dtw_params)
    except Exception:
        return None


def test_vs_default_list(video_path):
    default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)

    return test(video_path, default_lst)


def test_vs_user_list(video_path):
    lst = PAVAList.get(filter=(PAVAList.user_id == USER.id), first=True)

    return test(video_path, lst)


def get_phrase_coverage():
    # coverage over completed phrases
    num_completed_phrases = 0
    num_phrases = 20
    lst = PAVAList.get(filter=(PAVAList.user_id == USER.id), first=True)

    for phrase in lst.phrases:
        num_templates = len(phrase.templates)
        if num_templates == 5:
            num_completed_phrases += 1

    return num_completed_phrases / num_phrases


def get_session_coverage():
    # coverage over complete sessions
    lst = PAVAList.get(filter=(PAVAList.user_id == USER.id), first=True)

    max_sessions = 5
    num_templates = 0
    num_phrases = 20

    for phrase in lst.phrases:
        num_templates += len(phrase.templates)

    num_sessions = num_templates // num_phrases

    return num_sessions / max_sessions


def get_template_coverage():
    # coverage over number of templates
    lst = PAVAList.get(filter=(PAVAList.user_id == USER.id), first=True)

    max_templates = 100  # 20 phrases * 5 sessions
    num_templates = 0
    for phrase in lst.phrases:
        num_templates += len(phrase.templates)

    return num_templates / max_templates


def combine_predictions(default_predictions, user_predictions, cov_weight):
    if not default_predictions and user_predictions:
        return user_predictions

    if not user_predictions and default_predictions:
        return default_predictions

    if not default_predictions and not user_predictions:
        return None

    def weight(predictions, coverage):
        for prediction in predictions:
            prediction['accuracy'] *= coverage

        return predictions

    default_predictions = weight(default_predictions, 1 - cov_weight)
    user_predictions = weight(user_predictions, cov_weight)

    def append(voting, prediction):
        label = prediction['label']
        accuracies = voting.get(label, [])
        accuracies.append(prediction['accuracy'])
        voting[label] = accuracies

        return voting

    voting = {}
    for prediction_a, prediction_b in \
            zip(default_predictions, user_predictions):
        voting = append(voting, prediction_a)
        voting = append(voting, prediction_b)

    # sum accuracies
    voting = {k: sum(v) for k, v in voting.items()}

    # sort and limit to 3
    predictions = sorted(voting.items(), key=lambda x: x[1], reverse=True)[:3]
    for i in range(len(predictions)):
        prediction = {
            'label': predictions[i][0],
            'accuracy': predictions[i][1]
        }
        predictions[i] = prediction

    return predictions


def test_both(video_path, cov_weight):
    default_predictions = test_vs_default_list(video_path=video_path)
    user_predictions = test_vs_user_list(video_path=video_path)

    return combine_predictions(default_predictions, user_predictions,
                               cov_weight)


def make_predictions(video_path):
    session_coverage = get_session_coverage()
    phrase_coverage = get_phrase_coverage()
    template_coverage = get_template_coverage()

    # REMEMBER: default predictions given weight of 1 - cov_weight

    if session_coverage == 1:
        return test_vs_user_list(video_path=video_path)

    print(session_coverage, phrase_coverage)
    if phrase_coverage >= session_coverage:
        # user uses more phrases than has complete sessions
        if phrase_coverage > 0.7 and session_coverage > 0.5:
            # 3 complete sessions
            return test_vs_user_list(video_path=video_path)
        elif phrase_coverage > 0.7 and session_coverage < 0.5:
            # <= 2 complete sessions
            return test_both(video_path=video_path,
                                    cov_weight=phrase_coverage)
        elif phrase_coverage > 0.5 and session_coverage >= 0.2:
            return test_both(video_path=video_path,
                                    cov_weight=session_coverage)
        else:
            return test_vs_default_list(video_path=video_path)

    if session_coverage > phrase_coverage:
        if session_coverage >= 0.8 and phrase_coverage >= 0.5:
            return test_vs_user_list(test)
        elif session_coverage >= 0.8 and phrase_coverage < 0.5:
            return test_both(video_path=video_path,
                                    cov_weight=session_coverage)
        elif session_coverage >= 0.4 and phrase_coverage >= 0.2:
            return test_both(video_path=video_path,
                                    cov_weight=phrase_coverage)
        else:
            return test_vs_default_list(video_path=video_path)

    # print('Coverage: ', phrase_coverage)
    #
    # if phrase_coverage < 0.2:
    #     print('Comparing against default')
    #     predictions = test_vs_default_list(video_path=video_path)
    # elif phrase_coverage < 1:
    #     print('Comparing against both')
    #     predictions1 = test_vs_default_list(video_path=video_path)
    #     predictions2 = test_vs_user_list(video_path=video_path)
    #     predictions = combine_predictions(predictions1, predictions2,
    #                                       phrase_coverage)
    # else:
    #     print('Comparing against user')
    #     predictions = test_vs_user_list(video_path=video_path)
    #
    # return predictions


def create_template(video_path, phrase_id):
    dtw_params = USER.config.__dict__

    with open(video_path, 'rb') as f:
        feature_matrix = run_cfe(f)

    feature_matrix = \
        pre_process_signals([feature_matrix], **dtw_params)[0]

    PAVATemplate.create(blob=feature_matrix, phrase_id=phrase_id)


def rank_templates(pava_phrase):
    # clustering algorithm

    dtw = DTW(**Config().__dict__)
    templates = pava_phrase.templates

    # get average dtw distance between all templates of the phrase
    averages = []
    for i in range(0, len(templates)):
        average_distance = 0
        for j in range(0, len(templates)):
            if i == j:
                continue

            template1 = templates[i].blob
            template2 = templates[j].blob

            distance = dtw.calculate_distance(
                test_signal=template1,
                ref_signal=template2)
            average_distance += distance

        average_distance /= (len(templates) - 1)
        averages.append(average_distance)

    # rank them
    assert len(averages) == len(templates)

    # replace highest
    max_index = averages.index(max(averages))
    # del templates[min_index]

    template_to_remove = templates[max_index]
    PAVATemplate.delete(id=template_to_remove.id)


def add_to_list(predictions, actual_label, video_path):
    for prediction in predictions:
        if prediction['label'] == actual_label:
            pava_phrase = PAVAPhrase.get(
                filter=(
                    (PAVAPhrase.content == actual_label) &
                    (PAVAPhrase.list_id == USER_LIST_ID)
                ),
                first=True
            )

            create_template(video_path, pava_phrase.id)
            if len(pava_phrase.templates) < 5:
                return True
            else:
                rank_templates(pava_phrase)

    return False


def train_test_phrase_split(videos_path, test_sessions):
    """Split videos into training and test"""
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    # split unseen user templates into sessions
    phrases_to_add = {}
    test_videos = []
    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        video = os.path.join(videos_path, video)

        phrase = pava_phrases[phrase_id]
        if int(session_id) in test_sessions:
            test_videos.append((phrase, video))
        else:
            phrase_videos = phrases_to_add.get(int(phrase_id), [])
            phrase_videos.append((phrase, video))
            phrases_to_add[int(phrase_id)] = phrase_videos

    return phrases_to_add, test_videos


def train_test_session_split(videos_path, session_ids_to_add,
                             test_session_ids):
    pava_phrases = read_json_file(configuration.PHRASES_PATH)['PAVA-DEFAULT']

    sessions_to_add = {}
    test_videos = []
    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()

        session_id = int(session_id)

        video = os.path.join(videos_path, video)
        phrase = pava_phrases[phrase_id]
        if session_id in test_session_ids:
            test_videos.append((phrase, video))
        elif session_id in session_ids_to_add:
            session_videos = sessions_to_add.get(session_id, [])
            session_videos.append((phrase, video))
            sessions_to_add[session_id] = session_videos

    return sessions_to_add, test_videos


def experiment_1():
    """
    Check what the prediction accuracies are like for common vs common
    and unseen vs common on a user list where the user only uses common
    phrases
    """
    videos_path = '/home/domhnall/Documents/sravi_dataset/liopa/12'
    phrases_to_add, test_videos = train_test_session_split(videos_path, [8, 9])

    # now add most common phrases
    most_common_phrases = [1, 7, 10]
    num_added = 0
    for phrase_id, videos in phrases_to_add.items():
        if phrase_id in most_common_phrases:
            for label, video_path in videos:
                add_to_list([{'label': label}], label, video_path)
                num_added += 1

    assert num_added == 21

    output_file = 'experiment_1.csv'

    def vs_common(phrases):
        print(phrases)
        cmc = CMC(num_ranks=3)
        for test_label, test_video in test_videos:
            file_name = test_video.split('/')[-1]
            user_id, phrase_id, session_id = \
                re.match(NEW_RECORDINGS_REGEX, file_name).groups()
            if int(phrase_id) in phrases:
                predictions = test_vs_user_list(test_video)
                if predictions:
                    with open(output_file, 'a') as f:
                        f.write(f'{phrases},{predictions},{test_label}\n')
                    predictions = [prediction['label']
                                   for prediction in predictions]
                    cmc.tally(predictions, test_label)
        cmc.calculate_accuracies(len(test_videos), count_check=False)

        return cmc.all_rank_accuracies[0]

    # common vs common
    rank_accuracies_1 = vs_common(most_common_phrases)

    # uncommon vs common
    least_common_phrases = \
        list(set(phrases_to_add.keys()) - set(most_common_phrases))
    rank_accuracies_2 = \
        vs_common(least_common_phrases)

    print('Num test videos: ', len(test_videos))
    print(most_common_phrases, rank_accuracies_1)
    print(least_common_phrases, rank_accuracies_2)


def get_session_ids(videos_path):
    """Get unique session ids from videos path"""
    # get all sessions
    session_ids = set()
    for video in os.listdir(videos_path):
        user_id, phrase_id, session_id = \
            re.match(NEW_RECORDINGS_REGEX, video).groups()
        session_ids.add(int(session_id))

    return list(session_ids)


def run_tests(test_videos, coverage, output_file):
    cmc = CMC(num_ranks=3)
    for test_label, test_video in test_videos:
        predictions = make_predictions(test_video)
        if predictions:
            predictions = [prediction['label']
                           for prediction in predictions]
            cmc.tally(predictions, test_label)

    cmc.calculate_accuracies(num_tests=len(test_videos),
                             count_check=False)

    with open(output_file, 'a') as f:
        f.write(f'{coverage},{cmc.all_rank_accuracies[0]}\n')


def increasing_session_coverage(videos_path):
    print('Increasing session coverage')
    session_ids = get_session_ids(videos_path)
    sampled_ids = random.sample(session_ids, 7)

    session_ids_to_add = sampled_ids[:-2]
    test_sessions = sampled_ids[-2:]

    sessions_to_add, test_videos = train_test_session_split(videos_path,
                                                            session_ids_to_add,
                                                            test_sessions)

    for session_id, videos in sessions_to_add.items():
        for label, video_path in videos:
            add_to_list([{'label': label}], label, video_path)

        # complete session added, let's run tests
        session_coverage = get_session_coverage()
        run_tests(test_videos, session_coverage, 'increasing_session_cov.csv')

    session_coverage = get_session_coverage()
    run_tests(test_videos, session_coverage, 'increasing_session_cov.csv')


def increasing_phrase_coverage(videos_path):
    print('Increasing phrase coverage')
    session_ids = get_session_ids(videos_path)
    test_sessions = random.sample(session_ids, 2)

    phrases_to_add, test_videos = train_test_phrase_split(videos_path,
                                                          test_sessions)

    for phrase_id, videos in phrases_to_add.items():
        for label, video_path in videos:
            add_to_list([{'label': label}], label, video_path)

        # complete phrase, let's run tests
        phrase_coverage = get_phrase_coverage()
        run_tests(test_videos, phrase_coverage, 'increasing_phrase_cov.csv')


def increasing_session_and_phrase(videos_path):
    """Add session, add phrases, add session ..."""
    print('Increasing session phrase coverage')
    session_ids = get_session_ids(videos_path)
    test_sessions = random.sample(session_ids, 2)

    sessions_to_add, test_videos = train_test_session_split(videos_path,
                                                            test_sessions)

    add_session = True
    for session_id, videos in sessions_to_add.items():
        if add_session:
            for label, video_path in videos:
                add_to_list([{'label': label}], label, video_path)

            # complete session added, let's run tests
            session_coverage = get_session_coverage()
            run_tests(test_videos, session_coverage,
                      'increasing_session_template_cov.csv')
            add_session = False
        else:
            for label, video_path in videos:
                add_to_list([{'label': label}], label, video_path)
                template_coverage = get_template_coverage()
                run_tests(test_videos, template_coverage,
                          'increasing_session_template_cov.csv')

            add_session = True


def increasing_phrase_half_session(videos_path):
    session_ids = get_session_ids(videos_path)
    test_sessions = random.sample(session_ids, 2)


def main():
    videos_path = '/home/domhnall/Documents/sravi_dataset/liopa/12'
    phrases_to_add, test_videos = train_test_session_split(videos_path,
                                                           [8, 9])

    output_file = 'coverage_accuracies.csv'
    if os.path.exists(output_file):
        os.remove(output_file)

    # now add most common phrases
    print('Adding most common phrases')
    most_common_phrases = [1, 7, 10]
    num_added = 0
    for phrase_id, videos in phrases_to_add.items():
        if phrase_id in most_common_phrases:
            for label, video_path in videos:
                add_to_list([{'label': label}], label, video_path)
                num_added += 1

    assert num_added == 21

    # add remaining phrases, testing with test videos each time
    print('Adding rest of phrases')
    num_videos_per_phrase = len(phrases_to_add[2])
    tested_coverages = []
    for i in range(num_videos_per_phrase):
        for phrase_id, videos in phrases_to_add.items():
            if phrase_id not in most_common_phrases:
                label, video_path = videos[i]
                predictions = make_predictions(video_path)
                if predictions:
                    add_to_list(predictions, label, video_path)

                # run tests after every 10% coverage increase
                coverage = get_template_coverage()
                if (coverage * 100) % 10 == 0 \
                        and coverage not in tested_coverages:
                    print('Running tests: ', coverage)
                    cmc = CMC(num_ranks=3)
                    for test_label, test_video in test_videos:
                        predictions = make_predictions(test_video)
                        if predictions:
                            predictions = [prediction['label']
                                           for prediction in predictions]
                            cmc.tally(predictions, test_label)

                    cmc.calculate_accuracies(num_tests=len(test_videos),
                                             count_check=False)
                    tested_coverages.append(coverage)

                    with open(output_file, 'a') as f:
                        f.write(f'{coverage},{cmc.all_rank_accuracies[0]}\n')


if __name__ == '__main__':
    # main()
    videos_path = '/home/domhnall/Documents/sravi_dataset/liopa/12'

    increasing_session_coverage(videos_path)
    # increasing_phrase_coverage(videos_path)
    # increasing_session_and_phrase(videos_path)
