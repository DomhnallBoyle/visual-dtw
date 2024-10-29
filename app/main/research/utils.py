"""R&D Utilities.

Contains utility functions for performing R&D
"""
from main.models import SRAVIPhrase, SRAVITemplate
from main.services.transcribe import transcribe_signal
from main.utils.pre_process import pre_process_signals
from tqdm import tqdm


def get_templates(feature_type, users, sessions, phrase_set):
    return SRAVITemplate.get(
        filter=(
            (SRAVITemplate.feature_type == feature_type) &
            (SRAVITemplate.user_id.in_(users)) &
            (SRAVITemplate.phrase.has(phrase_set=phrase_set)) &
            (SRAVITemplate.session_id.in_(sessions))
        )
    )


def experiment(ref_users, ref_sessions, test_users, test_sessions, phrase_set,
               feature_type, dtw_params, cmc, confusion_matrix=None):
    """Begin the experiment process.

    Args:
        ref_users (list): containing reference user ids
        ref_sessions (list): containing reference user session ids
        test_users (list): containing test users ids
        test_sessions (list): containing test user session ids
        phrase_set (string): containing
        feature_type (string): type of features to limit the experiment to
        dtw_params (dict): containing key-value pairings of dtw/knn params
        cmc (CMC): object for constructing CMC curve
        confusion_matrix (ConfusionMatrix): for constructing confusion matrix

    Returns:
        None
    """
    # query reference and test templates from the database
    ref_templates = get_templates(feature_type=feature_type,
                                  users=ref_users,
                                  sessions=ref_sessions,
                                  phrase_set=phrase_set)
    test_templates = get_templates(feature_type=feature_type,
                                   users=test_users,
                                   sessions=test_sessions,
                                   phrase_set=phrase_set)

    # print useful debugging information
    print(f'\nReference -> Users: {ref_users}, Sessions: {ref_sessions}, '
          f'# Templates: {len(ref_templates)}')
    print(f'Test -> Users: {test_users}, Sessions: {test_sessions}, '
          f'# Templates: {len(test_templates)}')

    if not ref_templates or not test_templates:
        print(f'No Ref or Test templates')
        return

    # pre-process reference signals
    ref_signals = pre_process_signals(
        signals=[template.blob for template in ref_templates],
        **dtw_params
    )
    ref_signals = [(template.phrase_id, ref_signal)
                   for template, ref_signal in zip(ref_templates, ref_signals)]

    # get all ground-truth class labels
    classes = [p[0] for p in SRAVIPhrase.get(query=SRAVIPhrase.id,
                                             filter=(SRAVIPhrase.phrase_set ==
                                                     phrase_set))
               ]

    # compare every test template with reference templates
    for test_template in tqdm(test_templates):

        # first pre-process test template
        test_signal = pre_process_signals(
            signals=[test_template.blob], **dtw_params
        )[0]

        # get predictions by transcribing template signal
        predictions = transcribe_signal(ref_signals=ref_signals,
                                        test_signal=test_signal,
                                        classes=classes,
                                        **dtw_params)

        # CMC tally the ranks
        cmc.tally(predictions=[next(iter(d)) for d in predictions],
                  ground_truth=test_template.phrase_id)

        # add the top prediction and actual ground-truth label to the
        # confusion matrix pairings
        if confusion_matrix:
            top_prediction = list(predictions[0].keys())[0]
            confusion_matrix.append(prediction=top_prediction,
                                    ground_truth=test_template.phrase_id)

    # calculate the accuracy from each rank from their tallies
    cmc.calculate_accuracies(num_tests=len(test_templates))


def csl(s):
    """Comma Separated List command-line arguments.

    Args:
        s (string): comma separated list in string format

    Returns:
        list: containing separated CSL elements
    """
    if s.lower() == 'all':
        return s

    _csl = s.split(',')
    _csl_copy = []

    for i in range(len(_csl)):
        if '-' in _csl[i]:
            start, end = _csl[i].split('-')
            _csl_copy.extend([j for j in range(int(start), int(end) + 1)])
        else:
            _csl_copy.append(int(_csl[i]))

    return _csl_copy


# TODO: Remove this
def generate_dtw_params(frame_step=1,
                        delta_1=100,
                        delta_2=0,
                        dtw_top_n_tail=0,
                        dtw_transition_cost=0.1,
                        dtw_beam_width=0,
                        dtw_distance_metric='euclidean_squared',
                        knn_type=2,
                        knn_k=50,
                        frames_per_second=25,
                        top_n=3, **kwargs):
    """Generate a dictionary of parameters relating to DTW/KNN only.

    Args:
        frame_step (int): frame sub-sample step
        delta_1 (int): time-difference features window 1 ms
        delta_2 (int): time-difference features window 2 ms
        dtw_top_n_tail (int):
        dtw_transition_cost (float):
        dtw_beam_width (int):
        dtw_distance_metric (string):
        knn_type (int):
        knn_k (int):
        frames_per_second (int):
        top_n (int):
        **kwargs (dict): other params not required

    Returns:
        dict: containing key-value DTW/KNN params
    """
    return {
        'frame_step': frame_step,
        'delta_1': delta_1,
        'delta_2': delta_2,
        'dtw_top_n_tail': dtw_top_n_tail,
        'dtw_transition_cost': dtw_transition_cost,
        'dtw_beam_width': dtw_beam_width,
        'dtw_distance_metric': dtw_distance_metric,
        'knn_type': knn_type,
        'knn_k': knn_k,
        'frames_per_second': frames_per_second,
        'top_n': top_n
    }
