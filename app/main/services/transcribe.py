"""Transcribe service.

Contains logic for transcribing a test signal against reference signals
"""
from main.utils.dtw import DTW


def transcribe_signal(ref_signals, test_signal, classes=None, **dtw_params):
    """Transcribe a test signal against reference signals.

    Args:
        ref_signals (list): contains tuples of (label, numpy array) pairings
        test_signal (numpy array): test feature matrix to compare
        classes (list): all ground-truth classes (optional)
        **dtw_params (dict): containing key-value dtw parameters

    Returns:
        list: containing dictionaries of label-probability pairings
    """
    dtw = DTW(**dtw_params)
    predictions = dtw.classify(test_signal=test_signal,
                               ref_signals=ref_signals,
                               classes=classes,
                               **dtw_params)

    return predictions
