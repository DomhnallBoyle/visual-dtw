import main.utils.pre_process as module
import numpy as np
import pytest
from tests.unit.test_utils import matrix_equal

NUM_VECTORS, NUM_FEATURES = 5, 50


def test_clip_start(random_signal):
    signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    expected = signal

    actual = module.clip_start(signal, clip_type=0)

    assert matrix_equal(actual, expected)


@pytest.mark.parametrize(
    'expected, delta_num_frames',
    [
        (np.array([[2.0, 2.5, 3.0],
                   [3.0, 3.0, 3.0],
                   [-2.0, -2.5, -3.0]]), 1),
        (np.array([[1.8, 2.1, 2.4],
                   [0.6, 0.6, 0.6],
                   [-0.6, -0.9, -1.2]]), 2)
    ])
def test_compute_delta_matrix(fixed_signal, expected, delta_num_frames):
    actual = module.compute_delta_matrix(fixed_signal, delta_num_frames)

    assert matrix_equal(actual, expected)


def test_add_deltas_to_signal_no_added_features(mocker, random_signal):
    signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    expected = signal

    mock_compute_matrix = mocker.patch.object(module, 'compute_delta_matrix')

    actual = module.add_deltas_to_signal(signal, 0, 0)

    assert matrix_equal(actual, expected)
    mock_compute_matrix.assert_not_called()


def test_add_deltas_to_signal_only_one_window(mocker, random_signal):
    delta_num_frames = 1
    signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    delta_1_signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    expected = np.concatenate((signal, delta_1_signal), axis=1)

    mock_compute_matrix = mocker.patch.object(module, 'compute_delta_matrix',
                                              return_value=delta_1_signal)

    actual = module.add_deltas_to_signal(signal, delta_num_frames, 0)

    assert matrix_equal(actual, expected)
    assert actual.shape == (NUM_VECTORS, NUM_FEATURES * 2)
    mock_compute_matrix.assert_called_once_with(
        signal=signal, delta_num_frames=delta_num_frames
    )


def test_all_deltas_to_signal_both_windows(mocker, random_signal):
    delta_1_num_frames, delta_2_num_frames = 2, 2
    signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    delta_1_signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    delta_2_signal = random_signal(NUM_VECTORS, NUM_FEATURES)
    expected = np.concatenate((signal, delta_1_signal, delta_2_signal), axis=1)

    mock_compute_matrix = mocker.patch.object(module, 'compute_delta_matrix')
    mock_compute_matrix.side_effect = [delta_1_signal, delta_2_signal]

    actual = module.add_deltas_to_signal(signal, delta_1_num_frames,
                                         delta_2_num_frames)

    assert matrix_equal(actual, expected)
    assert actual.shape == (NUM_VECTORS, NUM_FEATURES * 3)
    mock_compute_matrix.assert_has_calls([
        mocker.call(signal=signal, delta_num_frames=delta_1_num_frames),
        mocker.call(signal=signal, delta_num_frames=delta_2_num_frames)
    ])
