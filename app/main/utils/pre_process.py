"""Pre-processing utilities.

Contains logic for pre-processing feature matrix signals
"""
import numpy as np


def clip_start(signal, clip_type=0):
    """Clip the video to where the spoken phrase begins.

    Args:
        signal (numpy array): feature matrix
        clip_type (int): type of clipping to do

    Returns:
        numpy array: clipped signal
    """
    # no reliable automatic way of detecting phrase start
    if clip_type == 0:
        return signal


def compute_delta_matrix(signal, delta_num_frames):
    """Compute time-difference features for each feature vector.

    Uses the number of frames as a window computed from delta

    Args:
        signal (numpy array): original feature matrix
        delta_num_frames (int): time-difference window measured in frames

    Returns:
        numpy array: feature matrix with time difference features
    """
    # 2 * (sum over i = 1..delwin of i**2) = n(n+1)(2n+1)/3
    sigma_t_squared = (delta_num_frames * (delta_num_frames + 1)
                       * (2 * delta_num_frames + 1)) / 3

    delta_matrix = []
    delta_matrix_append = delta_matrix.append
    num_vectors, num_features = signal.shape

    for i in range(num_vectors):
        vector_delta = np.zeros(num_features)
        # append to new feature vector the sum of the time difference between
        # a scanning window of feature vectors depending on the size of the
        # frame window
        for t in range(1, delta_num_frames + 1):
            # indices of feature vectors to use
            low, high = i - t, i + t

            # feature vectors to use
            vector_low = np.zeros(num_features) if low < 0 else signal[low]
            vector_high = np.zeros(num_features) \
                if high >= num_vectors else signal[high]

            # append results of difference between
            vector_delta += t * (vector_high - vector_low)

        # normalise and append vector
        vector_delta /= sigma_t_squared
        delta_matrix_append(vector_delta)

    return np.array(delta_matrix)


def add_deltas_to_signal(signal, delta_1_num_frames, delta_2_num_frames):
    """Calculate time difference features for 2 delta ms windows.

    Append each new feature vector to the previous matrix signal

    Args:
        signal (numpy array):
        delta_1_num_frames (int): num frames window 1
        delta_2_num_frames (int): num frames window 2

    Returns:
        numpy array: feature matrix with new time difference features added
    """
    if delta_1_num_frames == delta_2_num_frames == 0:
        return signal

    delta_1_matrix = compute_delta_matrix(signal=signal,
                                          delta_num_frames=delta_1_num_frames)

    if delta_2_num_frames > 0:
        delta_2_matrix = \
            compute_delta_matrix(signal=signal,
                                 delta_num_frames=delta_2_num_frames)

    # signal_with_deltas = []
    # signal_with_deltas_append = signal_with_deltas.append
    # for i in range(num_vectors):
    #     vector = np.concatenate((signal[i], delta_1_matrix[i]))
    #     if delta_2_num_frames > 0:
    #         vector = np.concatenate((vector, delta_2_matrix[i]))
    #     signal_with_deltas_append(vector)
    signal_with_deltas = np.hstack((signal, delta_1_matrix))
    if delta_2_num_frames > 0:
        signal_with_deltas = np.hstack((signal_with_deltas, delta_2_matrix))

    return signal_with_deltas


def get_number_frames_from_delta(delta, frames_per_second):
    """Calculate the number of frames from a delta window in milli-seconds.

    Args:
        delta (int): frame window in milli-seconds
        frames_per_second (int): video frames per second

    Returns:
        int:
    """
    return int((delta / frames_per_second) + 0.5)


def sub_sample(signal, step=1):
    """Returns numpy matrix x with every step^th feature vector added.

    Args:
        signal (numpy array): original numpy matrix
        step (int): feature vectors at every step vector

    Returns:
        numpy array: sub-sampled matrix
    """
    if step == 1:
        return signal

    num_vectors, num_features = signal.shape

    counter = 0
    sub_sampled_signal = np.zeros((num_vectors // step, num_features))
    for i in range(0, num_vectors, step):
        sub_sampled_signal[i] = signal[counter]
        counter += step

    return sub_sampled_signal


def pre_process_signal(signal, delta_1_num_frames, delta_2_num_frames, step):
    """Pre-process an individual signal.

    Clip the signal to phrase start
    Add time-difference features to each feature vector
    Sub-sample the feature matrix

    Args:
        signal (numpy array): original feature matrix
        delta_1_num_frames (int): num frames window 1
        delta_2_num_frames (int): num frames window 2
        step (int): sub-sample step

    Returns:
        numpy array: pre-processed signal
    """
    signal = clip_start(signal=signal)
    signal = add_deltas_to_signal(signal=signal,
                                  delta_1_num_frames=delta_1_num_frames,
                                  delta_2_num_frames=delta_2_num_frames)
    signal = sub_sample(signal=signal, step=step)

    return signal


def pre_process_signals(signals, **kwargs):
    """Pre-process a number of signals.

    Args:
        signals (list): containing a list of signals to be pre-processed
        **kwargs (dict): containing pre-processing parameters

    Returns:
        list: containing the corresponding pre-processed signals
    """
    delta_1, delta_2 = kwargs['delta_1'], kwargs['delta_2']
    frames_per_second = kwargs['frames_per_second']
    frame_step = kwargs['frame_step']

    delta_1_num_frames = \
        get_number_frames_from_delta(delta=delta_1,
                                     frames_per_second=frames_per_second)
    delta_2_num_frames = \
        get_number_frames_from_delta(delta=delta_2,
                                     frames_per_second=frames_per_second)

    processed_signals = []
    processed_signals_append = processed_signals.append
    for signal in signals:
        signal = pre_process_signal(signal=signal,
                                    delta_1_num_frames=delta_1_num_frames,
                                    delta_2_num_frames=delta_2_num_frames,
                                    step=frame_step)
        processed_signals_append(signal)

    return processed_signals
