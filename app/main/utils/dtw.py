"""DTW - Dynamic Time Warping.

Contains a code implementation for Dynamic Time Warping
"""
import queue
# import cupy as cp
# from numba import jit
from threading import Thread

import numpy as np

# from .calculate_cost_matrix import ccm
from .compute_dtw_distance import compute_dtw_distance
from .knn import KNN


# @jit(nopython=True)
# def ccm(test_signal, ref_signal, distance_metric):
#     num_test_vectors, num_test_features = test_signal.shape
#     num_ref_vectors, num_ref_features = ref_signal.shape
#
#     # TODO: Implement other metrics or just euclidean distance?
#     # calculate the euclidean distance matrix between each pair of feature
#     # vectors in X and Y
#     # euclidean distance = sqrt((x - y)^2) == sqrt((x^2 - 2xy + y))
#     xy = np.dot(test_signal, ref_signal.T)
#
#     x_squared = np.square(test_signal).sum(axis=1)
#     y_squared = np.square(ref_signal).sum(axis=1)
#     cost_matrix = (x_squared + (-2 * xy + y_squared).T).T
#
#     if distance_metric != 'euclidean_squared':
#         # don't sqrt if using square of euclidean distance
#         cost_matrix = np.sqrt(cost_matrix)
#
#     # normalise with the number of vector features
#     cost_matrix /= num_test_features
#
#     cost_matrix = cost_matrix.astype(np.float64)
#
#     return cost_matrix


# def ccm_cupy(test_signal, ref_signal, distance_metric):
#     num_test_vectors, num_test_features = test_signal.shape
#
#     xy = cp.dot(test_signal, ref_signal.T)
#
#     x_squared = cp.square(test_signal).sum(axis=1)
#     y_squared = cp.square(ref_signal).sum(axis=1)
#     cost_matrix = cp.add(x_squared,
#                          cp.add(cp.multiply(xy, -2), y_squared).T).T
#
#     # cost_matrix = (x_squared + (-2 * xy + y_squared).T).T
#
#     if distance_metric != 'euclidean_squared':
#         # don't sqrt if using square of euclidean distance
#         cost_matrix = cp.sqrt(cost_matrix)
#
#     # normalise with the number of vector features
#     cost_matrix = cp.true_divide(cost_matrix, num_test_features)
#
#     cost_matrix = cp.asnumpy(cost_matrix).astype(np.float64)
#
#     return cost_matrix


class DTW:
    """DTW class.

    DTW is used to calculate a distance from reference to test signal.
    KNN is then used to classify the test signal from the reference signals
    using these distances.
    See https://nipunbatra.github.io/blog/2014/dtw.html for the DTW algorithm

    Attributes:
        beam_width (int):
        transition_cost (float):
        top_n_tailing (int):
        distance_metric (string):
        max_num_vectors (int):
        version (string):
        find_path (boolean):
    """

    def __init__(self, **kwargs):
        """Constructor.

        Args:
            **kwargs (dict): contains DTW parameters
        """
        self.beam_width = kwargs.get('dtw_beam_width', 0)
        self.transition_cost = kwargs.get('dtw_transition_cost',
                                          np.float32(0.1))
        self.top_n_tailing = kwargs.get('dtw_top_n_tail', 1)
        self.distance_metric = kwargs.get('dtw_distance_metric', 'euclidean')
        self.max_num_vectors = kwargs.get('max_num_vectors', 1000)
        self.version = kwargs.get('version', 'cython')
        self.find_path = kwargs.get('find_path', False)

    def classify(self, test_signal, ref_signals, classes, **kwargs):
        """Make a prediction using DTW and KNN.

        Calculate DTW distances between references and test signal
        Apply KNN to classify the test signal using the DTW distances
        Return the top_n predictions

        Args:
            test_signal (numpy array): feature matrix
            ref_signals (list): containing reference signal numpy arrays
            classes (list): all ground-truth classes (optional)
            **kwargs (dict): key-value pairings containing DTW/KNN parameters

        Returns:
            list: containing top_n dictionaries of label-probability pairings
        """
        knn = KNN(**kwargs)

        test_signal = test_signal.astype(np.float32)

        distances, labels = [], []

        if kwargs.get('is_dtw_concurrent', False):
            if len(ref_signals) >= 100:

                # use threads to calculate distances concurrently
                num_worker_threads = 2
                num_threads = num_worker_threads + 1

                # divide reference signals between threads
                count_ar = np.linspace(0, len(ref_signals), num_threads + 1,
                                       dtype=int)
                ref_signals_list = []
                temp_list = []
                i = 1
                for entry in ref_signals:
                    temp_list.append(entry)
                    if i in count_ar:
                        ref_signals_list.append(temp_list)
                        temp_list = []
                    i += 1

                distance_label_queue = queue.Queue()

                def calculate_distances(_ref_signals, _queue):
                    for ref_label, ref_signal in _ref_signals:
                        distance = self.calculate_distance(
                            test_signal=test_signal,
                            ref_signal=ref_signal.astype(np.float32))
                        _queue.put((ref_label, distance))

                threads = []
                for i in range(num_worker_threads):
                    thread = Thread(target=calculate_distances,
                                    args=(ref_signals_list[i],
                                          distance_label_queue))
                    thread.start()
                    threads.append(thread)

                calculate_distances(_ref_signals=ref_signals_list[-1],
                                    _queue=distance_label_queue)

                for i in range(num_worker_threads):
                    threads[i].join()  # main wait until workers finished

                while True:
                    try:
                        label, distance = distance_label_queue.get_nowait()
                        labels.append(label)
                        distances.append(distance)
                    except queue.Empty:
                        break
            else:
                labels = [ref[0] for ref in ref_signals]
                distances = [
                    self.calculate_distance(test_signal,
                                            ref_signal[1].astype(np.float32))
                    for ref_signal in ref_signals
                ]
        else:
            # without threads
            labels = [ref[0] for ref in ref_signals]
            distances = [
                self.calculate_distance(test_signal,
                                        ref_signal[1].astype(np.float32))
                for ref_signal in ref_signals
            ]

        knn.fit(distances=np.array(distances), labels=np.array(labels))

        return knn.predict(classes=classes, **kwargs)

    def calculate_distance(self, test_signal, ref_signal):
        cost_matrix = \
            self._calculate_cost_matrix(test_signal=test_signal,
                                        ref_signal=ref_signal)
        path, distance = \
            self._calculate_path_and_distance(cost_matrix=cost_matrix)[:2]

        return distance

    def _calculate_cost_matrix(self, test_signal, ref_signal):
        """DTW Part 1.

        Calculate the cost matrix between a reference and test signal
        Each pair of feature vectors in X and Y are compared
        Distance metrics include:
            - Euclidean
            - Euclidean squared

        Args:
            test_signal (numpy array): test feature matrix
            ref_signal (numpy array): reference feature matrix

        Returns:
            numpy array: num_x_vectors * num_y_vectors cost matrix
        """
        num_test_vectors, num_test_features = test_signal.shape
        num_ref_vectors, num_ref_features = ref_signal.shape

        # make sure signal durations don't exceed limit
        assert max(num_test_vectors, num_ref_vectors) < self.max_num_vectors, \
            f'{max(num_test_vectors, num_ref_vectors)} > ' \
            f'{self.max_num_vectors}'

        # tests and reference signals should have same number of features
        assert num_test_features == num_ref_features, \
            f'{num_test_features} != {num_ref_features}'

        # TODO: Implement other metrics or just euclidean distance?
        # calculate the euclidean distance matrix between each pair of feature
        # vectors in X and Y
        # euclidean distance = sqrt((x - y)^2) == sqrt((x^2 - 2xy + y))
        xy = np.dot(test_signal, ref_signal.T)

        x_squared = np.square(test_signal).sum(axis=1)
        y_squared = np.square(ref_signal).sum(axis=1)
        cost_matrix = (x_squared + (-2 * xy + y_squared).T).T

        if self.distance_metric != 'euclidean_squared':
            # don't sqrt if using square of euclidean distance
            cost_matrix = np.sqrt(cost_matrix)

        # normalise with the number of vector features
        cost_matrix /= num_test_features

        # cost_matrix = cost_matrix.astype(np.float64)

        return cost_matrix

    def _calculate_path_and_distance(self, cost_matrix):
        """DTW Part 2.

        Calculate optimal path and distance in the cost matrix
        Operates a back-tracking approach to find the optimal path of least
        distance
        Uses fast cython implementation
        Path is a list of (x, y) coordinate pairings for the optimal path.
        Distance is the sum of the cumulative cost values in this path

        Args:
            cost_matrix (numpy array): num_x_vectors * num_y_vectors matrix

        Returns:
            tuple: containing the path and distance
        """
        if self.version == 'cython':
            # calculate cumulative cost matrix and back-track
            path, distance, cost_matrix, path_length = compute_dtw_distance(
                np.empty((self.max_num_vectors, self.max_num_vectors),
                         dtype=np.float32),
                cost_matrix,
                tc=self.transition_cost,
                bwd=self.beam_width, tnt=self.top_n_tailing,
                find_path=self.find_path
            )
        else:
            cumulative_cost_matrix = \
                self._calculate_cumulative_cost_matrix(cost_matrix=cost_matrix)
            path, distance = \
                self._back_track(cumulative_cost_matrix=cumulative_cost_matrix)

        return path, distance, cost_matrix, path_length

    def _calculate_cumulative_cost_matrix(self, cost_matrix):
        """Non cython based implementation.

        Calculate the minimum cumulative cost matrix

        Args:
            cost_matrix (numpy array): num_x_vectors * num_y_vectors matrix

        Returns:
            numpy array: minimum cumulative sum cost matrix
        """
        return None

    def _back_track(self, cumulative_cost_matrix):
        """Non cython based implementation.

        Args:
            cumulative_cost_matrix:

        Returns:
            tuple: containing list of coords for path and distance
        """
        return None, None


class SegmentedDTW(DTW):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate_distance(self, test_signal, ref_signal, debug=False):
        vec_dist = self._calculate_cost_matrix(test_signal, ref_signal)
        # vec_dist = np.zeros((7, 11))

        # print(test_signal.shape, ref_signal.shape)
        # print('Vec Dist Shape:', vec_dist.shape)

        m, n = vec_dist.shape
        r = 6
        max_num_warping_paths = int((n - 1) / 2)
        width = (2 * r) + 1

        import matplotlib.pyplot as plt
        from shapely.geometry import Point
        from shapely.geometry.polygon import Polygon

        distances = []
        for k in range(1, max_num_warping_paths):
            top_left = (0, (k - 1) * r)
            top_right = (0, top_left[1] + r)

            bottom_right = (m - 1, (m - 1) + (r * k))
            bottom_left = (m - 1, bottom_right[1] - (width - 1))

            top_left_down = (top_left[0] + r, (k - 1) * r)

            if bottom_right[1] >= n:
                break

            if debug:
                print('Warp Path (npy co-ords):', top_left, top_left_down, top_right, bottom_left, bottom_right)
            polygon = Polygon([top_left[::-1], top_right[::-1],
                               bottom_right[::-1], bottom_left[::-1],
                               top_left_down[::-1]])  # expects (x, y) i.e. (columns, rows)
            if debug:
                plt.plot(*polygon.exterior.xy)
                plt.gca().invert_yaxis()
                plt.show()

            new_vec_dist = np.zeros(vec_dist.shape)
            for i in range(m):
                for j in range(top_left[1], bottom_right[1] + 1):
                    point = Point(j, i)
                    if not polygon.intersects(point):
                        continue
                    new_vec_dist[i, j] = vec_dist[i, j]

            path, distance = compute_dtw_distance(
                np.empty((self.max_num_vectors, self.max_num_vectors),
                         dtype=np.float32),
                new_vec_dist.astype(np.float32),
                tc=self.transition_cost,
                bwd=self.beam_width, tnt=self.top_n_tailing,
                find_path=self.find_path
            )
            distances.append(distance)

        # print(distances)
        # print(len(distances))
        # print(distances.index(min(distances)))
        # print(min(distances))

        # import matplotlib.pyplot as plt
        # plt.plot(distances)
        # plt.show()
        #
        # return min(distances)

        return distances

    def calculate_distance_2(self, test_signal, ref_signal, debug=False):
        vec_dist = self._calculate_cost_matrix(test_signal, ref_signal)

        import matplotlib.pyplot as plt

        if debug:
            plt.imshow(vec_dist, cmap='hot', interpolation='nearest')
            plt.colorbar(plt.pcolor(vec_dist))
            plt.show()

        test_signal_length = test_signal.shape[0]
        num_similarity_cols = vec_dist.shape[1]

        distances = []
        for i in range(num_similarity_cols - test_signal_length):
            sim_sub_matrix = vec_dist[:, i:i+test_signal_length]
            assert sim_sub_matrix.shape == (test_signal_length, test_signal_length)

            path, distance = compute_dtw_distance(
                np.empty((self.max_num_vectors, self.max_num_vectors),
                         dtype=np.float32),
                sim_sub_matrix,
                tc=self.transition_cost,
                bwd=self.beam_width, tnt=self.top_n_tailing,
                find_path=self.find_path
            )
            distances.append(distance)

        return distances
