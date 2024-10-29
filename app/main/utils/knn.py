"""K Nearest-Neighbours Utility.

Contains logic for running KNN for classification
"""
from collections import OrderedDict

import numpy as np
from main.utils.exceptions import InaccuratePredictionException


class KNN:
    """KNN class.

    Attributes:
        num_neighbours (int): K neighbours to include in the classification
        knn_type (int): type of KNN to perform
        top_n (int): number of predictions to return
        distances (list): training data
        labels (list): corresponding ground-truth labels for the training data
    """

    def __init__(self, **kwargs):
        """Constructor.

        Args:
            **kwargs (dict): contains knn parameters
        """
        self.num_neighbours = kwargs.get('knn_k', 1)
        self.knn_type = kwargs.get('knn_type', 2)
        self.top_n = kwargs.get('top_n', 3)
        self.distances, self.labels = None, None

    def fit(self, distances, labels):
        """KNN does not have a 'training' step.

        Just set the training data and labels

        Args:
            distances (list): training data
            labels (list): corresponding ground-truth labels

        Returns:
            None
        """
        self.distances = distances
        self.labels = labels

    def predict(self, classes=None, threshold=None, **kwargs):
        """Predict the class of the test signal.

        Args:
            classes (list): all ground-truth class labels
            threshold (float): used to determine if the phrase can be
            accurately predicted or not

        Returns:
            list: containing top_n dictionaries with label-probability pairings
        """
        if not classes:
            classes = np.unique(self.labels)

        nearest_distance_indexes = \
            np.argsort(self.distances)[:self.num_neighbours]
        nearest_distances = self.distances[nearest_distance_indexes]
        nearest_labels = self.labels[nearest_distance_indexes]

        # TODO: Put into different functions
        if self.knn_type == 1:
            votes = {}
            for label in nearest_labels:
                votes[label] = votes.get(label, 0) + 1

            print(votes, flush=True)
            prediction = max(votes, key=votes.get)

            return prediction
        else:
            mean, standard_deviation = \
                nearest_distances.mean(), nearest_distances.std()
            nearest_distances -= mean
            if standard_deviation > 0:
                nearest_distances /= standard_deviation

            # apply the (1 - sigmoid function) to map values between 0 - 1
            # sigmoid function = 1 / (1 + e^-x)
            nearest_probability_distances = \
                1 - (1 / (1 + np.exp(-nearest_distances)))

            probability_sum, vote_sum = OrderedDict(), OrderedDict()
            for label in classes:
                probability_sum[label] = 0
                vote_sum[label] = 0

            for label, probability_distance in \
                    zip(nearest_labels, nearest_probability_distances):
                probability_sum[label] += probability_distance
                vote_sum[label] += 1

            # boost values to give higher weights to labels deemed correct
            for label in classes:
                probability_sum[label] = np.power(probability_sum[label], 3)
                vote_sum[label] = -np.power(vote_sum[label], 3)

            probability_sum = normalise(_dict=probability_sum)
            vote_sum = normalise(_dict=vote_sum)

            # sort by probability first, then vote
            overall_sum = [[label, probability_sum[label], vote_sum[label]]
                           for label in classes]
            sorted_top_classes = sorted(overall_sum, reverse=True,
                                        key=lambda triple: (triple[1],
                                                            triple[2]))

            if threshold and len(sorted_top_classes) >= 2:
                # calculate percentage decrease between 1st & 2nd prediction
                # and use it as a threshold to determine if the phrase can
                # be accurately predicted or not
                diff = sorted_top_classes[0][1] - sorted_top_classes[1][1]
                percent_decrease = (diff * 100) / sorted_top_classes[0][1]
                if percent_decrease < threshold:
                    raise InaccuratePredictionException

            predictions = [
                {
                    'label': triple[0],
                    'accuracy': round(triple[1], 2)
                } for triple in sorted_top_classes[:self.top_n]
            ]

            return predictions


def normalise(_dict):
    """Normalise the dictionary values.

    Normalise the values in a dictionary to their sum

    Args:
        _dict (dict): dictionary of key-value pairings

    Returns:
        dict: normalised valued dictionary
    """
    _sum = sum(_dict.values())
    if _sum > 0:
        for label in _dict.keys():
            _dict[label] /= _sum

    return _dict
