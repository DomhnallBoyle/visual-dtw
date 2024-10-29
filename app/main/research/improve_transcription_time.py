"""Taking the default sessions e.g.

If we cluster these on phrases and found the centroid for each cluster
If we then used the centroids only for transcriptions rather than every
single template, this would obviously improve the transcription time but
how is the accuracy affected?

Results (num_per_centroid=1):
This doesn't use the threshold either
9 [67.67676767676768, 86.86868686868686, 87.87878787878788]
11 [72.5, 86.66666666666667, 90.83333333333333]
12 [85.55555555555556, 93.33333333333333, 97.77777777777777]

Results (averaging distances per centroid (num_per_centroid=5))
This doesn't include the threshold
9 [74.74747474747475, 85.85858585858585, 88.88888888888889]
11 [77.5, 90.0, 95.0]
12 [87.22222222222223, 95.55555555555556, 97.77777777777777]

Results (running DTW/KNN on the centroids per cluster (num_per_centroid=5))
This does include the threshold
9 [65.65656565656566, 71.71717171717172, 71.71717171717172]
11 [66.66666666666667, 70.83333333333333, 72.5]
12 [77.22222222222223, 78.88888888888889, 81.11111111111111]

Not as good as just using the default sessions
Could be because of selecting the centroids - just picking out 1 template
doesn't work very well for testing
"""
import argparse

import numpy as np
from main.models import Config
from main.research.cmc import CMC
from main.research.test_update_list_3 import get_default_sessions
from main.research.test_update_list_5 import get_test_data
from main.utils.dtw import DTW
from main.utils.knn import KNN


class Clustering:

    def __init__(self, num_centroids_per_cluster=1):
        self.num_centroids_per_cluster = num_centroids_per_cluster
        self.clusters = {}
        self.centroids = {}
        self.params = Config().__dict__
        self.dtw = DTW(**self.params)

    def fit(self, sessions):
        # separate into phrase clusters first
        for session_label, session in sessions:
            for phrase, template in session:
                cluster = self.clusters.get(phrase, [])
                cluster.append(template)
                self.clusters[phrase] = cluster

        # find centroids for every cluster
        # must be most reflective of that phrase cluster
        # e.g. minimum DTW distance
        for phrase, cluster in self.clusters.items():
            num_cluster_samples = len(cluster)
            accuracies = [[] for i in range(num_cluster_samples)]
            for i, t1 in enumerate(cluster):
                for j, t2 in enumerate(cluster):
                    if i == j:
                        continue

                    signal_1 = cluster[i].blob.astype(np.float32)
                    signal_2 = cluster[j].blob.astype(np.float32)
                    accuracies[i].append(
                        self.dtw.calculate_distance(signal_1, signal_2)
                    )

            # get average and grab minimum
            num_selected = 0
            min_accuracies = [sum(l) / len(l) for l in accuracies]
            while num_selected != self.num_centroids_per_cluster:
                min_index = min_accuracies.index(min(min_accuracies))

                centroids = self.centroids.get(phrase, [])
                centroids.append(cluster.pop(min_index))
                self.centroids[phrase] = centroids
                min_accuracies.pop(min_index)

                num_selected += 1

    def predict(self, template):
        # centroid_distances = []
        # centroid_labels = np.asarray(list(self.centroids.keys()))
        # for centroid in self.centroids.values():
        #     signal_1 = centroid.blob.astype(np.float32)
        #     signal_2 = template.blob.astype(np.float32)
        #     distance = self.dtw.calculate_distance(signal_1, signal_2)
        #     centroid_distances.append(distance)
        #
        # # sort distances in ascending order
        # centroid_distances = np.asarray(centroid_distances)
        # sorted_indices = np.argsort(centroid_distances)
        #
        # # return top 3 results
        # return centroid_labels[sorted_indices][:3]

        # # one way to do it would be to average the distances
        # centroid_labels = np.asarray(list(self.centroids.keys()))
        # test_signal = template.blob.astype(np.float32)
        #
        # average_distances = []
        # for centroids_per_cluster in self.centroids.values():
        #     average = 0
        #     for centroid in centroids_per_cluster:
        #         ref_signal = centroid.blob.astype(np.float32)
        #         distance = self.dtw.calculate_distance(test_signal, ref_signal)
        #         average += distance
        #     average /= self.num_centroids_per_cluster
        #     average_distances.append(average)
        #
        # average_distances = np.asarray(average_distances)
        # sorted_indices = np.argsort(average_distances)
        #
        # return centroid_labels[sorted_indices][:3]

        # another way would be to run KNN on all the centroids per cluster
        # together and vote
        ref_signals = []
        test_signal = template.blob
        for phrase, centroids_per_cluster in self.centroids.items():
            for centroid in centroids_per_cluster:
                ref_signal = centroid.blob
                ref_signals.append((phrase, ref_signal))

        predictions = self.dtw.classify(test_signal, ref_signals, None,
                                        **self.params)
        return [prediction['label'] for prediction in predictions]


def main(args):
    default_sessions = get_default_sessions()

    clustering = Clustering(
        num_centroids_per_cluster=args.num_centroids_per_cluster)
    clustering.fit(default_sessions)

    for user_id in ['9', '11', '12']:
        test_data = get_test_data(user_id=user_id)

        cmc = CMC(num_ranks=3)

        for actual_label, test_template in test_data:
            try:
                predicted_labels = clustering.predict(test_template)
                cmc.tally(predicted_labels, actual_label)
            except Exception as e:
                print(e)
                continue

        cmc.calculate_accuracies(num_tests=len(test_data), count_check=False)
        print(user_id, cmc.all_rank_accuracies[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_centroids_per_cluster', type=int, default=1)

    main(parser.parse_args())
