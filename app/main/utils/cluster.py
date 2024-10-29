import threading

import numpy as np
from main.utils.dtw import DTW
from main.utils.knn import KNN


class Clustering:

    def __init__(self, clusters, **dtw_params):
        self.dtw = DTW(**dtw_params)
        self.clusters = clusters
        self.centroids = None

    def average_distances_per_phrase(self, clusters):
        """
        Returns {
            'phrase1': [avg distance for every template at phrase]
        }
        """
        average_distances_per_phrase = {}

        def internal(_clusters):
            for phrase, templates in _clusters.items():
                calculated_lookup = np.zeros((len(templates), len(templates)))
                average_distances = []
                for i in range(0, len(templates)):
                    average_distance = 0
                    for j in range(0, len(templates)):
                        if i == j:
                            continue

                        if calculated_lookup[i][j] != 0:
                            average_distance += calculated_lookup[i][j]
                            continue

                        template1 = templates[i].blob
                        template2 = templates[j].blob

                        distance = self.dtw.calculate_distance(
                            test_signal=template1,
                            ref_signal=template2)
                        average_distance += distance
                        calculated_lookup[i][j] = calculated_lookup[j][i] \
                            = distance

                    average_distance /= (len(templates) - 1)
                    average_distances.append(average_distance)
                average_distances_per_phrase[phrase] = \
                    np.asarray(average_distances)

        # use threads to speed up calculation
        num_worker_threads = 2
        num_threads = num_worker_threads + 1  # to include main thread

        # divide clusters between threads
        count_ar = np.linspace(0, len(clusters), num_threads + 1, dtype=int)
        cluster_list = []
        temp_dict = {}
        i = 1
        for key, value in clusters.items():
            temp_dict[key] = value
            if i in count_ar:
                cluster_list.append(temp_dict)
                temp_dict = {}
            i += 1

        threads = []
        for i in range(num_worker_threads):
            thread = threading.Thread(
                target=internal,
                args=(cluster_list[i],))
            thread.start()
            threads.append(thread)

        internal(cluster_list[-1])

        for i in range(num_worker_threads):
            threads[i].join()  # main thread wait until worker finished

        return average_distances_per_phrase

    def find_centroids(self):
        """Find initial centroids per phrase cluster

        Get the DTW distance between all templates for the same phrase,
        rank the templates according to their average distance from the others
        of the same phrase, (maybe then remove the X% of templates with the
        biggest average distance (X=10?) - to remove dodgy outliers templates),
        then recalculate the intra-class average distances again and select
        the template with the min average as the centroid

        Returns:

        """
        average_distances_per_phrase = \
            self.average_distances_per_phrase(self.clusters)
        clusters, centroids = {}, {}

        # drop 10% of templates with higher average distance
        for phrase, average_distances in average_distances_per_phrase.items():
            amount_to_drop = int(len(average_distances) * 0.1)
            sorted_indexes = np.argsort(average_distances)
            if amount_to_drop != 0:
                sorted_indexes = sorted_indexes[:-amount_to_drop]
            templates = self.clusters[phrase][sorted_indexes]

            clusters[phrase] = templates

        average_distances_per_phrase = \
            self.average_distances_per_phrase(clusters)

        # now select the min average distance as the phrase centroid
        for phrase, average_distances in average_distances_per_phrase.items():
            sorted_indexes = np.argsort(average_distances)
            centroids[phrase] = self.clusters[phrase][sorted_indexes[0]]

        return centroids

    def is_phrase_correct(self, uttered_phrase, template):
        """
        Vote on the closest phrase - more often than not will return the same
        phrase as the top option in the predictions (same algorithm)

        We don't want the user selecting a different option than the one
        given as the top phrase in the predictions when the top phrase is
        correct
        """
        labels, distances = [], []
        for phrase, templates in self.clusters.items():
            for cluster_template in templates:
                distance = self.dtw.calculate_distance(
                    test_signal=template.blob,
                    ref_signal=cluster_template.blob
                )
                distances.append(distance)
                labels.append(phrase)

        knn = KNN(knn_type=1, knn_k=50)
        knn.fit(distances=np.asarray(distances), labels=np.asarray(labels))
        prediction = knn.predict()

        print(f'Prediction: {prediction}, Uttered Phrase: {uttered_phrase}',
              flush=True)
        if prediction == uttered_phrase:
            return True

        return False

    def average_intra_class_distance(self, clusters, phrase):
        cluster_templates = clusters[phrase]
        centroid_template = self.centroids[phrase]
        average_distance = 0
        for cluster_template in cluster_templates:
            distance = self.dtw.calculate_distance(
                test_signal=cluster_template.blob,
                ref_signal=centroid_template.blob
            )
            average_distance += distance
        average_distance /= len(cluster_templates)

        return average_distance

    def is_suitable(self, closest_phrase, template):
        """
        First find the centroids per cluster (phrase) by
        calculating smallest av. distance between cluster (phrase) templates

        Then do before and after intra class distance with template added
        to the correct phrase.

        Args:
            closest_phrase:
            template:

        Returns:

        """
        self.centroids = self.find_centroids()

        # check before and after intra class distance
        before = self.average_intra_class_distance(self.clusters,
                                                   closest_phrase)
        clusters_copy = self.clusters
        clusters_copy[closest_phrase] = \
            np.hstack((clusters_copy[closest_phrase], template))
        after = self.average_intra_class_distance(clusters_copy,
                                                  closest_phrase)

        print(f'After: {after}, Before: {before}', flush=True)
        if after <= before:
            return True

        return False
