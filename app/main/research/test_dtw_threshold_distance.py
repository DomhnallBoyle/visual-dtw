"""
Initial experiment looking at whether there is some threshold that can be
found to prevent people from saying phrases that aren't in the default list
e.g. if someone says "Hello", is there a max dtw distance that when surpassed,
means the phrase isn't in the list

Assign all the default phrases to their clusters and find the centroid for each
cluster (min distance of all centroid phrases). Then check DTW distances
between centroids and new CFE templates
"""
import argparse
import glob
import os

import numpy as np
from main import configuration
from main.models import Config
from main.research.test_full_update import create_template
from main.utils.dtw import DTW
from main.utils.io import read_pickle_file

dtw_params = Config().__dict__
# dtw_params['dtw_distance_metric'] = 'euclidean'
dtw = DTW(**dtw_params)


def get_default_clusters():
    default_lst = read_pickle_file(configuration.DEFAULT_LIST_PATH)

    default_clusters = {}
    for phrase in default_lst.phrases:
        default_clusters[phrase.content] = phrase.templates
    del default_clusters['None of the above']

    return default_clusters


def get_cluster_centroids(default_clusters):
    cluster_centroids = {}
    for phrase, templates in default_clusters.items():
        average_dtw_distances = [0] * len(templates)
        for i in range(len(templates)):
            test_template = templates[i]
            ref_templates = templates[:i] + templates[i+1:]
            for ref_template in ref_templates:
                dtw_distance = dtw.calculate_distance(
                    test_template.blob.astype(np.float32),
                    ref_template.blob.astype(np.float32)
                )
                average_dtw_distances[i] += dtw_distance
            average_dtw_distances[i] /= len(ref_templates)
        min_dtw_distance_index = \
            average_dtw_distances.index(min(average_dtw_distances))
        centroid_template = templates[min_dtw_distance_index]
        cluster_centroids[phrase] = centroid_template

    return cluster_centroids


def get_test_templates(videos_directory):
    video_paths = glob.glob(os.path.join(videos_directory, '*.mp4'))

    for video_path in video_paths:
        template = create_template(video_path)
        if not template:
            continue

        yield video_path, template


def main(args):
    default_clusters = get_default_clusters()
    cluster_centroids = get_cluster_centroids(default_clusters)

    from fastdtw import fastdtw
    from pydtw import dtw2d
    from scipy.spatial.distance import euclidean

    for video_path, test_template in get_test_templates(args.videos_directory):
        # dtw_distances = {
        #     phrase: dtw.calculate_distance(
        #         test_template.blob.astype(np.float32),
        #         centroid_template.blob.astype(np.float32)
        #     )
        #     for phrase, centroid_template in cluster_centroids.items()
        # }

        # dtw_distances = {
        #     phrase: dtw2d(
        #         test_template.blob,
        #         centroid_template.blob
        #     )[1]
        #     for phrase, centroid_template in cluster_centroids.items()
        # }

        # dtw_distances = {
        #     phrase: fastdtw(
        #         test_template.blob,
        #         centroid_template.blob,
        #         dist=euclidean
        #     )[0]
        #     for phrase, centroid_template in cluster_centroids.items()
        # }
        #
        # print(video_path, dtw_distances)

        # # get average distances
        # dtw_distances = list(dtw_distances.values())
        # av_dtw_distance = sum(dtw_distances) / len(dtw_distances)
        # print(av_dtw_distance)

        dtw_cluster_averages = {}
        for phrase, cluster_templates in default_clusters.items():
            average = 0
            for cluster_template in cluster_templates:
                # average += fastdtw(
                #     test_template.blob,
                #     cluster_template.blob,
                #     dist=euclidean
                # )[0]

                average += dtw.calculate_distance(
                    test_template.blob.astype(np.float32),
                    cluster_template.blob.astype(np.float32)
                )

            average /= len(cluster_templates)
            dtw_cluster_averages[phrase] = average

        min_average_phrase = min(dtw_cluster_averages,
                                 key=dtw_cluster_averages.get)
        min_average_distance = dtw_cluster_averages[min_average_phrase]
        is_in_list = min_average_distance < 4500

        print(video_path, dtw_cluster_averages,
              min_average_phrase, min_average_distance, '\n')

        # print(video_path, is_in_list, min_average_phrase, min_average_distance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('videos_directory')

    main(parser.parse_args())
