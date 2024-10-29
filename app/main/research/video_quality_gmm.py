import argparse
import ast
import copy
import glob
import os
import random
import re
import sys
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main.research.egan_dataset import get_percentage_brightness_bins_fast, \
    binary_threshold_and_extract_counts, get_most_dominant_intensities, \
    extract_haralick_features
from main.research.process_videos import get_video_rotation, fix_frame_rotation
from main.research.videos_vs_videos import _create_template, get_accuracy, \
    get_templates
from main.utils.io import read_pickle_file, write_pickle_file
from sklearn.mixture import GaussianMixture

sys.path.append('/home/domhnall/Repos/EnlightenGAN')
from extract_mouth_region import get_jaw_roi_dnn, \
    get_dnn_face_detector_and_facial_predictor

_detector, _predictor = get_dnn_face_detector_and_facial_predictor()

EGAN_SAVE_PATH = '/home/domhnall/egan_video.mp4'
DEFAULT_REF_TEMPLATES = 'ref_templates.pkl'
DEFAULT_FEATURES_PATH = 'default_features.pkl'
NUM_CENTROIDS = 2
RANDOM_STATE = 2020
DEFAULT_VIDEOS_PER_CENTROID_PKL = 'default_videos_per_centroid.pkl'


def to_feature_vector(video_path, cluster_part='mouth', debug=False):
    from extract_mouth_region import get_jaw_roi_dnn

    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    rotation = get_video_rotation(video_path)

    feature_vector = []
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame = fix_frame_rotation(frame, rotation)

        # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # v_channel = frame_hsv[:, :, 2]

        # extract mouth roi
        roi = get_jaw_roi_dnn(frame, cluster_part, _detector, _predictor)
        if not roi:
            continue
        roi, roi_x1, roi_y1, roi_x2, roi_y2 = roi
        if roi.size == 0:
            continue

        if debug:
            cv2.imshow('Original', frame)
            cv2.imshow('Cluster ROI', roi)
            cv2.waitKey(fps)

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = roi_hsv[:, :, 2]

        # feature_vector.append([
        #     *get_percentage_brightness_bins_fast(v_channel.copy()),
        #     *binary_threshold_and_extract_counts(v_channel.copy()),
        #     *get_most_dominant_intensities(v_channel.copy()),
        #     *extract_haralick_features(v_channel.copy())
        # ])

        feature_vector.append([
            *get_percentage_brightness_bins_fast(v_channel.copy()),
            *binary_threshold_and_extract_counts(v_channel.copy()),
            *get_most_dominant_intensities(v_channel.copy())
        ])

    # average across features
    feature_vector = np.asarray(feature_vector)
    feature_vector = feature_vector.mean(axis=0)

    return feature_vector


def enlighten_rois(video_path, part, debug=False, enlighten_method='egan'):
    from predict_2 import process_image as egan
    from main.research.iagcwd import process_frame as gamma

    if enlighten_method == 'egan':
        enlighten_method = egan
    else:
        enlighten_method = gamma

    video_reader = cv2.VideoCapture(video_path)
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    rotation = get_video_rotation(video_path)

    save_path = f'/home/domhnall/{uuid.uuid4()}.mp4'

    video_writer = cv2.VideoWriter(save_path,
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, (height, width))

    while True:
        success, frame = video_reader.read()
        if not success:
            break

        frame = fix_frame_rotation(frame, rotation)

        # extract jaw roi
        roi = get_jaw_roi_dnn(frame, part, _detector, _predictor)
        if not roi:
            continue
        roi, roi_x1, roi_y1, roi_x2, roi_y2 = roi
        if roi.size == 0:
            continue

        # enlighten roi
        roi = enlighten_method(roi)

        frame_copy = frame.copy()
        frame_copy[roi_y1:roi_y2, roi_x1:roi_x2] = roi

        if debug:
            cv2.imshow('Original', frame)
            cv2.imshow('Enlightened', frame_copy)
            cv2.waitKey(fps)

        video_writer.write(frame_copy)

    video_reader.release()
    video_writer.release()

    return save_path


def initial_experiment(args):
    default_video_paths = glob.glob(
        '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/default_videos/*.mp4')

    # get default video paths
    if not os.path.exists(DEFAULT_FEATURES_PATH):
        # get default video features
        video_features = [to_feature_vector(video_path,
                                            cluster_part=args.cluster_part)
                          for video_path in default_video_paths]
        video_features = np.asarray(video_features)

        write_pickle_file(video_features, DEFAULT_FEATURES_PATH)
    else:
        video_features = read_pickle_file(DEFAULT_FEATURES_PATH)

    gmm = GaussianMixture(n_components=args.num_centroids,
                          random_state=RANDOM_STATE)
    gmm.fit(video_features)

    # # generate random samples from each centroid
    # xs, ys = gmm.sample(n_samples=3)
    # print(xs[0], ys[0])
    # print(xs[1], ys[1])

    # show what videos are in what centroid
    videos_per_centroid = [[] for i in range(args.num_centroids)]
    for default_video, feature_vector in zip(default_video_paths,
                                             video_features):
        probs = gmm.predict_proba(feature_vector.reshape(1, -1))[0]
        max_prob_index = probs.argmax()
        videos_per_centroid[max_prob_index].append(default_video)
    write_pickle_file(videos_per_centroid, DEFAULT_VIDEOS_PER_CENTROID_PKL)
    show_gmm_centroids(videos_per_centroid, sample=args.sample_size)

    user_paths = [
        '/media/alex/Storage/Domhnall/datasets/sravi_dataset/pava_users/shon/*.mp4',
        '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/9/*.mp4',
        '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/11/*.mp4',
        '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/12/*.mp4'
    ]
    # user_paths = [
    #     '/media/alex/Storage/Domhnall/datasets/sravi_dataset/pava_users/shon/enlighten_gan/*.mp4',
    #     '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/9/enlighten_gan/*.mp4',
    #     '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/11/enlighten_gan/*.mp4',
    #     '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa/pava/12/enlighten_gan/*.mp4'
    # ]

    # hypothesis: [c1, c2], c2 seems to be darker conditions centroid

    for user_path in user_paths:
        user_video_paths = glob.glob(user_path)
        centroid_associations = [0] * gmm.n_components

        for video_path in user_video_paths:
            # print(video_path)
            feature_vector = to_feature_vector(video_path,
                                               cluster_part=args.cluster_part)
            probs = gmm.predict_proba(feature_vector.reshape(1, -1))[0]

            max_prob_index = probs.argmax()
            centroid_associations[max_prob_index] += 1

        print(user_path, centroid_associations)


def selective_egan(args):
    from predict_2 import process as enlighten_video

    default_features = read_pickle_file(DEFAULT_FEATURES_PATH)

    gmm = GaussianMixture(n_components=NUM_CENTROIDS,
                          random_state=RANDOM_STATE)
    gmm.fit(default_features)

    ref_video_paths = \
        glob.glob(os.path.join(args.ref_videos_directory, '*.mp4'))
    test_video_paths = \
        glob.glob(os.path.join(args.test_videos_directory, '*.mp4'))

    if not os.path.exists(DEFAULT_REF_TEMPLATES):
        ref_templates = get_templates(ref_video_paths, None)
        ref_signals = [(label, template.blob)
                       for label, template, video_path in ref_templates]
        write_pickle_file(ref_signals, DEFAULT_REF_TEMPLATES)
    else:
        ref_signals = read_pickle_file(DEFAULT_REF_TEMPLATES)

    # first get original accuracy
    test_templates = get_templates(test_video_paths, args.groundtruth_file)
    original_accuracy, original_num_tests = \
        get_accuracy(ref_signals, test_templates)

    # apply egan to every video and get accuracy
    new_test_templates = []
    for label, old_template, test_video_path in test_templates:
        enlighten_video('video', test_video_path, debug=args.debug,
                        save_path=EGAN_SAVE_PATH)
        template = _create_template(EGAN_SAVE_PATH)
        new_test_templates.append((label, template, test_video_path))
    all_accuracy, all_num_tests = get_accuracy(ref_signals, new_test_templates)

    # now apply selective gmm egan processing and get accuracy
    new_test_templates_1, new_test_templates_2 = [], []
    num_improved_videos = 0
    for label, old_template, test_video_path in test_templates:
        feature_vector = to_feature_vector(test_video_path, args.debug)

        probs = gmm.predict_proba(feature_vector.reshape(1, -1))[0]
        max_prob_index = probs.argmax()
        if max_prob_index == 1:
            # run egan on entire video
            enlighten_video('video', test_video_path, debug=args.debug,
                            save_path=EGAN_SAVE_PATH)
            template_1 = _create_template(EGAN_SAVE_PATH)

            # run egan on specific part of video
            video_save_path = enlighten_rois(test_video_path, args.part,
                                             args.debug)
            template_2 = _create_template(video_save_path)
            os.remove(video_save_path)

            num_improved_videos += 1
        else:
            template_1 = old_template
            template_2 = old_template

        new_test_templates_1.append((label, template_1, test_video_path))
        new_test_templates_2.append((label, template_2, test_video_path))

    selective_accuracy, selective_num_tests = \
        get_accuracy(ref_signals, new_test_templates_1)

    selective_part_accuracy, selective_part_num_tests = \
        get_accuracy(ref_signals, new_test_templates_2)

    print('Num improved videos: ', num_improved_videos)

    with open('selective_egan_results.csv', 'a') as f:
        f.write(f'{args.test_videos_directory},'
                f'{original_accuracy},{original_num_tests},'
                f'{all_accuracy},{all_num_tests},'
                f'{selective_accuracy},{selective_num_tests},'
                f'{selective_part_accuracy},{selective_part_num_tests},'
                f'{num_improved_videos}\n')


def show_gmm_centroids(videos_per_centroid, sample=None):
    for i, centroid_videos in enumerate(videos_per_centroid):
        print(f'Centroid {i}: {len(centroid_videos)}')

    if sample:
        # randomly sample per centroid
        videos_per_centroid = [
            random.sample(centroid_videos, sample)
            for centroid_videos in videos_per_centroid
        ]

    for i, centroid_videos in enumerate(videos_per_centroid):
        for video in centroid_videos:
            video_capture = cv2.VideoCapture(video)
            fps = int(video_capture.get(cv2.CAP_PROP_FPS))
            video_rotation = get_video_rotation(video)

            while True:
                success, frame = video_capture.read()
                if not success:
                    break

                frame = fix_frame_rotation(frame, video_rotation)

                cv2.imshow(f'Centroid {i}', frame)
                if cv2.waitKey(fps) & 0xFF == ord('q'):
                    break

            video_capture.release()


def analysis(args):
    """
    Analysis:
    Original accuracies
    EGAN all accuracies on entire video
    EGAN selective accuracies on entire video
    EGAN selective accuracies on on specific part (mouth/jaw) region
    """
    columns = ['Videos Directory',
               'Orig Accuracies', 'Orig Num Tests',
               'All Accuracies', 'All Num Tests',
               'Selective Accuracies', 'Selective Num Tests',
               'Selective Part Accuracies', 'Selective Part Num Tests',
               'Num Improved Videos']
    regex = r'(.+),(\[.+\]),(.+),(\[.+\]),(.+),(\[.+\]),(.+),(\[.+\]),(.+),(\d+)'

    data = []
    with open(args.results_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line.startswith('#'):
                videos_directory, \
                    orig_accuracies, orig_num_tests, \
                    all_accuracies, all_num_tests, \
                    selective_accuracies, selective_num_tests, \
                    selective_jaw_accuracies, selective_jaw_num_tests, \
                    num_improved_videos = \
                    re.match(regex, line).groups()
                orig_accuracies = ast.literal_eval(orig_accuracies)
                all_accuracies = ast.literal_eval(all_accuracies)
                selective_accuracies = ast.literal_eval(selective_accuracies)
                selective_jaw_accuracies = \
                    ast.literal_eval(selective_jaw_accuracies)
                data.append([videos_directory,
                             orig_accuracies, orig_num_tests,
                             all_accuracies, all_num_tests,
                             selective_accuracies, selective_num_tests,
                             selective_jaw_accuracies, selective_jaw_num_tests,
                             num_improved_videos])

    df = pd.DataFrame(data=data, columns=columns)

    for index, row in df.iterrows():
        x = [1, 2, 3]
        y1 = row['Orig Accuracies']
        y2 = row['All Accuracies']
        y3 = row['Selective Accuracies']
        # y2, y3 = [-1, -1, -1], [-1, -1, -1]
        y4 = row['Selective Part Accuracies']
        user_id = row['Videos Directory'].strip().split('/')[-1]

        for y, label, marker in zip([y1, y2, y3, y4],
                                    [f'Original {row["Orig Num Tests"]}',
                                     f'EGAN All {row["All Num Tests"]}',
                                     f'EGAN Selective {row["Selective Num Tests"]}',
                                     f'EGAN Selective Part {row["Selective Part Num Tests"]}'],
                                    ['o', '*', 's', 'v']
                                    ):
            plt.plot(x, y, label=label, marker=marker)
        plt.xlabel('Ranks')
        plt.ylabel('Accuracy %')
        plt.title(f"User: {user_id}\n"
                  f"Num Improved Videos: {row['Num Improved Videos']}")
        plt.legend()
        plt.ylim((0, 100))
        plt.xticks(x)
        if args.save_directory:
            plt.savefig(os.path.join(args.save_directory, f'{user_id}.png'))
        plt.show()


def selective_enhancement(args):
    print(args)

    # first generate features from default videos
    default_video_paths = glob.glob(
        '/media/alex/Storage/Domhnall/datasets/sravi_dataset/liopa'
        '/default_videos/*.mp4'
    )

    print('Getting default features...')
    if args.refit:
        default_video_features = np.asarray(
            [to_feature_vector(video_path, cluster_part=args.cluster_part,
                               debug=args.debug)
             for video_path in default_video_paths]
        )
        write_pickle_file(default_video_features, DEFAULT_FEATURES_PATH)
    else:
        default_video_features = read_pickle_file(DEFAULT_FEATURES_PATH)

    # fit GMM
    print('Fitting GMM...')
    gmm = GaussianMixture(n_components=args.num_centroids,
                          random_state=RANDOM_STATE)
    gmm.fit(default_video_features)

    ref_video_paths = \
        glob.glob(os.path.join(args.ref_videos_directory, '*.mp4'))
    test_video_paths = \
        glob.glob(os.path.join(args.test_videos_directory, '*.mp4'))

    # grab reference signals
    print('Getting ref signals...')
    if args.enlighten_refs:
        enlightened_ref_templates_path = 'enlightened_ref_templates.pkl'
        if not os.path.exists(enlightened_ref_templates_path):
            ref_signals = []
            ref_templates = get_templates(ref_video_paths, None)
            for label, template, ref_video_path in ref_templates:
                feature_vector = \
                    to_feature_vector(ref_video_path,
                                      cluster_part=args.cluster_part,
                                      debug=args.debug)

                probs = gmm.predict_proba(feature_vector.reshape(1, -1))[0]
                max_prob_index = probs.argmax()
                if max_prob_index == args.enlighten_cluster_index:
                    video_save_path = enlighten_rois(ref_video_path,
                                                     args.enlighten_part,
                                                     args.debug,
                                                     args.enlighten_method)
                    new_template = _create_template(video_save_path)
                    os.remove(video_save_path)
                else:
                    new_template = copy.deepcopy(template)

                ref_signals.append((label, new_template.blob))
            write_pickle_file(ref_signals, enlightened_ref_templates_path)
        else:
            ref_signals = read_pickle_file(enlightened_ref_templates_path)
    else:
        if not os.path.exists(DEFAULT_REF_TEMPLATES):
            ref_templates = get_templates(ref_video_paths, None)
            ref_signals = [(label, template.blob)
                           for label, template, video_path in ref_templates]
            write_pickle_file(ref_signals, DEFAULT_REF_TEMPLATES)
        else:
            ref_signals = read_pickle_file(DEFAULT_REF_TEMPLATES)

    # first get original accuracy
    print('Getting original accuracies...')
    test_templates = get_templates(test_video_paths, args.groundtruth_file)
    original_accuracy, original_num_tests = \
        get_accuracy(ref_signals, test_templates)

    # apply selective egan on the jaw area
    print('Extracting test features...')
    new_test_templates = []
    num_improved_videos = 0
    for label, old_template, test_video_path in test_templates:
        feature_vector = to_feature_vector(test_video_path,
                                           cluster_part=args.cluster_part,
                                           debug=args.debug)

        probs = gmm.predict_proba(feature_vector.reshape(1, -1))[0]
        max_prob_index = probs.argmax()
        if max_prob_index == args.enlighten_cluster_index:
            # run egan on specific part of video
            video_save_path = enlighten_rois(test_video_path,
                                             args.enlighten_part,
                                             args.debug, args.enlighten_method)
            new_template = _create_template(video_save_path)
            os.remove(video_save_path)
            num_improved_videos += 1
        else:
            new_template = copy.deepcopy(old_template)

        new_test_templates.append((label, new_template, test_video_path))

    print('Getting selective accuracies...')
    selective_accuracy, selective_num_tests = \
        get_accuracy(ref_signals, new_test_templates)

    print('Num improved videos: ', num_improved_videos)

    with open(f'selective_enhancement_'
              f'{args.cluster_part}_'
              f'{args.enlighten_part}_'
              f'{args.num_centroids}_'
              f'{args.enlighten_refs}_'
              f'{args.enlighten_method}.csv', 'a') as f:
        f.write(f'{args.test_videos_directory},'
                f'{original_accuracy},{original_num_tests},'
                f'{selective_accuracy},{selective_num_tests},'
                f'{num_improved_videos}\n')


def selective_analysis(args):
    args.show_graphs = args.show_graphs == 'True'

    columns, data = ['Test Directory', 'Original Accuracies',
                     'Original Num Tests', 'Enlightened Accuracies',
                     'Enlightened Num Tests', 'Num Improved'], []

    regex = r'(.+),(\[.+\]),(.+),(\[.+\]),(.+),(\d+)'

    with open(args.file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            test_directory, original_accuracies, original_num_tests,\
                enlightened_accuracies, enlightened_num_tests, num_improved = \
                re.match(regex, line).groups()
            original_accuracies = ast.literal_eval(original_accuracies)
            enlightened_accuracies = ast.literal_eval(enlightened_accuracies)
            data.append([
                test_directory, original_accuracies, original_num_tests,
                enlightened_accuracies, enlightened_num_tests, num_improved
            ])

    df = pd.DataFrame(data=data, columns=columns)
    rows = [row for index, row in df.iterrows()]

    experiment_args = \
        args.file_path.split('/')[-1].replace('.csv', '').split('_')
    figure_title = f'Cluster Part: {experiment_args[2]}, ' \
                   f'Enlighten Part: {experiment_args[3]}, ' \
                   f'Num Centroids: {experiment_args[4]}, ' \
                   f'Enlighten Refs: {experiment_args[5]}, ' \
                   f'Enlighten Method: {experiment_args[6]}'

    def plot_rows(sub_rows):
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
        fig.suptitle(figure_title)
        x = [1, 2, 3]
        for i, row in enumerate(sub_rows):
            test_directory = row['Test Directory']
            original_accuracies = row['Original Accuracies']
            enlightened_accuracies = row['Enlightened Accuracies']
            axs[i].plot(x, original_accuracies,
                        label=f'Original {row["Original Num Tests"]}')
            axs[i].plot(x, enlightened_accuracies,
                        label=f'Enlightened {row["Enlightened Num Tests"]}')
            axs[i].set_title(f'User: {test_directory.split("/")[-1]}\n'
                             f'Num Improved: {row["Num Improved"]}')
            axs[i].set_xlabel('Ranks')
            axs[i].set_ylabel('Accuracy %')
            axs[i].set_ylim([0, 100])
            axs[i].set_xticks(x)
            axs[i].legend()

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # plot graphs
    if args.show_graphs:
        while True:
            sub_rows = []
            try:
                sub_rows.append(rows.pop())
                sub_rows.append(rows.pop())
                sub_rows.append(rows.pop())
                plot_rows(sub_rows)
            except IndexError:
                plot_rows(sub_rows)
                break

    # get overall change in accuracy
    for index, row in df.iterrows():
        test_directory = row['Test Directory']
        o_as = row['Original Accuracies']
        e_as = row['Enlightened Accuracies']

        overall_accuracy_diff = 0
        for i in range(3):
            overall_accuracy_diff += (e_as[i] - o_as[i])

        print(f'{test_directory.split("/")[-1]}: ', overall_accuracy_diff)


def main(args):
    run_type = args.run_type

    if run_type == 'initial_experiment':
        initial_experiment(args)
    elif run_type == 'selective_egan':
        selective_egan(args)
    elif run_type == 'show_default_centroids':
        show_gmm_centroids(read_pickle_file(DEFAULT_VIDEOS_PER_CENTROID_PKL))
    elif run_type == 'analysis':
        analysis(args)
    elif run_type == 'selective_enhancement':
        selective_enhancement(args)
    elif run_type == 'selective_analysis':
        selective_analysis(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('initial_experiment')
    parser_1.add_argument('--cluster_part', default='mouth')
    parser_1.add_argument('--num_centroids', default=NUM_CENTROIDS, type=int)
    parser_1.add_argument('--sample_size', default=10, type=int)

    parser_2 = sub_parsers.add_parser('selective_egan')
    parser_2.add_argument('ref_videos_directory')
    parser_2.add_argument('test_videos_directory')
    parser_2.add_argument('--groundtruth_file')
    parser_2.add_argument('--part', default='jaw')
    parser_2.add_argument('--debug', default=False)

    parser_3 = sub_parsers.add_parser('show_default_centroids')

    parser_4 = sub_parsers.add_parser('analysis')
    parser_4.add_argument('--save_directory')
    parser_4.add_argument('--results_path',
                          default='selective_egan_results.csv')

    def _bool(s):
        return s == 'True'

    parser_5 = sub_parsers.add_parser('selective_enhancement')
    parser_5.add_argument('ref_videos_directory')
    parser_5.add_argument('test_videos_directory')
    parser_5.add_argument('enlighten_cluster_index', type=int)
    parser_5.add_argument('--num_centroids', default=NUM_CENTROIDS, type=int)
    parser_5.add_argument('--cluster_part', default='mouth')
    parser_5.add_argument('--enlighten_part', default='jaw')
    parser_5.add_argument('--refit', type=_bool, default='False')
    parser_5.add_argument('--groundtruth_file')
    parser_5.add_argument('--debug', type=_bool, default='False')
    parser_5.add_argument('--enlighten_refs', type=_bool, default='False')
    parser_5.add_argument('--enlighten_method', choices=['egan', 'gamma'])

    parser_6 = sub_parsers.add_parser('selective_analysis')
    parser_6.add_argument('file_path')
    parser_6.add_argument('--show_graphs', default='True')

    main(parser.parse_args())
