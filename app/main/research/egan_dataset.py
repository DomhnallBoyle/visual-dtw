"""
EnlightenGAN dataset creator
"""
import argparse
import ast
import cv2
import os
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from img2vec_pytorch import Img2Vec
from main import configuration
from main.models import Config
from main.research.test_update_list_5 import create_template, \
    get_default_sessions
from main.services.transcribe import transcribe_signal
from main.utils.db import find_phrase_mappings, invert_phrase_mappings
from main.utils.io import read_json_file
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# enlighten gan path
sys.path.append('/home/domhnall/Repos/EnlightenGAN')

DATA_PATH = '/media/alex/Storage/Domhnall/datasets/sravi_dataset'
LIOPA_DATA_PATH = os.path.join(DATA_PATH, 'liopa', 'pava')
PAVA_USERS_DATA_PATH = os.path.join(DATA_PATH, 'pava_users')
DOCTORS_DATA_PATH = os.path.join(DATA_PATH, 'doctors')
PATIENTS_DATA_PATH = os.path.join(DATA_PATH, 'patients')

EGAN_VIDEO_DATASET_PATH = 'egan_video_datase.csv'
EGAN_FEATURES_DATASET_PATH = 'egan_features_dataset.csv'

PAVA_VIDEO_REGEX = r'PV0*(\d+)_P10*(\d+)_S0*(\d+)'
SRAVI_VIDEO_REGEX = r'(\d+)_(S\w?[A-Z])(\d+)_S(\d+)'

ALL_PHRASES = read_json_file(configuration.PHRASES_PATH)
PAVA_PHRASES = ALL_PHRASES['PAVA-DEFAULT']

# NUM_FEATURES = 9
# NUM_FEATURES = 19
# NUM_FEATURES = 64  # more bins and 5 dominant colours
NUM_FEATURES = 66  # + otsu thresholding black and white pixel counts
# NUM_FEATURES = 512  # image embeddings
# NUM_FEATURES = 4096
# NUM_FEATURES = 49

img2vec = Img2Vec(model='alexnet')

# TODO: Add more video data from liopa users


def read_video_dataset():
    if not os.path.exists(EGAN_VIDEO_DATASET_PATH):
        return None

    data = []
    columns = ['Video Path', 'Groundtruth', 'Original Preds', 'Egan Preds',
               'Improvement']

    with open(EGAN_VIDEO_DATASET_PATH, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            video_path, groundtruth, original_preds, egan_preds, improvement \
                = re.match(r'(.+),(.+),(\[.*\]),(\[.*\]),(.+)', line).groups()
            data.append([video_path, groundtruth,
                         ast.literal_eval(original_preds),
                         ast.literal_eval(egan_preds),
                         improvement == 'True'
                         ])

    return pd.DataFrame(data=data, columns=columns)


def read_features_dataset(path=EGAN_FEATURES_DATASET_PATH,
                          num_features=NUM_FEATURES):
    if not os.path.exists(path):
        return None

    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            data_point = line.split(',')
            video_path = data_point[0]
            features = [float(f) for f in data_point[1:num_features+1]]
            improvement = data_point[num_features+1]

            data.append([
                video_path,
                *features,
                improvement == 'True'
            ])

    columns = [
        'Video Path',
        *[f'Feature {i+1}' for i in range(num_features)],
        'Improvement'
    ]

    return pd.DataFrame(data=data, columns=columns)


def data_generator():
    from video_quality import process_video as enlighten_video

    # grab all data as templates with groundtruth labels
    _get_user_paths = lambda parent, excludes: \
        [os.path.join(parent, child) for child in os.listdir(parent)
         if os.path.isdir(os.path.join(parent, child))
         and child not in excludes]

    liopa_user_paths = _get_user_paths(LIOPA_DATA_PATH, ['2', '3', '7'])
    pava_user_paths = _get_user_paths(PAVA_USERS_DATA_PATH, [])
    doctor_user_paths = _get_user_paths(DOCTORS_DATA_PATH, [])
    patient_user_paths = _get_user_paths(PATIENTS_DATA_PATH, [])

    def _get_groundtruth(d, from_pava=False, from_file=False):
        data = []
        for video in os.listdir(d):
            if not video.endswith('.mp4'):
                continue

            if from_file:
                validated_path = os.path.join(d, 'validated_groundtruth.txt')
                validated_exists = os.path.exists(validated_path)
                if validated_exists:
                    df = pd.read_csv(validated_path,
                                     names=['Video', 'Actual'])
                else:
                    df = pd.read_csv(os.path.join(d, 'groundtruth.txt'),
                                     names=['Video', 'Said', 'Actual'])

                phrase = str(df[df['Video'] == video]['Actual'].values[0])
            else:
                if from_pava:
                    phrase_id = re.match(PAVA_VIDEO_REGEX, video).groups()[1]
                    phrase = PAVA_PHRASES[phrase_id]
                else:
                    phrase_set, phrase_id = \
                        re.match(SRAVI_VIDEO_REGEX, video).groups()[1:3]
                    phrase = ALL_PHRASES[phrase_set][str(int(phrase_id))]
                    if phrase not in PAVA_PHRASES.values():
                        continue

            full_video_path = os.path.join(d, video)
            data.append((phrase, full_video_path))

        return data

    data = []
    for user_path in liopa_user_paths:
        data.extend(_get_groundtruth(user_path, from_pava=True))
    for user_path in pava_user_paths:
        data.extend(_get_groundtruth(user_path, from_file=True))
    for user_path in doctor_user_paths:
        data.extend(_get_groundtruth(user_path, from_pava=False))
    for user_path in patient_user_paths:
        data.extend(_get_groundtruth(user_path, from_pava=False))

    print('Num data: ', len(data))

    # check if any of the dataset already exists
    df = read_video_dataset()

    for phrase, video_path in data:
        if df is not None and \
                ((df['Video Path'] == video_path) & (df['Groundtruth'] == phrase)).any():
            continue

        print(video_path)

        # create template
        original_template = create_template(video_path)
        if not original_template:
            continue

        # run enlighten-gan and get template
        enlighten_video(video_path, save=True, debug=False)
        egan_template = create_template('/home/domhnall/video.mp4')
        if not egan_template:
            continue

        yield phrase, video_path, original_template, egan_template


def get_video_rotation(video_path):
    import subprocess, re
    cmd = f'ffmpeg -i {video_path}'

    p = subprocess.Popen(
        cmd.split(' '),
        stderr=subprocess.PIPE,
        close_fds=True
    )
    stdout, stderr = p.communicate()

    reo_rotation = re.compile('rotate\s+:\s(\d+)')
    match_rotation = reo_rotation.search(str(stderr))
    rotation = match_rotation.groups()[0]

    return int(rotation)


def get_percentage_brightness_bins(channel, num_bins=10):
    start = time.time()
    levels = np.linspace(0, 255, num=10)
    d_levels = {i: 0 for i in range(0, num_bins)}
    for pixel_value in channel.flatten():
        try:
            image_bright_level = np.digitize(pixel_value, levels, right=True)
            d_levels[image_bright_level] += 1
        except KeyError as e:
            print(pixel_value, image_bright_level)
            raise e

    factor = 1.0 / sum(d_levels.values())
    for level in d_levels:
        d_levels[level] *= factor

    end = time.time()
    print(end - start)

    return list(d_levels.values())


def get_percentage_brightness_bins_fast(channel, num_bins=10):
    hist = cv2.calcHist([channel], [0], None, [num_bins], (0, 256))
    hist = [b[0] for b in hist]
    factor = 1.0 / sum(hist)  # normalise so sum(values) = 1

    return [b * factor for b in hist]


def get_most_dominant_intensities(channel, num=5):
    colors, count = np.unique(channel.flatten(), return_counts=True)

    return colors[count.argsort()[::-1][:num]]


def binary_threshold_and_extract_counts(channel):
    # # adaptive thresholding
    # th2 = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)
    # th3 = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY, 11, 2)

    # otsus thresholding
    ret2, output = cv2.threshold(channel, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # cv2.imshow('Otsu', output)
    # cv2.waitKey(5)

    # count number of black and white pixels
    output = output.flatten()
    colors, counts = np.unique(output, return_counts=True)

    num_pixels = len(output)

    return [counts[0] / num_pixels, counts[1] / num_pixels]


def extract_frame_embeddings(frame):
    pil_image = Image.fromarray(frame.astype('uint8'), 'RGB')
    vec = img2vec.get_vec(pil_image)

    return vec


def extract_lbp_features(frame):
    from mahotas.features.lbp import lbp

    hist = lbp(frame, radius=3, points=8)

    # print(hist.shape)

    return list(hist)


def extract_haralick_features(frame):
    from mahotas.features.texture import haralick

    haralick_features = haralick(frame, return_mean=True)

    # print(haralick_features.shape)

    return list(haralick_features)


def to_grayscale(frame):
    # bgr frame
    b, g, r = cv2.split(frame)

    # grayscale = 1. - (0.299*r + 0.587*g + 0.114*b) / 2.
    grayscale_2 = 255. - (0.299*r + 0.587*g + 0.114*b) / 2.

    # fig = plt.figure(figsize=(8, 8))
    # columns = 2
    # rows = 1
    # for i, img in enumerate([grayscale, grayscale_2]):
    #     fig.add_subplot(rows, columns, i + 1)
    #     plt.imshow(img, cmap='gray')
    # plt.show()

    return grayscale_2.astype(np.uint8)


def extract_video_features():
    from check_brightness import get_entropy_contrast, get_rms_contrast_fast, \
        get_hsv_brightness, get_rms_contrast
    from extract_mouth_region import \
        get_dnn_face_detector_and_facial_predictor, get_jaw_roi_dnn
    from skimage.measure import shannon_entropy

    df_videos = read_video_dataset()
    if df_videos is None:
        print('No dataset available')
        return

    df_features = read_features_dataset()

    _detector, _predictor = get_dnn_face_detector_and_facial_predictor()

    for index, row in df_videos.iterrows():
        video_path = row['Video Path']
        improvement = row['Improvement']

        # # check if video already has features
        # if df_features is not None and \
        #         (df_features['Video Path'] == video_path).any():
        #     print('Exists')
        #     continue

        video_reader = cv2.VideoCapture(video_path)
        num_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        print(video_path, num_frames)

        video_rotation = get_video_rotation(video_path)

        video_features = [0] * NUM_FEATURES

        video_embeddings = []

        while True:
            success, frame = video_reader.read()
            if not success:
                break

            # fix rotation issue
            if video_rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif video_rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # roi = get_jaw_roi_dnn(frame, 'mouth', _detector, _predictor)
            roi = get_jaw_roi_dnn(frame, 'jaw', _detector, _predictor)
            if not roi:
                continue
            roi, roi_x1, roi_y1, roi_x2, roi_y2 = roi

            if roi.size == 0:
                continue

            # frame_embedding = extract_frame_embeddings(roi)
            # video_embeddings.append(frame_embedding)

            channel_roi = to_grayscale(roi)

            # roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # v_channel = roi_hsv[:, :, 2]

            # lbp_features = extract_lbp_features(v_channel.copy())
            # haralick_features = extract_haralick_features(v_channel.copy())
            #
            # frame_features = [*lbp_features, *haralick_features]

            # get features
            bin_brightness_levels = \
                get_percentage_brightness_bins_fast(channel_roi.copy(),
                                                    num_bins=50)
            dominant_intensities = get_most_dominant_intensities(
                channel_roi.copy())  # most dominant colour
            black_white_counts = \
                binary_threshold_and_extract_counts(channel_roi.copy())
            brightness_values = get_hsv_brightness(channel_roi.copy())
            hsv_entropy_contrast = shannon_entropy(channel_roi.copy())
            hsv_rms_contrast = get_rms_contrast_fast(channel_roi.copy())

            frame_features = [
                *brightness_values,
                hsv_entropy_contrast,
                hsv_rms_contrast,
                *bin_brightness_levels,
                *dominant_intensities,
                *black_white_counts
            ]

            # # get features
            # bin_brightness_levels = \
            #     get_percentage_brightness_bins_fast(v_channel.copy(),
            #                                         num_bins=50)
            #
            # # most dominant colour
            # dominant_intensities = get_most_dominant_intensities(
            #     v_channel.copy())
            #
            # black_white_counts = \
            #     binary_threshold_and_extract_counts(v_channel.copy())
            #
            # brightness_values = get_hsv_brightness(v_channel.copy())
            # hsv_entropy_contrast = shannon_entropy(v_channel.copy())
            # # hsv_entropy_contrast = get_entropy_contrast(v_channel.copy())
            # hsv_rms_contrast = get_rms_contrast_fast(v_channel.copy())
            #
            # for i, feature_value in enumerate([
            #     *brightness_values,
            #     hsv_entropy_contrast,
            #     hsv_rms_contrast,
            #     *bin_brightness_levels,
            #     *dominant_intensities,
            #     *black_white_counts
            # ]):
            #     features[i] += feature_value

            for i, feature_value in enumerate(frame_features):
                video_features[i] += feature_value

        feature_string = ','.join([str(f / num_frames)
                                   for f in video_features])
        # print(feature_string)

        # video_embeddings = np.asarray(video_embeddings)
        # video_embedding = video_embeddings.mean(axis=0)
        # feature_string = ','.join([str(v) for v in video_embedding])

        with open(EGAN_FEATURES_DATASET_PATH, 'a') as f:
            line = f'{video_path},{feature_string},{improvement}\n'
            f.write(line)


def test(test_template, ref_signals):
    try:
        predictions = transcribe_signal(ref_signals, test_template.blob, None,
                                        **Config().__dict__)
    except Exception as e:
        return []

    return predictions


def is_improvement(actual_label, original_predictions, egan_predictions):
    _sep_preds = lambda preds: \
        ([pred['label'] for pred in preds],
         [pred['accuracy'] for pred in preds])

    _get_position = lambda l, e: l.index(e) if e in l else None

    if original_predictions and not egan_predictions:
        return False
    elif not original_predictions and egan_predictions:
        return True \
            if actual_label in _sep_preds(egan_predictions)[0] else False
    elif not original_predictions and not egan_predictions:
        return False

    original_labels, original_accuracies = _sep_preds(original_predictions)
    egan_labels, egan_accuracies = _sep_preds(egan_predictions)

    # improvement in position
    original_position = _get_position(original_labels, actual_label)
    egan_position = _get_position(egan_labels, actual_label)

    if original_position is not None and egan_position is not None:

        if original_position == egan_position:
            # check for improvement in accuracy

            original_accuracy = original_accuracies[original_position]
            egan_accuracy = egan_accuracies[egan_position]

            if original_accuracy == egan_accuracy:
                return False  # 3rd option?
            elif original_accuracy < egan_accuracy:
                return True
            elif original_accuracy > egan_accuracy:
                return False

        elif original_position < egan_position:
            return False
        elif original_position > egan_position:
            return True

    elif original_position is not None and egan_position is None:
        return False
    elif original_position is None and egan_position is not None:
        return True
    elif original_position is None and egan_position is None:
        return False  # not sure about this (maybe 3rd option?)


def initialise_dataset():
    default_sessions = get_default_sessions()
    ref_signals = [(label, template.blob)
                   for session_label, ref_session in default_sessions
                   for label, template in ref_session]

    for actual_label, video_path, original_template, egan_template \
            in data_generator():

        # original dtw
        original_predictions = test(test_template=original_template,
                                    ref_signals=ref_signals)

        # egan dtw
        egan_predictions = test(test_template=egan_template,
                                ref_signals=ref_signals)

        print(original_predictions, egan_predictions)

        improvement = is_improvement(actual_label, original_predictions,
                                     egan_predictions)

        with open(EGAN_VIDEO_DATASET_PATH, 'a') as f:
            line = f'{video_path},{actual_label},{original_predictions},' \
                   f'{egan_predictions},{improvement}\n'
            f.write(line)


def train_classifier(dataset_path):
    from sklearn.model_selection import GridSearchCV, train_test_split, \
        RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
        VotingClassifier
    from sklearn.model_selection import cross_val_score, \
        RepeatedStratifiedKFold
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.mixture import GaussianMixture
    from sklearn.feature_selection import SelectKBest, SelectFromModel
    from sklearn.feature_selection import chi2

    seed = 2020

    dataset = read_features_dataset(path=dataset_path)

    # # apply under-sampling to get 50/50 class distribution
    # improved = dataset[dataset['Improvement'] == True].sample(402)
    # not_improved = dataset[dataset['Improvement'] == False].sample(402)
    # dataset = pd.concat([improved, not_improved])

    # convert True/False to 1/0
    dataset['Improvement'] = dataset['Improvement'].astype(int)

    # # show correlations of each features in dataset
    # import seaborn as sns
    # corrmat = dataset.corr()
    # top_corr_features = corrmat.index
    # plt.figure(figsize=(20, 20))
    # sns.heatmap(dataset[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    # plt.show()

    # # select specific user
    # dataset = dataset[dataset['Video Path'].str.contains('liopa/pava/11')]
    # print(len(dataset))

    x, y = dataset.filter(regex='Feature *').values, \
           dataset['Improvement'].values

    # selected_features = [59]
    # x, y = dataset[[f'Feature {i}' for i in selected_features]].values, \
    #        dataset['Improvement'].values

    # # feature selection
    # x = SelectKBest(chi2, k=10).fit_transform(x, y)

    # # l1 based feature selection
    # lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(x, y)
    # model = SelectFromModel(lsvc, prefit=True)
    # x = model.transform(x)
    # print(x.shape)

    # # tree based feature selection
    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = clf.fit(x, y)
    # model = SelectFromModel(clf, prefit=True)
    # x = model.transform(x)
    # print(x.shape)

    # standardize the dataset’s features onto unit scale
    # (mean = 0 and variance = 1)
    x = StandardScaler().fit_transform(x)

    # # run PCA
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # x = pca.fit_transform(x)
    # new_df = pd.DataFrame(data=np.hstack((x, np.array([y]).T)),
    #                       columns=['Feature 1',
    #                                'Feature 2',
    #                                'Target'])
    # ax = None
    # for colour, target in zip(['r', 'b'], np.unique(y)):
    #     sub_df = new_df[new_df['Target'] == target]
    #     ax = sub_df.plot(x='Feature 1', y='Feature 2', kind='scatter', c=colour,
    #                      label=str(target), ax=ax)
    # plt.show()

    # check class distribution of dataset
    num_improved = len(dataset[dataset['Improvement'] == True])
    num_not_improved = len(dataset) - num_improved
    print('\nNum Improved: ', num_improved, num_improved / len(dataset))
    print('Num Not Improved: ', num_not_improved,
          num_not_improved / len(dataset), '\n')

    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, train_size=0.7, shuffle=True, random_state=seed)

    print('Training: ', len(x_train), len(y_train))
    print('Testing: ', len(x_test), len(y_test))

    # dummy classifier
    classifier = DummyClassifier(strategy='uniform')
    classifier.fit(x_train, y_train)
    train_accuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('\nDummy: ', train_accuracy, test_accuracy)

    # rbf svm
    classifier = SVC(gamma=2)
    classifier.fit(x_train, y_train)
    train_accuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('SVM: ', train_accuracy, test_accuracy)

    # logistic regression
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(x_train, y_train)
    train_accuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('Logistic Regression: ', train_accuracy, test_accuracy)

    # mlp
    classifier = MLPClassifier(hidden_layer_sizes=(50, 10),
                               max_iter=1000,
                               shuffle=True)
    classifier.fit(x_train, y_train)
    train_accuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('MLP: ', train_accuracy, test_accuracy)

    # knn
    classifier = KNeighborsClassifier(10)
    classifier.fit(x_train, y_train)
    train_acccuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('KNN', train_acccuracy, test_accuracy)

    # class weight helps class imbalance, higher weight given to smaller class
    classifier = RandomForestClassifier(n_estimators=150,
                                        class_weight='balanced',
                                        random_state=seed)
    classifier.fit(x_train, y_train)
    train_accuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('Random Forest: ', train_accuracy, test_accuracy)
    # print('Feature importances: ', classifier.feature_importances_)
    # y_pred = classifier.predict(x_test)
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # for label, value in zip(['TN', 'FP', 'FN', 'TP'], [tn, fp, fn, tp]):
    #     print(label, value)

    # voting classifier
    classifier = VotingClassifier(estimators=[
        ('svm', SVC(gamma=2, random_state=seed, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=150,
                                      class_weight='balanced',
                                      random_state=seed)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(50, 10), max_iter=1000,
                              shuffle=True))
    ])
    classifier.fit(x_train, y_train)
    train_accuracy = classifier.score(x_train, y_train)
    test_accuracy = classifier.score(x_test, y_test)
    print('Voting classifier: ', train_accuracy, test_accuracy)

    # random_grid_search = RandomizedSearchCV(
    #     RandomForestClassifier(class_weight='balanced', random_state=seed),
    #     {
    #         'n_estimators': [50, 100, 200, 300, 400, 500],
    #         'max_depth': [2, 4, 8, 16, 32, 64, 128],
    #         'max_features': list(range(1, NUM_FEATURES+1))
    #     },
    #     n_iter=50,  # number of random searches to do
    #     cv=5
    # )
    # random_grid_search.fit(x_train, y_train)
    # print('Best score: ', random_grid_search.best_score_)
    # print('Best params: ', random_grid_search.best_params_)

    # # grid search for random forest
    # params = {
    #     'n_estimators': [50, 100, 200, 300, 400, 500],
    #     'max_depth': [2, 4, 8, 16, 32, 64, 128],
    #     'max_features': list(range(1, NUM_FEATURES+1))
    # }
    # grid_search = GridSearchCV(RandomForestClassifier(
    #     class_weight='balanced', random_state=seed
    # ), params, cv=5)
    # grid_search.fit(x_train, y_train)
    # print('Best score: ', grid_search.best_score_)
    # print(f'Best parameters are: {grid_search.best_params_}\n')
    # best_model = grid_search.best_estimator_
    # test_accuracy = best_model.score(x_test, y_test)
    # print(test_accuracy)

    # # feature selection univariate chi squared (non-negative only)
    # bestfeatures = SelectKBest(score_func=chi2, k=10)
    # fit = bestfeatures.fit(x, y)
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(x.columns)
    # # concat two dataframes for better visualization
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print(featureScores.nlargest(10, 'Score'))

    # # gaussian mixture model
    # gmm = GaussianMixture(n_components=2)
    # gmm.fit(x_train, y_train)
    # y_preds = gmm.predict(x_test)
    # test_accuracy = accuracy_score(y_test, y_preds)
    # print('GMM: ', test_accuracy)

    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # scores = cross_val_score(RandomForestClassifier(), x, y,
    #                          scoring='roc_auc', cv=cv)
    # print('Mean ROC AUC: %.3f' % np.mean(scores))
    #
    # # grid search for class weights
    # # we want to give a higher weight to True/1 class
    # balance = [{0: 1, 1: 1}, {0: 1, 1: 10}, {0: 2, 1: 3}, {0: 1, 1:1000}]
    # param_grid = dict(class_weight=balance)
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # grid = GridSearchCV(estimator=RandomForestClassifier(),
    #                     param_grid=param_grid, n_jobs=-1,
    #                     cv=cv, scoring='roc_auc')
    # grid_result = grid.fit(x, y)
    # print("Best: %f using %s" % (grid_result.best_score_,
    #                              grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))


def main(args):
    run_type = args.run_type

    if run_type == 'initialise_dataset':
        initialise_dataset()
    elif run_type == 'extract_video_features':
        extract_video_features()
    elif run_type == 'train_classifier':
        train_classifier(args.dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('initialise_dataset')

    parser_2 = sub_parsers.add_parser('extract_video_features')

    parser_3 = sub_parsers.add_parser('train_classifier')
    parser_3.add_argument('--dataset_path', default=EGAN_FEATURES_DATASET_PATH)

    main(parser.parse_args())
