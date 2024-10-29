"""Experiments to prevent users from saying phrases not in the list"""
import argparse
import ast
import glob
import os

import cv2
import featuretools as ft
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main import configuration
from main.models import Config
from main.research.phrase_list_composition import WORDS_TO_VISEMES
from main.research.research_utils import create_sessions, create_template, \
    sessions_to_templates, create_templates, create_templates_doctors
from main.utils.confusion_matrix import ConfusionMatrix
from main.utils.dtw import DTW
from main.utils.knn import KNN
from main.utils.io import read_csv_file, read_pickle_file, write_pickle_file
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import LocalOutlierFactor

DTW_PARAMS = Config().__dict__
DTW = DTW(**DTW_PARAMS)
KNN = KNN(**DTW_PARAMS)


def get_dtw_data(test_signal, default_templates, n):
    distances = [DTW.calculate_distance(
        test_signal=test_signal,
        ref_signal=ref_template[1].astype(np.float32)
    ) for ref_template in default_templates]
    assert len(distances) == len(default_templates)

    # sorted in ascending order (smallest/closest first)
    sorted_indexes = np.argsort(distances)
    top_n_indexes = sorted_indexes[:n]

    return distances, top_n_indexes


def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    return duration


def get_num_phonemes(word):
    return len(
        WORDS_TO_VISEMES[word.lower().replace('?', '').replace('!', '')]
    )


def get_top_n_phrases_and_distances(phrases, distances, indexes):
    return np.array(phrases)[indexes].tolist(), \
           np.array(distances)[indexes].tolist()


def get_top_n_template_lengths(templates, indexes):
    return [
        t[1].shape[0]
        for t in np.array(templates)[indexes]
    ]


def get_phoneme_lengths(phrases):
    return [
        sum([get_num_phonemes(word) for word in phrase.split(' ')])
        for phrase in phrases
    ]


def get_signal_length(signal):
    return signal.shape[0]


def get_num_unique_phrases(phrases):
    return len(set(phrases))


def get_median_ratio(x, xs):
    return x / np.median(xs)


def get_default_templates():
    default_list = read_pickle_file(configuration.DEFAULT_LIST_PATH)
    default_templates = default_list.get_reference_signals()
    default_template_labels = [t[0] for t in default_templates]

    return default_templates, default_template_labels


def create_dataset(args):
    """Cluster the default templates.
    GOAL: Find centroids that minimise intra but maximise inter class distance
    Create dataset from these centroids
    """
    # get default templates
    default_templates, default_template_labels = get_default_templates()
    default_phrases = list(np.unique(default_template_labels))

    # construct phrase templates clusters
    get_phrase_templates = \
        lambda phrase: [t[1].astype(np.float32)
                        for t in default_templates if t[0] == phrase]
    clusters_d = {
        phrase: get_phrase_templates(phrase)
        for phrase in default_phrases
    }

    # find the best centroids
    centroids_d = {}

    # template with max distance to all others becomes phrase centroid
    # max distance = distinguishable
    if not os.path.exists('default_centroids.pkl'):
        for phrase, templates in clusters_d.items():
            cluster_d_copy = clusters_d.copy()
            del cluster_d_copy[phrase]
            distance_sums = []
            for test_template in templates:
                distances = []
                for other_phrase, other_templates in cluster_d_copy.items():
                    distances.extend([
                        DTW.calculate_distance(test_template, ref_template)
                        for ref_template in other_templates
                    ])
                distance_sums.append(sum(distances))
            max_distance_sum_index = distance_sums.index(max(distance_sums))
            centroids_d[phrase] = templates[max_distance_sum_index]
        default_centroids = [
            (phrase, template) for phrase, template in centroids_d.items()
        ]
        write_pickle_file(default_centroids, 'default_centroids.pkl')
    else:
        default_centroids = read_pickle_file('default_centroids.pkl')

    # now create dataset to determine if phrase in the list or not
    in_list_phrases = [f'P{i}' for i in range(1, 21)]
    in_list_phrases += default_phrases

    KNN.top_n = 20

    regexes = [
        r'SRAVIExtended-(.+)-P(\d+)-S(\d+)',
        r'SRAVIExtended(.+)P(\d+)-S(\d+)',
        r'PV0*(\d+)_P10*(\d+)_S0*(\d+)',
    ]

    for base_videos_directory in args.test_directories:
        user_ids = os.listdir(base_videos_directory)
        for user_id in user_ids:
            if '.zip' in user_id: continue
            videos_directory = os.path.join(base_videos_directory, user_id)
            user_id = f'{base_videos_directory.split("/")[-1]}_{user_id}'

            if args.no_sessions:
                user_templates = create_templates(
                    videos_directory,
                    regexes=None,
                    save=True,
                    include_video_paths=True
                )
            elif args.doctors:
                user_templates = create_templates_doctors(
                    videos_directory,
                    save=True,
                    include_video_paths=True
                )
            else:
                user_sessions = create_sessions(
                    videos_directory,
                    regexes,
                    save=True,
                    include_video_paths=True
                )
                user_templates = sessions_to_templates(user_sessions)

            for gt_label, test_signal, test_video_path in user_templates:
                test_signal = test_signal.astype(np.float32)

                in_list = 1 if gt_label in in_list_phrases else 0

                # get distances to default centroids
                centroid_distances = np.asarray([
                    DTW.calculate_distance(test_signal, ref_signal)
                    for phrase, ref_signal in default_centroids
                ])
                centroid_ds_sorted_indices = np.argsort(centroid_distances)

                # get top classification phrases and distances
                distances, top_n_indexes = get_dtw_data(
                    test_signal,
                    default_templates,
                    args.n
                )

                # get top n phrases and distances
                top_n_phrases, top_n_distances = \
                    get_top_n_phrases_and_distances(
                        default_template_labels,
                        distances,
                        top_n_indexes
                    )

                # get KNN predictions
                KNN.fit(np.asarray(distances),
                        np.asarray(default_template_labels))
                top_predictions = KNN.predict()
                top_3_predictions = top_predictions[:3]

                # get centroid template lengths
                centroid_template_lengths = [
                    ref_signal.shape[0]
                    for phrase, ref_signal in default_centroids
                ]

                # get top n template lengths
                top_n_signal_lengths = get_top_n_template_lengths(
                    default_templates,
                    top_n_indexes
                )

                # get test signal length features
                test_signal_length = test_signal.shape[0]
                test_signal_length_ratio_to_median_centroid_lengths = \
                    test_signal_length / np.median(centroid_template_lengths)

                # get video length
                test_video_length = get_video_length(test_video_path)
                test_video_length_ratio_to_median_centroid_lengths = \
                    test_video_length / np.median(centroid_template_lengths)

                # positions of top 3 prediction phrases in the
                # centroid distances
                top_3_prediction_phrases = [pred['label']
                                            for pred in top_3_predictions]
                centroid_phrases = \
                    np.asarray([phrase
                                for phrase, signal in default_centroids])
                sorted_centroid_phrases = \
                    centroid_phrases[centroid_ds_sorted_indices].tolist()
                top_phrase_positions_in_centroid_distances = [
                    sorted_centroid_phrases.index(top_phrase)
                    for top_phrase in top_3_prediction_phrases
                ]

                with open('phrases_in_list_clustering_dataset.csv', 'a') as f:
                    f.write(f'{user_id},'
                            f'{gt_label},'
                            f'{centroid_distances.tolist()},'
                            f'{top_n_phrases},'
                            f'{top_n_distances},'
                            f'{top_predictions},'
                            f'{top_phrase_positions_in_centroid_distances},'
                            f'{top_n_signal_lengths},'
                            f'{test_signal_length},'
                            f'{test_signal_length_ratio_to_median_centroid_lengths},'
                            f'{test_video_length},'
                            f'{test_video_length_ratio_to_median_centroid_lengths},'
                            f'{in_list}\n'
                            )


def auto_feature_engineering(X, y, selection_percent=0.1,
                             selection_strategy="best", num_depth_steps=2,
                             transformatives=['divide_numeric',
                                              'multiply_numeric']):
    """
    Automatically perform deep feature engineering and
    feature selection.

    Parameters
    ----------
    X : pd.DataFrame
        Data to perform automatic feature engineering on.
    y : pd.DataFrame
        Target variable to find correlations of all
        features at each depth step to perform feature
        selection, y is not needed if selection_percent=1.
    selection_percent : float, optional
        Defines what percent of all the new features to
        keep for the next depth step.
    selection_strategy : {'best', 'random'}, optional
        Strategy used for feature selection, if 'best',
        it will select the best features for the next depth
        step, if 'random', it will select features at random.
    num_depth_steps : integer, optional
        The number of depth steps. Every depth step, the model
        generates brand new features from the features made in
        the last step, then selects a percent of these new
        features.
    transformatives : list, optional
        List of all possible transformations of the data to use
        when feature engineering, you can find the full list
        of possible transformations as well as what each one
        does using the following code:
        `ft.primitives.list_primitives()[ft.primitives.list_primitives()["type"]=="transform"]`
        make sure to `import featuretools as ft`.

    Returns
    -------
    pd.DataFrame
        a dataframe of the brand new features.
    """
    from sklearn.feature_selection import mutual_info_classif
    selected_feature_df = X.copy()
    for i in range(num_depth_steps):

        # Perform feature engineering
        es = ft.EntitySet()
        es.entity_from_dataframe(entity_id='data',
                                 dataframe=selected_feature_df,
                                 make_index=True, index='index')
        feature_matrix, feature_defs = ft.dfs(entityset=es,
                                              target_entity='data',
                                              trans_primitives=transformatives,
                                              verbose=True)

        # Remove features that are the same
        feature_corrs = feature_matrix.corr()[list(feature_matrix.keys())[0]]

        existing_corrs = []
        good_keys = []
        for key in feature_corrs.to_dict().keys():
            if feature_corrs[key] not in existing_corrs:
                existing_corrs.append(feature_corrs[key])
                good_keys.append(key)
        feature_matrix = feature_matrix[good_keys]

        # Remove illegal features
        legal_features = list(feature_matrix.columns)
        for feature in list(feature_matrix.columns):
            raw_feature_list = []
            for j in range(len(feature.split(" "))):
                if j % 2 == 0:
                    raw_feature_list.append(feature.split(" ")[j])
            if len(
                    raw_feature_list) > i + 2:  # num_depth_steps = 1, means max_num_raw_features_in_feature = 2
                legal_features.remove(feature)
        feature_matrix = feature_matrix[legal_features]

        # Perform feature selection
        if int(selection_percent) != 1:
            if selection_strategy == "best":
                corrs = mutual_info_classif(
                    feature_matrix.reset_index(drop=True), y)
                corrs = pd.Series(corrs, name="")
                selected_corrs = corrs[
                    corrs >= corrs.quantile(1 - selection_percent)]
                selected_feature_df = feature_matrix.iloc[:,
                                      list(selected_corrs.keys())].reset_index(
                    drop=True)
            elif selection_strategy == "random":
                selected_feature_df = feature_matrix.sample(
                    frac=(selection_percent), axis=1).reset_index(drop=True)
            else:
                raise Exception(
                    "selection_strategy can be either 'best' or 'random', got '" + str(
                        selection_strategy) + "'.")
        else:
            selected_feature_df = feature_matrix.reset_index(drop=True)
        if num_depth_steps != 1:
            rename_dict = {}
            for col in list(selected_feature_df.columns):
                rename_dict[col] = "(" + col + ")"
            selected_feature_df = selected_feature_df.rename(
                columns=rename_dict)
    if num_depth_steps != 1:
        rename_dict = {}
        for feature_name in list(selected_feature_df.columns):
            rename_dict[feature_name] = feature_name[
                                        int(num_depth_steps - 1):-int(
                                            num_depth_steps - 1)]
        selected_feature_df = selected_feature_df.rename(columns=rename_dict)

    return selected_feature_df


def dataset_analysis(args):
    df = read_csv_file(
        args.dataset_path,
        ['User ID',
         'GT Label',
         'C Distances',
         'Top N Phrases',
         'Top N Distances',
         'Top 3 Predictions',
         'Top Phrase Positions In Centroid Distances',
         'Top N Signal Lengths',
         'Test Signal Length',
         'Test Signal Length Ratio To Median Centroid Lengths',
         'Test Video Length',
         'Test Video Length Ratio To Median Centroid Lengths',
         'Label'],
        r'(.+),(.+),(\[.+\]),(\[.+\]),(\[.+\]),(\[.+\]),(\[.+\]),(\[.+\]),'
        r'(\d+),(.+),(.+),(.+),(.+)',
        lambda row: [
            row[0],
            row[1],
            ast.literal_eval(row[2]),
            ast.literal_eval(row[3]),
            ast.literal_eval(row[4]),
            eval(row[5]),
            ast.literal_eval(row[6]),
            ast.literal_eval(row[7]),
            int(row[8]),
            float(row[9]),
            float(row[10]),
            float(row[11]),
            int(row[12])
        ]
    )

    num_c_distances = 20
    n = 10
    num_predictions = 5

    # create features
    for i in range(num_predictions):
        df[f'Prediction Confidence {i+1}'] = df.apply(
            lambda row: row['Top 3 Predictions'][i]['accuracy'],
            axis=1
        )
    for i in range(num_predictions-1):
        for j in range(i+1, num_predictions):
            df[f'Prediction Confidence Range {i+1}-{j+1}'] = df.apply(
                lambda row: row[f'Prediction Confidence {i+1}'] -
                            row[f'Prediction Confidence {j+1}'],
                axis=1
            )
    df['Prediction Confidence Sum'] = df.apply(
        lambda row: sum([row[f'Prediction Confidence {i+1}']
                         for i in range(num_predictions)]),
        axis=1
    )
    df['C Distances'] = df.apply(
        lambda row: sorted(row['C Distances']),
        axis=1
    )
    for i in range(num_c_distances)[:3]:
        df[f'C Distance {i+1}'] = df.apply(
            lambda row: row[f'C Distances'][i],
            axis=1
        )
    df['C Distance Sum'] = df.apply(
        lambda row: sum([
            row[f'C Distance {i+1}']
            for i in range(num_c_distances)[:3]
        ]),
        axis=1
    )
    sum_ranges = [(1, 5), (1, 4), (1, 3), (1, 2)]
    for i, j in sum_ranges:
        df[f'Prediction Confidence Sum {i}-{j}'] = df.apply(
            lambda row: sum([
                row[f'Prediction Confidence {k}']
                for k in range(i, j+1)
            ]),
            axis=1
        )
    for i, j in sum_ranges:
        df[f'Prediction Confidence Range/Sum {i}-{j}'] = df.apply(
            lambda row: (row[f'Prediction Confidence Range {i}-{j}'] / row[f'Prediction Confidence Sum {i}-{j}']),
            axis=1
        )
        df[f'Prediction Confidence Range*Sum {i}-{j}'] = df.apply(
            lambda row: (row[f'Prediction Confidence Range {i}-{j}'] * row[f'Prediction Confidence Sum {i}-{j}']),
            axis=1
        )
    df['Prediction Confidence 1 > 40'] = df.apply(
        lambda row: row['Prediction Confidence 1'] > 0.4,
        axis=1
    )
    df['Num Unique Phrases'] = df.apply(
        lambda row: len(set(row['Top N Phrases'])),
        axis=1
    )
    for i in range(n):
        df[f'Top N Distance {i+1}'] = df.apply(
            lambda row: row['Top N Distances'][i],
            axis=1
        )
    for i in range(n-1):
        for j in range(i+1, n):
            df[f'Top N Distance Range {i+1}-{j+1}'] = df.apply(
                lambda row: row[f'Top N Distance {i+1}'] -
                            row[f'Top N Distance {j+1}'],
                axis=1
            )
    df['Top N Distance Sum'] = df.apply(
        lambda row: sum([
            row[f'Top N Distance {i+1}'] for i in range(n)
        ]),
        axis=1
    )

    # feature correlation
    corr_matrix = df.corr()
    print(corr_matrix['Label'].sort_values(ascending=False).to_string())

    # attributes = [
    #     'Label',
    #     'Prediction Confidence Sum 1-2',
    #     'Prediction Confidence Range 1-5',
    #     'Prediction Confidence Range 1-4'
    # ]
    # from pandas.plotting import scatter_matrix
    # scatter_matrix(df[attributes])
    # plt.show()

    features = [
        *[f'Prediction Confidence {i+1}' for i in range(num_predictions)],
        *[f'Prediction Confidence Range {i+1}-{j+1}'
          for i in range(num_predictions - 1)
          for j in range(i + 1, num_predictions)],
        *['Prediction Confidence Sum'],
        # *[f'Prediction Confidence Sum {i}-{j}' for i, j in sum_ranges]
    ]

    # print class balances
    # print(df['GT Label'].value_counts().to_string())

    X = df[features]
    Y = df['Label']
    PHRASE_LABELS = df['GT Label']

    # min-max normalisation
    scaler = preprocessing.MinMaxScaler()
    X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns)

    # polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly_1 = PolynomialFeatures(3)
    to_poly_features = [f'Prediction Confidence {i+1}'
                        for i in range(num_predictions)]
    to_poly_features += ['Prediction Confidence Sum']
    to_poly_X = X.loc[:, to_poly_features]
    to_poly_X = poly_1.fit_transform(to_poly_X)
    num_new_features = to_poly_X.shape[1]
    columns = [f'PC {i+1} Poly 3' for i in range(num_new_features)]
    X = pd.concat([X, pd.DataFrame(to_poly_X, columns=columns)], axis=1)

    # polynomial features
    poly_2 = PolynomialFeatures(3)
    to_poly_features = [f'Prediction Confidence Range {i+1}-{j+1}'
                        for i in range(num_predictions-1)
                        for j in range(i+1, num_predictions)]
    to_poly_features += ['Prediction Confidence Sum']
    to_poly_X = X.loc[:, to_poly_features]
    to_poly_X = poly_2.fit_transform(to_poly_X)
    num_new_features = to_poly_X.shape[1]
    columns = [f'PCR {i+1} Poly 3' for i in range(num_new_features)]
    X = pd.concat([X, pd.DataFrame(to_poly_X, columns=columns)], axis=1)

    # # polynomial features
    # poly_3 = PolynomialFeatures(3)
    # to_poly_features = [f'Prediction Confidence Sum {i}-{j}' for i, j in sum_ranges]
    # to_poly_X = X.loc[:, to_poly_features]
    # to_poly_X = poly_3.fit_transform(to_poly_X)
    # num_new_features = to_poly_X.shape[1]
    # columns = [f'Prediction Confidence Sum {i+1} Poly 3'
    #            for i in range(num_new_features)]
    # X = pd.concat([X, pd.DataFrame(to_poly_X, columns=columns)], axis=1)

    # # feature correlation
    # df_copy = pd.concat([X, Y], axis=1)
    # corr_matrix = df_copy.corr()
    # print(corr_matrix['Label'].sort_values(ascending=False))

    # # apply SelectKBest class to extract top 10 best features
    # print('Running SelectKBest 1:')
    # from sklearn.feature_selection import SelectKBest
    # from sklearn.feature_selection import chi2
    # bestfeatures = SelectKBest(score_func=chi2, k=10)
    # fit = bestfeatures.fit(X, Y)
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(X.columns)
    # # concat two dataframes for better visualization
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features
    # # features = featureScores.nlargest(100, 'Score')['Specs']
    # # X = X[features]

    # # # deep feature synthesis
    # # print(ft.primitives.list_primitives()[ft.primitives.list_primitives()['type'] == 'transform'])
    # X = auto_feature_engineering(X, Y, selection_percent=1,
    #                              num_depth_steps=4,
    #                              transformatives=[
    #                                  'multiply_numeric',
    #                                  'percentile',
    #                                  'add_numeric'
    #                              ])
    # print('Finished Auto Feature Engineering')
    #
    # if isinstance(X, tuple):
    #     X = X[0]
    # X = X.reset_index(drop=True)  # drop=True removes index column

    # min-max normalisation
    # X = (X - X.min()) / (X.max() - X.min())

    # fill NaN/Inf values
    # print(np.where(X.values >= np.finfo(np.float64).max))
    # print(np.where(X.values >= np.Inf))
    # print(np.isnan(X.values.any()))
    # print(X.isnull().any())
    # X.fillna(X.mean(), inplace=True)
    # X[X == X.inf] = np.nan
    # X.fillna(X.mean(), inplace=True)

    # # normalising features
    # # scaler = preprocessing.StandardScaler()
    # scaler = preprocessing.MinMaxScaler()
    # X = pd.DataFrame(data=scaler.fit_transform(X), columns=X.columns)

    # # apply SelectKBest class to extract top 10 best features
    # print('Running SelectKBest 2:')
    # from sklearn.feature_selection import SelectKBest
    # from sklearn.feature_selection import chi2
    # bestfeatures = SelectKBest(score_func=chi2, k=10)
    # fit = bestfeatures.fit(X, Y)
    # dfscores = pd.DataFrame(fit.scores_)
    # dfcolumns = pd.DataFrame(X.columns)
    # # concat two dataframes for better visualization
    # featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # print(featureScores.nlargest(10, 'Score'))  # print 10 best features
    # # features = featureScores.nlargest(300, 'Score')['Specs']
    # # X = X[features]

    # # dimensionality reduction
    # from sklearn.decomposition import PCA
    # num_components = 10
    # pca = PCA(n_components=num_components)
    # pca_results = pca.fit_transform(X)
    # X = pd.DataFrame(data=pca_results,
    #                  columns=[f'C{i + 1}' for i in range(num_components)])
    #
    # # min-max normalisation
    # X = (X - X.min()) / (X.max() - X.min())

    print(X, '\n', Y)

    test_user = args.test_user_id
    if args.include_only:
        training_users = args.include_only
    else:
        excluded_users = args.excluded_users
        excluded_users += ['1_initial_2', '1_initial_3', '1_initial_7',
                           '2_lighting_2', '2_lighting_3', '3_more_phrases_AP']
        training_users = list(set(df['User ID']) - {test_user} -
                              set(excluded_users))
        training_users = [user_id for user_id in training_users
                          if 'doctor' not in user_id]  # exclude doctors

    train_x = X[df['User ID'].isin(training_users)]
    train_y = Y[df['User ID'].isin(training_users)]
    test_x = X[df['User ID'] == test_user]
    test_y = Y[df['User ID'] == test_user]
    test_phrase_labels = PHRASE_LABELS[df['User ID'] == test_user]

    # outlier detection
    train_x = train_x[train_y == 1]
    train_y = train_y[train_y == 1]
    test_y[test_y == 0] = -1  # Outlier Prediction: -1

    print('Training/Test Counts:', len(train_x), len(test_x))
    print('Training Label Counts:\n', train_y.value_counts())
    print('Test Label Counts:\n', test_y.value_counts())

    num_test_neg = len(test_y[test_y == -1])
    num_test_pos = len(test_y[test_y == 1])

    print('*********************************************************')
    print('SVM')

    # SVM - good for novelty detection (when samples are clean - no outliers)
    from sklearn.svm import OneClassSVM
    # One class classification SVM
    # nu = % of outlier samples
    if num_test_neg:
        nu = num_test_pos / num_test_neg
    else:
        nu = 0.2
    svm_clf = OneClassSVM(nu=nu, kernel='poly', degree=3)
    svm_clf.fit(train_x)  # fit on majority class
    if test_user:
        predictions = svm_clf.predict(test_x)
        print(pd.crosstab(test_y, predictions, rownames=['True'],
                          colnames=['Predicted'], margins=True))
        print(classification_report(test_y, predictions))

        incorrect_count = {}
        for pred, gt, phrase_label in zip(predictions, test_y,
                                          test_phrase_labels):
            if pred != gt:
                incorrect_count[phrase_label] = \
                    incorrect_count.get(phrase_label, 0) + 1
        print(sorted(incorrect_count.items(), key=lambda x: x[1], reverse=True))

    print('*********************************************************')
    print('ISO F')

    # Isolation Forest
    iso_clf = IsolationForest(n_estimators=500, random_state=0)
    iso_clf.fit(train_x, train_y)
    predictions = iso_clf.predict(test_x)
    print(pd.crosstab(test_y, predictions, rownames=['True'],
                      colnames=['Predicted'], margins=True))
    print(classification_report(test_y, predictions))

    print('*********************************************************')
    print('GMM')

    # GMM
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=20, random_state=0)
    gmm.fit(train_x)
    positive_score = gmm.score_samples(train_x)
    threshold = positive_score.mean() - (3 * positive_score.std())
    # threshold = np.percentile(positive_score, 4)
    if test_user:
        test_score = gmm.score_samples(test_x)
        test_pos_acc = len(test_score[(test_y == 1) & (test_score > threshold)]) / len(test_score[(test_y == 1)])  # pos recall
        if num_test_neg:
            test_neg_acc = len(test_score[(test_y == -1) & (test_score < threshold)]) / len(test_score[(test_y == -1)])  # neg recall
        else:
            test_neg_acc = 0
        print(test_pos_acc, test_neg_acc, threshold)
        # plt.scatter(range(len(test_x)), test_score,
        #             c=['skyblue' if x == 1 else 'pink' for x in test_y])
        # plt.axhline(threshold, color='black')
        # plt.show()

    print('*********************************************************')
    print('LOF')

    # lof_clf = LocalOutlierFactor(n_neighbors=20, novelty=True)
    # lof_clf.fit(train_x, train_y)
    # predictions = lof_clf.predict(test_x)
    # print(pd.crosstab(test_y, predictions, rownames=['True'],
    #                   colnames=['Predicted'], margins=True))
    # print(classification_report(test_y, predictions))

    print('*********************************************************')
    from sklearn.utils import resample

    print('Bagging SVM')
    num_classifiers = 5

    num_rows_each = len(train_x) // num_classifiers
    print('Num rows each:', num_rows_each)

    clfs = []
    for i in range(num_classifiers):
        sub_df = resample(train_x, n_samples=num_rows_each)
        clf = OneClassSVM(nu=nu, kernel='poly', degree=3)
        clf.fit(sub_df)
        clfs.append(clf)

    accuracy = 0
    num_test = len(test_x)
    for x, gt in zip(test_x.values, test_y.values):
        preds = [clf.predict(x.reshape(1, -1))[0] for clf in clfs]
        count = {1: 0, -1: 0}
        for pred in preds:
            count[pred] += 1
        winner = max(count, key=count.get)
        if winner == gt:
            accuracy += 1
    accuracy /= num_test
    print('Accuracy:', accuracy)

    # from sklearn.ensemble import BaggingClassifier
    # bag_clf = BaggingClassifier(
    #     OneClassSVM(nu=nu, kernel='poly', degree=3),
    #     n_estimators=100,
    #     max_samples=100,
    #     bootstrap=True,
    #     n_jobs=-1
    # )
    # bag_clf.fit(train_x, train_y)
    # predictions = bag_clf.predict(test_x)
    # print(pd.crosstab(test_y, predictions, rownames=['True'],
    #                   colnames=['Predicted'], margins=True))
    # print(classification_report(test_y, predictions))

    print('*********************************************************')

    # do some external testing using webcam or video directory
    if args.webcam_test or args.external_test:
        # default_centroids = read_pickle_file('default_centroids.pkl')
        default_templates, default_template_labels = get_default_templates()
        KNN.top_n = num_predictions

        if args.webcam_test:
            from main.utils.cam import Cam
            cam = Cam(debug=True)
            video_paths = cam.record_loop
        else:
            video_paths = lambda: glob.glob(os.path.join(
                args.external_test, '*.mp4'
            ))

        accuracy = 0
        num_tests = 0

        for video_path in video_paths():
            template = create_template(video_path, debug=True)
            if not template:
                continue

            # get features
            test_signal = template.blob.astype(np.float32)

            # # get distances to default centroids
            # centroid_distances = [
            #     DTW.calculate_distance(test_signal, ref_signal)
            #     for ref_signal in default_centroids
            # ]
            # centroid_distances = sorted(centroid_distances)

            # get top classification phrases and distances
            distances, top_n_indexes = get_dtw_data(
                test_signal,
                default_templates,
                args.n
            )

            # # get top n phrases and distances
            # top_n_phrases, top_n_distances = \
            #     get_top_n_phrases_and_distances(
            #         default_template_labels,
            #         distances,
            #         top_n_indexes
            #     )

            KNN.fit(
                np.asarray(distances),
                np.asarray(default_template_labels)
            )
            top_predictions = KNN.predict()
            top_confidences = [pred['accuracy'] for pred in top_predictions]
            top_confidences_ranges = [top_confidences[i] - top_confidences[j]
                                      for i in range(num_predictions-1)
                                      for j in range(i+1, num_predictions)]

            columns_1 = [f'PC {i+1}' for i in range(num_predictions)]
            columns_2 = [f'PCR {i+1}-{j+1}'
                         for i in range(num_predictions-1)
                         for j in range(i+1, num_predictions)]
            columns_3 = ['Prediction Confidence Sum']

            initial_features = pd.DataFrame(
                data=scaler.transform([[
                    *top_confidences,
                    *top_confidences_ranges,
                    sum(top_confidences)
                ]]),
                columns=[*columns_1, *columns_2, *columns_3]
            )

            poly_1_features = poly_1.transform(
                initial_features.loc[:, columns_1+columns_3]
            )
            poly_2_features = poly_2.transform(
                initial_features.loc[:, columns_2+columns_3]
            )

            features = pd.concat([
                initial_features,
                pd.DataFrame(poly_1_features),
                pd.DataFrame(poly_2_features)
            ], axis=1)

            prediction = svm_clf.predict(features)[0]
            if args.external_test and args.groundtruth:
                if prediction == args.groundtruth:
                    accuracy += 1
            prediction = 'In List' if prediction == 1 else 'Not In List'
            print(video_path, prediction)

            num_tests += 1

        accuracy /= num_tests
        print(accuracy)


def main(args):
    f = {
        'create_dataset': create_dataset,
        'dataset_analysis': dataset_analysis
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    def str_list(s):
        return s.split(',')

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_3 = sub_parsers.add_parser('create_dataset')
    parser_3.add_argument('test_directories', type=str_list)
    parser_3.add_argument('--n', type=int, default=10)
    parser_3.add_argument('--no_sessions', action='store_true')
    parser_3.add_argument('--phrase_set', default='PAVA-DEFAULT')
    parser_3.add_argument('--doctors', action='store_true')

    parser_4 = sub_parsers.add_parser('dataset_analysis')
    parser_4.add_argument('dataset_path')
    parser_4.add_argument('--test_user_id', default=None)
    parser_4.add_argument('--excluded_users', type=str_list, default=[])
    parser_4.add_argument('--webcam_test', action='store_true')
    parser_4.add_argument('--external_test')
    parser_4.add_argument('--groundtruth', type=int, default=None)
    parser_4.add_argument('--n', type=int, default=10)
    parser_4.add_argument('--include_only', type=str_list, default=[])

    main(parser.parse_args())
