import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from main.models import Config
from main.research.test_update_list_5 import get_test_data
from main.utils.dtw import DTW
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statistics import mode

DATA_PATH = 'user_12_data.pkl'


def matrix_to_vector_1(m):
    delta_num_frames = 2
    sigma_t_squared = (delta_num_frames * (delta_num_frames + 1)
                       * (2 * delta_num_frames + 1)) / 3

    def window(_m):
        num_rows = len(_m)

        if num_rows == 1:
            return _m[0]

        new_m = []
        for i in range(num_rows - 1):
            v_1, v_2 = m[i], m[i+1]
            new_v = (v_2 - v_1) / 2
            new_m.append(new_v)

        return window(new_m)

    return window(m)


def interpolate(m, max_rows):

    def window(_m):
        num_rows = len(_m)

        if num_rows == max_rows:
            return _m

        new_m = []
        for i in range(num_rows - 1):
            v_1, v_2 = _m[i], _m[i+1]
            new_v = (v_1 + v_2) / 2
            new_m.append(new_v)

        return window(new_m)

    return window(m)


def matrix_to_vector_2(m):
    m = StandardScaler().fit_transform(m)

    pca = PCA(n_components=1)

    p_c = pca.fit_transform(m.T)

    return p_c.T[0]


def pca(m, num_components=2):
    scaler = StandardScaler()
    pca = PCA(n_components=num_components)

    m = scaler.fit_transform(m)
    m_pca = pca.fit_transform(m)

    return m_pca


def reverse_pca(m, num_components=2):
    scaler = StandardScaler()
    pca = PCA(n_components=num_components)

    m = scaler.fit_transform(m.T)
    m_pca = pca.fit_transform(m)

    return m_pca.T


def pca_plot(data_point, num_components=2):
    label, t = data_point

    m_pca = pca(t.blob, num_components)

    plt.scatter(m_pca[:, 0], m_pca[:, 1])
    plt.title(label)
    plt.show()


def pca_plot_2(training_data, cluster_label, num_components=2):
    max_rows, max_columns = 3, 3
    rows, columns = 0, 0

    clusters = {}
    for label, template in training_data:
        cluster = clusters.get(label, [])
        cluster.append(template)
        clusters[label] = cluster

    fig, axs = plt.subplots(max_rows, max_columns)
    fig.tight_layout()

    cluster = clusters[cluster_label]
    for template in cluster[:max_rows * max_columns]:
        m_pca = pca(template.blob, num_components)
        axs[rows, columns].scatter(m_pca[:, 0], m_pca[:, 1])

        if columns == max_columns - 1:
            rows += 1
            columns = 0
        else:
            columns += 1

    plt.show()


def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T

    print(np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).shape)

    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)


def expectation_step(X, clusters):
    totals = np.zeros((X.shape[0], 1), dtype=np.float64)

    for cluster in clusters:
        pi_k = cluster['pi_k']
        mu_k = cluster['mu_k']
        cov_k = cluster['cov_k']

        gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)

        for i in range(X.shape[0]):
            totals[i] += gamma_nk[i]

        cluster['gamma_nk'] = gamma_nk
        cluster['totals'] = totals

    for cluster in clusters:
        cluster['gamma_nk'] /= cluster['totals']


def text_rank(m, max_iter=100):
    """
    https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/#comment-155652
    """
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity

    num_rows = m.shape[0]
    similarity_matrix = np.zeros((num_rows, num_rows))

    # get similarity score between frames
    for i in range(num_rows):
        for j in range(num_rows):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(
                    m[i].reshape(1, -1), m[j].reshape(1, -1)
                )[0][0]

    # to graph, nodes = frames, edges = similarity scores between frames
    nx_graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(nx_graph, max_iter=max_iter)

    return scores


def split_text_rank(m, num_splits=5, max_iter=100):
    """split video matrix into parts and run text rank over different parts"""
    split = 1 / num_splits
    num_rows = m.shape[0]

    per_split_amount = int(num_rows * split)
    start, end = 0, per_split_amount
    new_m = []
    for i in range(num_splits):
        sub_m = m[start:end, :]
        print(sub_m.shape)
        ranks = text_rank(sub_m, max_iter)
        max_rank = max(ranks, key=ranks.get)
        new_m.append(sub_m[max_rank])
        start = end
        end += per_split_amount

    ranks = text_rank(np.asarray(new_m), max_iter)
    max_rank = max(ranks, key=ranks.get)

    return new_m[max_rank]


def analyse_clusters(cluster_labels, actual_labels):
    clusters = {}
    for cluster_label, actual_label in zip(cluster_labels, actual_labels):
        cluster = clusters.get(cluster_label, [])
        cluster.append(actual_label)
        clusters[cluster_label] = cluster

    for cluster_label, cluster_values in clusters.items():
        print(f'{cluster_label}: {cluster_values}')

    return clusters


def dtw_knn(training_data, test_data):
    dtw = DTW(**Config().__dict__)

    accuracy = 0
    for test_label, test_template in test_data:
        train_blobs = [(train[0], train[1].blob) for train in training_data]
        predictions = dtw.classify(test_template.blob, train_blobs, None)
        top_prediction_label = predictions[0]['label']
        if test_label == top_prediction_label:
            accuracy += 1

    accuracy = (accuracy * 100) / len(test_data)

    return accuracy


def dtw_gmm(training_data, test_data):
    """Works quite well but what's the application?"""
    num_samples = len(training_data)

    dtw_matrix = np.zeros((num_samples, num_samples), dtype=np.float32)

    dtw = DTW(**Config().__dict__)

    # TODO: This could be sped up because distance(0, 1) == distance(1, 0)

    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                signal_1 = training_data[i][1].blob.astype(np.float32)
                signal_2 = training_data[j][1].blob.astype(np.float32)
                dtw_distance = dtw.calculate_distance(signal_1, signal_2)
                dtw_matrix[i][j] = dtw_distance

    training_labels = [data_point[0] for data_point in training_data]

    model = GaussianMixture(n_components=20)
    model.fit(dtw_matrix)
    cluster_labels = model.predict(dtw_matrix)

    # model = KMeans(n_clusters=20)
    # model.fit(dtw_matrix)
    # cluster_labels = model.predict(dtw_matrix)

    clusters = analyse_clusters(cluster_labels, training_labels)
    cluster_labels = {}
    for cluster_label, cluster in clusters.items():
        try:
            cluster_labels[cluster_label] = mode(cluster)
        except Exception:
            cluster_labels[cluster_label] = cluster[0]

    # now run it through the test data
    accuracy = 0
    for test_label, test_template in test_data:
        data_point = np.array([
            dtw.calculate_distance(test_template.blob.astype(np.float32),
                                   train_template[1].blob.astype(np.float32))
            for train_template in training_data
        ])
        prediction = model.predict(data_point.reshape(1, -1))[0]
        cluster_label = cluster_labels[prediction]

        if test_label == cluster_label:
            accuracy += 1

    accuracy = (accuracy * 100) / len(test_data)

    return accuracy


def dtw_clustering():
    from main.research.test_update_list_3 import get_user_sessions
    from main.research.test_update_list_5 import DATA_PATH

    # grab the sessions
    sessions = get_user_sessions(DATA_PATH.format(user_id='12'))

    # training sessions, test sessions
    random.shuffle(sessions)
    train_split = int(len(sessions) * 0.6)
    training_sessions = sessions[:train_split]
    test_sessions = sessions[train_split:]

    # grab dtw distances for every training sample
    training_samples = [
        t for session in training_sessions
        for t in session[1]
    ]
    num_samples = len(training_samples)
    dtw_matrix = np.zeros((num_samples, num_samples), dtype=np.float32)
    dtw = DTW(**Config().__dict__)
    for i in range(num_samples):
        for j in range(num_samples):
            if i != j:
                signal_1 = training_samples[i][1].blob.astype(np.float32)
                signal_2 = training_samples[j][1].blob.astype(np.float32)
                dtw_distance = dtw.calculate_distance(signal_1, signal_2)
                dtw_matrix[i][j] = dtw_distance

    training_labels = [t[0] for t in training_samples]

    # model = GaussianMixture(n_components=20)
    model = KMeans(n_clusters=20)
    model.fit(dtw_matrix)
    cluster_labels = model.predict(dtw_matrix)

    clusters = analyse_clusters(cluster_labels, training_labels)


def clustering(training_data):
    training_labels = [data_point[0] for data_point in training_data]

    # first: make templates same length
    minimum_length = int(min([len(data_point[1].blob)
                              for data_point in training_data]))
    training_data = np.array([
        interpolate(data_point[1].blob, minimum_length)
        for data_point in training_data
    ])
    print(minimum_length)
    print(training_data.shape)

    # second: run pca
    training_data = np.array([
        pca(data_point, num_components=20).flatten()
        for data_point in training_data
    ])
    print(training_data.shape)

    gmm = GaussianMixture(n_components=20)
    gmm.fit(training_data)
    cluster_labels = gmm.predict(training_data)
    analyse_clusters(cluster_labels, training_labels)

    kmeans = KMeans(n_clusters=20)
    kmeans.fit(training_data)
    cluster_labels = kmeans.predict(training_data)
    analyse_clusters(cluster_labels, training_labels)


def main():
    dtw_clustering()

    # if os.path.exists(DATA_PATH):
    #     with open(DATA_PATH, 'rb') as f:
    #         train_test_data = pickle.load(f)
    # else:
    #     train_test_data = get_test_data(user_id='12')
    #     with open(DATA_PATH, 'wb') as f:
    #         pickle.dump(train_test_data, f)
    #
    # # do train, test split
    # random.shuffle(train_test_data)
    # train_split = int(len(train_test_data) * 0.6)
    # training_data = train_test_data[:train_split]
    # test_data = train_test_data[train_split:]
    # print('Training Data: ', len(training_data))
    # print('Test Data: ', len(test_data))
    # assert len(training_data) + len(test_data) == len(train_test_data)
    #
    # training_labels = [data_point[0] for data_point in training_data]

    # training_data = np.array([data_point[1].blob
    #                           for data_point in training_data])

    # training_data = np.array([
    #     reverse_pca(data_point[1].blob, num_components=50)
    #     for data_point in training_data
    # ])

    # training_data = [np.mean(data_point[1].blob, axis=0)
    #                  for data_point in training_data]

    # training_data = [matrix_to_vector_1(data_point[1].blob)
    #                  for data_point in training_data]

    # training_data = [matrix_to_vector_2(data_point[1].blob)
    #                  for data_point in training_data]

    # training_data = [split_text_rank(data_point[1].blob, num_splits=50,
    #                                  max_iter=10000)
    #                  for data_point in training_data]

    # training_data = [
    #     reverse_pca(data_point[1].blob, num_components=10).flatten()
    #     for data_point in training_data
    # ]

    # training_data = [
    #     pca(data_point[1].blob, num_components=1)[0].T
    #     for data_point in training_data
    # ]

    # gmm = GaussianMixture(n_components=20)
    # gmm.fit(training_data, training_labels)
    # cluster_labels = gmm.predict(training_data)
    #
    # clusters = {}
    # for cluster_label, training_label in zip(cluster_labels, training_labels):
    #     cluster = clusters.get(cluster_label, [])
    #     cluster.append(training_label)
    #     clusters[cluster_label] = cluster
    #
    # for cluster_label, cluster_values in clusters.items():
    #     print(f'{cluster_label}: {cluster_values}')

    # pca_plot(training_data[0])
    #
    # dataset = []
    # for label, template in training_data:
    #     m_pca = pca(template.blob)
    #     print(m_pca)
    #     # print(m_pca.mean(axis=0))
    #     dataset.append(m_pca.mean(axis=0))
    #
    # print(np.asarray(dataset).shape)
    # dataset = np.asarray(dataset)
    # print(dataset[:, 0])
    # plt.scatter(dataset[:, 0], dataset[:, 1])
    # plt.show()

    # clusters = []
    # n_clusters = 20
    # for i in range(n_clusters):
    #     clusters.append({
    #         'pi_k': 1.0 / n_clusters,
    #         'mu_k': np.random.rand(50, 566),
    #         'cov_k': [np.identity(X.shape[2], dtype=np.float64) for i in range(50)]
    #     })
    #
    # expectation_step(X, clusters)

    # recursive_text_rank(training_data[0][1].blob)
    # text_rank(training_data[0][1].blob)
    # split_text_rank(training_data[0][1].blob)

    # m = reverse_pca(training_data[0][1].blob, num_components=100)
    # print(m.shape)
    # print(m.flatten().shape)

    # db = DBSCAN(eps=0.3, min_samples=10).fit(training_data)
    # labels = db.labels_
    # print(labels)

    # pca_plot_2(training_data, "Can I have a cough?")

    # dtw_gmm(training_data)

    # clustering(training_data)

    # rank_accuracies = dtw_knn(training_data, test_data)
    # print(rank_accuracies)
    #
    # accuracy = dtw_gmm(training_data, test_data)
    # print(accuracy)

    # m_pca = pca(training_data[0][1].blob, num_components=2)
    # print(m_pca.shape)
    # print(text_rank(m_pca[:150], max_iter=500))


if __name__ == '__main__':
    main()
