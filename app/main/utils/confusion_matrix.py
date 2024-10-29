"""Confusion matrix.

Contains logic to plot a confusion matrix
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sn
from main.utils.io import write_pickle_file
from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    """Confusion Matrix class.

    Attributes:
        ground_truths (list): contains ground truth data for plotting
        predictions (list): contains corresponding prediction data to plot
    """

    def __init__(self):
        """Constructor."""
        self.ground_truths = []
        self.predictions = []

    def append(self, prediction, ground_truth):
        """Append a (prediction, ground-truth) pairing.

        Args:
            prediction (obj): prediction data
            ground_truth (obj): corresponding actual ground-truth

        Returns:
            None
        """
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)

    def extend(self, predictions, groundtruths):
        self.predictions.extend(predictions)
        self.ground_truths.extend(groundtruths)

    def get_phrase_accuracies(self):
        phrase_accuracies = {}
        counts = {}
        for prediction, groundtruth in \
                zip(self.predictions, self.ground_truths):
            counts[groundtruth] = counts.get(groundtruth, 0) + 1
            if prediction == groundtruth:
                add = 1
            else:
                add = 0
            phrase_accuracies[groundtruth] = \
                phrase_accuracies.get(groundtruth, 0) + add
        phrase_accuracies = {k: ((v * 100) / counts[k])
                             for k, v in phrase_accuracies.items()}
        phrase_accuracies = {k: v
                             for k, v in
                             sorted(phrase_accuracies.items(),
                                    key=lambda item: item[1])}

        return phrase_accuracies

    def group_labels(self, d):
        # group labels together to save space for plots
        grouped_d = {}
        for k, v in d.items():
            phrases = grouped_d.get(v, [])
            phrases.append(k)
            grouped_d[v] = phrases
        grouped_d_copy = grouped_d.copy()
        for k, v in grouped_d_copy.items():
            phrases_str = ''
            for i, phrase in enumerate(v):
                phrases_str += phrase
                if (i+1) % 15 == 0:
                    phrases_str += '\n'
                elif (i+1) != len(v):
                    phrases_str += ', '
            grouped_d[k] = phrases_str

        return grouped_d

    def plot_phrase_accuracies(self, save_path=None):
        phrase_accuracies = self.get_phrase_accuracies()
        accuracy_phrases = self.group_labels(phrase_accuracies)

        # plot results
        x = list(accuracy_phrases.keys())
        y = range(len(accuracy_phrases.values()))
        assert len(x) == len(y)
        plt.barh(y, x, height=0.2)
        plt.yticks(y, list(accuracy_phrases.values()))
        plt.title('Phrase Accuracies')
        plt.ylabel('Phrases')
        plt.xlabel('Accuracy %')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_phrase_confusion_count(self, save_path=None):
        """How any other different phrases was each phrase confused with

        Use sets to exclude duplicates, count = length set

        Args:
            save_path:

        Returns:

        """
        phrase_confusion_counts = {}
        for pred, gt in zip(self.predictions, self.ground_truths):
            if pred != gt:
                s = phrase_confusion_counts.get(gt, set())
                s.add(pred)
                phrase_confusion_counts[gt] = s

        phrase_confusion_counts = {
            k: len(v) for k, v in phrase_confusion_counts.items()
        }
        phrase_confusion_counts = {
            k: v for k, v in sorted(phrase_confusion_counts.items(),
                                    key=lambda item: item[1])
        }
        confusion_count_phrases = self.group_labels(phrase_confusion_counts)

        # plot results
        x = list(confusion_count_phrases.keys())
        y = range(len(confusion_count_phrases.values()))
        assert len(x) == len(y)
        plt.barh(y, x, height=0.2)
        plt.xticks(x, x)
        plt.yticks(y, list(confusion_count_phrases.values()))
        plt.title('Phrase Confusion Count')
        plt.ylabel('Phrases')
        plt.xlabel('Num unique confused phrases')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_network_graph(self, save_path=None):
        graph = nx.DiGraph()

        phrase_confusion_sets = {}
        labels = {i: gt for i, gt in enumerate(set(self.ground_truths))}
        reversed_labels = {v: k for k, v in labels.items()}
        for pred, gt in zip(self.predictions, self.ground_truths):
            if pred != gt:
                s = phrase_confusion_sets.get(reversed_labels[gt], set())
                s.add(reversed_labels[pred])
                phrase_confusion_sets[reversed_labels[gt]] = s

        unique_labels = set()
        for phrase, confusion_set in phrase_confusion_sets.items():
            graph.add_node(phrase)
            unique_labels.add(phrase)
            for confused_phrase in confusion_set:
                graph.add_edge(phrase, confused_phrase)
                unique_labels.add(confused_phrase)

        labels = {k: v for k, v in labels.items()
                  if k in unique_labels}

        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        ax.set_title('Phrase Confusion Network Graph')
        nx.draw_circular(graph, node_color='y', edge_color='black',
                         node_size=200, labels=labels, arrows=True, ax=ax)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot(self, blocking=True, save_path=None):
        """Plot the confusion matrix graph.

        Uses seaborn, sklearn and pandas libraries

        Returns:
            None
        """
        if not self.predictions:
            return

        plt.figure(figsize=(20, 10))
        assert len(self.predictions) == len(self.ground_truths), \
            f'# Preds != # Ground Truths, ' \
            f'{len(self.predictions)} != {len(self.ground_truths)}'

        y_true, y_pred = self.ground_truths, self.predictions
        labels = list(set(y_true) | set(y_pred))

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

        df = pd.DataFrame(cm, columns=labels, index=labels)
        sn.heatmap(df, annot=True, cmap='YlGnBu', fmt='d', cbar=False)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if blocking:
            plt.show()
        else:
            plt.draw()

    def save(self, path):
        write_pickle_file(self, path)
