"""CMC curve.

This module contains logic for plotting a CMC curve
"""
import matplotlib.pyplot as plt


class CMC:
    """CMC class for creating curves.

    Attributes:
        num_ranks (int): number of ranks to calculate accuracy for
        rank_tallies (list): tally value for each rank
        all_rank_accuracies (list): list of rank tally lists
        title (string): title of the CMC graph
        sub_titles (list): titles for multiple graphs on 1 figure
        labels (list): labels for a graph's legend
        x (list): rank values on the x-axis of the graph
    """

    def __init__(self, num_ranks):
        """CMC constructor.

        Args:
            num_ranks (int): number of ranks to calculate accuracy for
        """
        self.num_ranks = num_ranks
        self.rank_tallies = [0] * num_ranks
        self.all_rank_accuracies = []
        self.title = ''
        self.sub_titles = []
        self.labels = []
        self.x = [i + 1 for i in range(self.num_ranks)]

    def tally(self, predictions, ground_truth):
        """Tally up ranks.

        For each rank, if gt == prediction at that rank, increment the rank

        Args:
            predictions (list): list of predictions
            ground_truth (obj): ground-truth value

        Returns:
            None
        """
        assert len(predictions) == self.num_ranks, \
            f'# Preds != # Ranks, {len(predictions)} != {self.num_ranks}'

        for i in range(self.num_ranks):
            if ground_truth == predictions[i]:
                self.rank_tallies[i] += 1

    def calculate_accuracies(self, num_tests, count_check=True):
        """Calculate rank accuracy.

        For a particular rank i, sum up the rank tallies from 0 - i (inclusive)
        and calculate percentage

        Args:
            num_tests (int): to calculate accuracy percentage with

        Returns:
            None
        """
        if count_check:
            assert sum(self.rank_tallies) == num_tests, \
                f'{sum(self.rank_tallies)} != {num_tests}'

        rank_accuracies = []
        for i in range(self.num_ranks):
            rank_sum = sum([self.rank_tallies[j] for j in range(i + 1)])
            accuracy = rank_sum * 100 / num_tests
            rank_accuracies.append(accuracy)
        self.all_rank_accuracies.append(rank_accuracies)
        self.rank_tallies = [0] * self.num_ranks

    def plot(self):
        """Plot the CMC graph.

        Returns:
            None
        """
        if not self.all_rank_accuracies:
            return

        plt.figure(figsize=(20, 10))

        if len(self.all_rank_accuracies) == 1:
            self.labels.append('')

        for y, label, in zip(self.all_rank_accuracies, self.labels):
            plt.plot(self.x, y, marker='o', label=label)

        plt.ylabel('Recognition Accuracy (%)')
        plt.xlabel('Rank')
        plt.ylim(0, 105)
        plt.xlim(self.x[0], self.x[-1])
        plt.xticks(self.x)
        plt.legend()
        plt.title(self.title)
        plt.tight_layout()
        plt.show()

    def sub_plots(self):
        """Plot a figure with multiple CMC sub-plots.

        Returns:
            None
        """
        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(self.title)
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        for i in range(1, len(self.all_rank_accuracies) + 1):
            plt.subplot(3, 3, i)
            plt.plot(self.x, self.all_rank_accuracies[i - 1])
            plt.ylabel('Recognition Accuracy (%)')
            plt.xlabel('Rank')
            plt.ylim(0, 105)
            plt.xlim(self.x[0], self.x[-1])
            plt.xticks(self.x)
            plt.title(self.sub_titles[i - 1])

        plt.tight_layout()
        plt.show()
