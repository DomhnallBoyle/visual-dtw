import copy
import random

import pytest
from main.utils.cmc import CMC

CLASSES = ['dog', 'cat', 'mouse', 'hen']


def get_random_test_values(seed):
    random.seed(seed)

    list_copy = copy.copy(CLASSES)
    random.shuffle(list_copy)

    return list_copy, random.choice(list_copy)


class TestCMC:

    @pytest.mark.parametrize(
        'predictions, ground_truth, rank_tallies',
        [
            (*get_random_test_values(10), [1, 0, 0, 0]),
            (*get_random_test_values(11), [0, 0, 0, 1])
        ])
    def test_tally(self, predictions, ground_truth, rank_tallies):
        cmc = CMC(num_ranks=len(CLASSES))
        cmc.tally(predictions=predictions, ground_truth=ground_truth)

        assert cmc.rank_tallies == rank_tallies

    def test_calculate_accuracies(self):
        num_tests = 5
        expected_rank_tallies = [0, 0, 4, 1]
        expected_rank_accuracies = [0, 0, 80, 100]

        cmc = CMC(num_ranks=len(CLASSES))

        for i in range(num_tests):
            # seed randomizer & get random predictions and ground-truth
            predictions, ground_truth = get_random_test_values(seed=i)
            cmc.tally(predictions=predictions, ground_truth=ground_truth)

        assert cmc.rank_tallies == expected_rank_tallies
        cmc.calculate_accuracies(num_tests=num_tests)
        assert cmc.all_rank_accuracies[0] == expected_rank_accuracies
