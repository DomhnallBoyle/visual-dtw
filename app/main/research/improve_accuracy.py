"""
Using the users transcriptions data to help improve their future transcriptions

2 DTW/KNN - 1 for default templates, 1 for users transcriptions

"""
import copy
import random

import matplotlib.pyplot as plt
from main.models import Config
from main.research.cmc import CMC
from main.research.test_update_list_5 import get_test_data, \
    get_default_sessions
from main.services.transcribe import transcribe_signal


def make_predictions(test_signal, ref_signals):
    predictions = transcribe_signal(
        ref_signals,
        test_signal,
        None,
        **Config().__dict__
    )

    return predictions


def get_default_predictions(test_template, default_sessions):
    ref_signals = [(label, template.blob)
                   for session_label, session in default_sessions
                   for label, template in session]
    assert len(ref_signals) == 13 * 20

    return make_predictions(test_template.blob, ref_signals)


def get_user_predictions(test_template, transcribed_templates):
    ref_signals = [(label, template.blob)
                   for label, template in transcribed_templates]

    return make_predictions(test_template.blob, ref_signals)


def combine_predictions(_user_predictions, _default_predictions, weight):
    for prediction in _user_predictions:
        prediction['accuracy'] *= weight

    _user_predictions = {
        pred['label']: pred['accuracy'] for pred in _user_predictions
    }
    _default_predictions = {
        pred['label']: pred['accuracy'] for pred in _default_predictions
    }

    final_predictions = []
    for label, user_accuracy in _user_predictions.items():
        default_accuracy = _default_predictions.get(label)
        if not default_accuracy:
            final_predictions.append((label, user_accuracy))
        else:
            final_predictions.append((label, user_accuracy + default_accuracy))
            del _default_predictions[label]

    # add the default ones left out
    for label, default_accuracy in _default_predictions.items():
        final_predictions.append((label, default_accuracy))

    # sort and select top 3
    final_predictions = sorted(final_predictions, key=lambda x: x[1],
                               reverse=True)[:3]

    # normalise to 1
    accuracy_sum = sum([pred[1] for pred in final_predictions])
    final_predictions = [(label, float(accuracy) / accuracy_sum)
                         for label, accuracy in final_predictions]
    # print(final_predictions)

    # return labels only
    return [pred[0] for pred in final_predictions]


def main():
    # grab default sessions
    default_sessions = get_default_sessions()

    # grab training and test data for the user
    train_test_data = get_test_data(user_id='12')
    random.shuffle(train_test_data)
    train_test_split = int(len(train_test_data) * 0.6)
    training_data = train_test_data[:train_test_split]
    test_data = train_test_data[train_test_split:]
    print('Train: ', len(training_data))
    print('Test: ', len(test_data))

    transcribed_templates = []
    combo_accuracies, default_accuracies = [], []
    for label, template in training_data:
        # add our "transcribed" template
        transcribed_templates.append((label, template))

        # cmc_default = CMC(num_ranks=3)
        # cmc_combination = CMC(num_ranks=3)

        default_num_correct, combo_num_correct = 0, 0

        for actual_label, test_template in test_data:
            try:
                copy_1 = copy.copy(test_template)
                copy_2 = copy.copy(test_template)

                default_predictions = get_default_predictions(
                    copy_1, default_sessions.copy())

                # tally cmc for defaults only
                default_labels = [prediction['label']
                                  for prediction in default_predictions]
                # cmc_default.tally(default_labels, actual_label)
                if actual_label == default_labels[0]:
                    default_num_correct += 1

                user_predictions = get_user_predictions(
                    copy_2, transcribed_templates.copy())

                # combine predictions, weight user predictions only
                # weight = len(transcribed_templates) / len(training_data)
                # weight *= weight
                weight = 0.8
                combined_labels = combine_predictions(user_predictions.copy(),
                                                      default_predictions.copy(),
                                                      weight)
                # cmc_combination.tally(combined_labels, actual_label)
                if actual_label == combined_labels[0]:
                    combo_num_correct += 1

            except Exception as e:
                continue

        # cmc_default.calculate_accuracies(num_tests=len(test_data),
        #                                  count_check=False)
        # cmc_combination.calculate_accuracies(num_tests=len(test_data),
        #                                      count_check=False)

        default_accuracy = (default_num_correct * 100) / len(test_data)
        combo_accuracy = (combo_num_correct * 100) / len(test_data)

        combo_accuracies.append(combo_accuracy)
        default_accuracies.append(default_accuracy)

        print('Transcribed templates: ', len(transcribed_templates))
        print('Default: ', default_accuracy)
        print('Combo: ', combo_accuracy)
        # print(cmc_default.all_rank_accuracies[0])
        # print(cmc_combination.all_rank_accuracies[0])
        print()

    x = [i + 1 for i in range(len(training_data))]
    plt.plot(x, combo_accuracies, label='Combo')
    plt.plot(x, default_accuracies, label='Default')
    plt.legend()
    plt.xlabel('Num Transcriptions')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    main()
