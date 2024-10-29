import random

import matplotlib.pyplot as plt
from main.models import Config
from main.research.test_update_list_3 import get_default_sessions
from main.research.test_update_list_5 import get_test_data, \
    test_data_vs_sessions
from main.services.transcribe import transcribe_signal


def get_accuracy(test_data, training_data):
    ref_signals = [
        (label, template.blob) for label, template in training_data
    ]

    num_correct = 0
    for actual_label, test_template in test_data:
        try:
            predictions = transcribe_signal(
                ref_signals,
                test_template.blob,
                None,
                **Config().__dict__
            )
        except Exception as e:
            continue

        prediction_labels = [prediction['label'] for prediction in predictions]

        if actual_label == prediction_labels[0]:
            num_correct += 1

    accuracy = (num_correct * 100) / len(test_data)

    return accuracy


def experiment_1():
    """Add templates at a time e.g. 1, 5"""
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

    default_accuracy = test_data_vs_sessions(test_data, default_sessions)

    data_so_far = [
        (label, template)
        for session_label, session in default_sessions
        for label, template in session
    ]

    x = []
    accuracies = []
    update_frequency = 5

    print('Default accuracy: ', default_accuracy)

    # now add every single training data point, get accuracy each time
    for i, (training_label, training_template) in enumerate(training_data):
        data_so_far.append((training_label, training_template))

        if (i + 1) % update_frequency == 0:
            accuracy = get_accuracy(test_data, data_so_far)
            print(accuracy)
            accuracies.append(accuracy)
            x.append(i+1)

    # plot results
    plt.plot(x, accuracies, label='Combo')
    plt.axhline(y=default_accuracy, color='black')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Num Transcriptions')
    plt.show()


def experiment_2():
    """Add templates at a time but this time remove defaults so
    len(data) == 13 * 20
    """
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

    default_accuracy = test_data_vs_sessions(test_data, default_sessions)

    # compose default data
    default_data = {}
    for session_label, session in default_sessions:
        for label, template in session:
            phrase_data = default_data.get(label, [])
            phrase_data.append(template)
            default_data[label] = phrase_data

    update_frequency = 5

    added_data = []
    accuracies = []
    x = []

    for i, (training_label, training_template) in enumerate(training_data):
        added_data.append((training_label, training_template))

        if (i + 1) % update_frequency == 0:
            # remove defaults of the same label as no. added
            new_data = default_data.copy()

            # remove random defaults first
            for label, template in added_data:
                phrase_data = new_data[label]
                random_index = random.choice(list(enumerate(phrase_data)))[0]
                del phrase_data[random_index]
                new_data[label] = phrase_data

            # add user templates
            for label, template in added_data:
                new_data[label].append(template)

            # convert dict to list
            training_data = []
            for label, templates in new_data.items():
                for template in templates:
                    training_data.append((label, template))

            assert len(training_data) == 13 * 20

            accuracy = get_accuracy(test_data, training_data)
            print(accuracy)
            accuracies.append(accuracy)
            x.append(i + 1)

    # plot results
    plt.plot(x, accuracies, label='Combo')
    plt.axhline(y=default_accuracy, color='black')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Num Transcriptions')
    plt.show()


if __name__ == '__main__':
    # experiment_1()
    experiment_2()
