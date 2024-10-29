import argparse
import ast
import os
import pprint
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from main.research.key_word_spotting import kws
from main.research.key_word_spotting_lrs3 import extract_user_kws_paths_lrs3_liopa

RANDOM_STATE = 2021
MIN_NUM_EXAMPLES_PER_KEY_WORD = 5

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# TODO:
#  Try out LSTM approach again - use pooling for timesteps - DONE (SHOWS OVERFITTING)
#  Get more data e.g. 220 keywords * num users
#  Convert True/False labels to 1/0


def get_data(key_word, search_video_paths, ref_video_paths):
    all_data = []
    for search_video_path, ref_video_path in zip(search_video_paths, ref_video_paths):
        search_phrase, search_video_path = search_video_path
        ref_phrase, ref_video_path = ref_video_path

        results = kws(ref_video_path, search_video_path, search_phrase, key_word, preprocess=True,
                      core_search_template=1.0, use_pad_finding=False, pose_direction='centre', debug=False)
        if not results:
            continue

        distances = results['Tried Paddings'][0]['Distances']
        all_data.append([
            key_word, search_phrase, search_video_path, ref_phrase, ref_video_path,
            results['FA Likelihood'], distances, 1 if key_word in ref_phrase else 0
        ])

    return all_data


def create_dataset(args):
    pprint.pprint(args.__dict__)

    # user-ids to key-words to paths
    user_kw_paths = extract_user_kws_paths_lrs3_liopa(**args.__dict__)

    # find users that say the same phrase
    key_words_to_users = {}
    for user_id, key_word_paths in user_kw_paths.items():
        for key_word in key_word_paths.keys():
            users = key_words_to_users.get(key_word, [])
            users.append(user_id)
            key_words_to_users[key_word] = users

    key_words = [key_word for key_word, users in key_words_to_users.items()
                 if len(users) >= MIN_NUM_EXAMPLES_PER_KEY_WORD]
    print('Num key-words:', len(key_words))

    columns = ['Key Word', 'Search Phrase', 'Search Video Path', 'Ref Phrase', 'Ref Video Path', 'FA Likelihood',
               'Distances', 'In Phrase']

    if os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
    else:
        df = pd.DataFrame(columns=columns)

    if len(df) > 0:
        last_row = df.iloc[[-1]]
        last_search_key_word = last_row['Key Word'].values[0]
        key_words = key_words[key_words.index(last_search_key_word):]

    counter = 0
    for key_word in tqdm(key_words):
        user_ids = key_words_to_users[key_word]
        other_key_words = list(set(key_words) - {key_word})

        # each user is a search user vs all other users of every key-word
        for user_id in user_ids:
            other_user_ids = list(set(user_ids) - {user_id})
            user_id_paths = user_kw_paths[user_id][key_word]
            positive_ref_video_paths = []
            for other_user_id in other_user_ids:
                positive_ref_video_paths.extend(user_kw_paths[other_user_id][key_word])

            for video_path in user_id_paths:
                counter += len(positive_ref_video_paths) * 2  # hope for same # of +ve and -ve examples

                # positive examples
                positive_samples = get_data(key_word, [video_path] * len(positive_ref_video_paths),
                                            positive_ref_video_paths)

                # negative examples
                negative_ref_video_paths = []
                while len(negative_ref_video_paths) != len(positive_ref_video_paths):
                    random_user_id = random.choice(other_user_ids)
                    random_key_word = random.choice(other_key_words)
                    ref_video_path = random.choice(user_kw_paths[random_user_id][random_key_word])
                    ref_phrase = ref_video_path[0]
                    if key_word not in ref_phrase:
                        negative_ref_video_paths.append(ref_video_path)

                negative_samples = get_data(key_word, [video_path] * len(negative_ref_video_paths),
                                            negative_ref_video_paths)

                if positive_samples and negative_samples:
                    new_rows = pd.DataFrame(data=positive_samples + negative_samples, columns=columns)
                    df = pd.concat([df, new_rows])
                    df.to_csv(args.csv_path, index=False)

    print('Max examples:', counter)


def pool(l, max_length, f='mean'):
    funcs = {
        'mean': lambda x, y: (x + y) / 2,
        'max': lambda x, y: max(x, y),
        'min': lambda x, y: min(x, y)
    }

    if len(l) == max_length:
        return l

    if len(l) > max_length:
        while True:
            l_new = []
            for i in range(len(l)-1):
                x1, x2 = l[i], l[i+1]
                v = funcs[f](x1, x2)
                l_new.append(v)
            l = l_new
            if len(l) == max_length:
                return l

    if len(l) < max_length:
        return []


def analysis(args):
    df = pd.read_csv(args.csv_path)
    df['Distances'] = df.apply(lambda row: ast.literal_eval(row['Distances']), axis=1)

    # reset random seed
    import time
    t = 1000 * time.time()  # current time in milliseconds
    random.seed(int(t) % 2**32)
    np.random.seed(int(t) % 2**32)

    print(df['In Phrase'].value_counts())

    # show frequency of distance lengths
    data = []
    for index, row in df.iterrows():
        data.append(len(row['Distances']))
    plt.hist(data)
    plt.show()

    # extract 10 random samples
    # show diff between normal 0 distances and pooled distances
    # which is better? average, max or min pooling?
    num_samples = 10
    random_samples = df.sample(n=num_samples, random_state=random.randint(1, 50))
    num_rows, num_columns = 2, 5
    fig, axs = plt.subplots(num_rows, num_columns)
    row_ind, column_ind = 0, 0
    for index, row in random_samples.iterrows():
        x = [i for i in range(len(row['Distances']))]
        y = row['Distances']
        axs[row_ind, column_ind].plot(x, y)

        y2 = pool(y, max_length=40)
        if y2:
            x2 = [i for i in range(len(y2))]
            axs[row_ind, column_ind].plot(x2, y2)

        in_phrase = True if row['In Phrase'] == 1 else False
        axs[row_ind, column_ind].set_title(f'In Phrase: {in_phrase}')

        axs[row_ind, column_ind].set_ylim(0, 2)

        if column_ind == num_columns - 1:
            row_ind += 1
            column_ind = 0
        else:
            column_ind += 1
    plt.show()


def train_lstm(args):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from datetime import datetime
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    from tensorflow.keras.layers import Conv1D, TimeDistributed, Dense, MaxPooling1D, Flatten, LSTM, Dropout, Masking
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    from main.utils.pre_process import pre_process_signal

    df = pd.read_csv(args.csv_path)
    df['Distances'] = df.apply(lambda row: ast.literal_eval(row['Distances']), axis=1)
    print(df)

    num_timesteps = args.num_timesteps
    train, test = train_test_split(df, test_size=0.2, stratify=df['In Phrase'])
    print('Training balance:', train['In Phrase'].value_counts())
    print('Test balance:', test['In Phrase'].value_counts())

    scaler = MinMaxScaler()

    def transform(_set, scaler_fit=False):
        # x, y = [], []
        # for index, row in _set.iterrows():
        #     pooled_distances = pool(row['Distances'], max_length=num_timesteps)
        #     if not pooled_distances:
        #         continue
        #     x.append(pooled_distances)
        #     y.append(row['In Phrase'])
        # x, y = np.asarray(x), np.asarray(y)

        x, y = [], []
        for index, row in _set.iterrows():
            if len(row['Distances']) > num_timesteps:
                continue

            min_distance = min(row['Distances'])
            sample = np.array([[d] for d in row['Distances']])

            sample = pre_process_signal(sample, 100, 0, 1)

            x.append(sample)
            y.append(row['In Phrase'])
        x, y = np.asarray(x), np.asarray(y)

        # pad timesteps with 0 to make them all the same length
        x = pad_sequences(x, maxlen=num_timesteps, dtype='float64', padding='post')

        # # scale between 0 and 1 range
        # if scaler_fit:
        #     x = scaler.fit_transform(x)
        # else:
        #     x = scaler.transform(x)

        # scale between 0 and 1 range
        if scaler_fit:
            x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        else:
            x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2])

        return x, y

    x_train, y_train = transform(train, scaler_fit=True)
    x_test, y_test = transform(test)

    num_features = x_train.shape[-1]

    # # time distributed transforms
    # num_steps = 4
    # num_timesteps = num_timesteps // num_steps
    # x_train = x_train.reshape(x_train.shape[0], num_steps, num_timesteps, num_features)
    # x_test = x_test.reshape(x_test.shape[0], num_steps, num_timesteps, num_features)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    # TODO:
    #  Bidirectional
    #  CNN + LSTM
    #  better features

    # lstm
    model = Sequential([
        # tf.keras.layers.Embedding(input_dim=1, output_dim=embedding_vector_length, mask_zero=True,
        #                           input_length=(num_timesteps, num_features)),

        Masking(input_shape=(num_timesteps, num_features)),

        # tf.keras.layers.LSTM(1000, input_shape=(num_timesteps, num_features), return_sequences=True),
        # tf.keras.layers.LSTM(500, input_shape=(num_timesteps, num_features), return_sequences=True),
        # tf.keras.layers.LSTM(100, input_shape=(num_timesteps, num_features)),

        # tf.keras.layers.LSTM(1, input_shape=(num_timesteps, num_features), kernel_initializer='he_uniform'), # this works well but slow to train on own

        # tf.keras.layers.LSTM(10, input_shape=(num_timesteps, num_features), kernel_initializer='he_uniform'),

        # tf.keras.layers.LSTM(50, input_shape=(num_timesteps, num_features), kernel_initializer='he_uniform', return_sequences=True),
        LSTM(10, input_shape=(num_timesteps, num_features), kernel_initializer='he_uniform', return_sequences=True),
        LSTM(1, kernel_initializer='he_uniform'),

        # tf.keras.layers.LSTM(100, input_shape=(num_timesteps, num_features), kernel_initializer='he_uniform', return_sequences=True),
        # tf.keras.layers.LSTM(50, kernel_initializer='he_uniform', return_sequences=True),
        # tf.keras.layers.LSTM(10, kernel_initializer='he_uniform'),

        # tf.keras.layers.Dropout(rate=0.3),

        # tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'),
        Dense(50, activation='relu', kernel_initializer='he_uniform'),
        Dense(10, activation='relu', kernel_initializer='he_uniform'),
        Dense(1, activation='sigmoid')

        # TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu'),
        #                 input_shape=(None, num_timesteps, num_features)),
        # TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu')),
        # TimeDistributed(Dropout(0.3)),
        # TimeDistributed(MaxPooling1D()),
        # TimeDistributed(Flatten()),

        # LSTM(10),

        # Dropout(0.3),

        # Dense(10, activation='relu'),
        # Dense(1, activation='sigmoid')
    ])
    print(model.summary())

    current_datetime = datetime.now().strftime(
        f'%Y-%m-%d-%H-%M-%S->{args.num_timesteps}-{args.num_epochs}-{args.batch_size}-{args.learning_rate}')

    # tensorboard callback
    log_dir = 'tensorboard_logs/' + current_datetime
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # checkpoint callback
    # save every X epochs
    checkpoint_path = 'tensorboard_checkpoints/' + current_datetime + '/cp-{epoch:04d}.ckpt'
    num_iterations_per_epoch = x_train.shape[0] // args.batch_size
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
                                          save_freq=50*num_iterations_per_epoch)

    # lr scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', mode='min')

    # training data shuffled before each epoch
    optimiser = Adam(learning_rate=args.learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=args.num_epochs, batch_size=args.batch_size, validation_data=(x_test, y_test),
              callbacks=[
                  tensorboard_callback,
                  checkpoint_callback,
                  # lr_scheduler
              ])


def main(args):
    f = {
        'create_dataset': create_dataset,
        'train_lstm': train_lstm,
        'analysis': analysis
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parser.add_parser('create_dataset')
    parser_1.add_argument('dataset_path')
    parser_1.add_argument('phrases_path')
    parser_1.add_argument('csv_path')

    parser_2 = sub_parser.add_parser('train_lstm')
    parser_2.add_argument('csv_path')
    parser_2.add_argument('--num_timesteps', type=int, default=50)
    parser_2.add_argument('--num_epochs', type=int, default=1000)
    parser_2.add_argument('--learning_rate', type=float, default=0.001)
    parser_2.add_argument('--batch_size', type=int, default=32)

    parser_3 = sub_parser.add_parser('analysis')
    parser_3.add_argument('csv_path')

    main(parser.parse_args())
