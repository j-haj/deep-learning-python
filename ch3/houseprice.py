from tensorflow.python.keras.datasets import boston_housing
from tensorflow.python.keras import models
from tensorflow.python.keras import layers

import matplotlib.pyplot as plt
import numpy as np

def load_data():
    '''Loads and normalizes the boston housing data'''
    (train_data, train_targets), (test_data, test_targets) =\
            boston_housing.load_data()

    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    return (train_data, train_targets), (test_data, test_targets)

def build_model(train_data_cols):
    '''Build a regression model

    Args:
        train_data_cols: the second argument of the training data shape
    Returns:
        keras model for regression
    '''

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=train_data_cols))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def main():
    '''MAIN'''

    (train_data, train_targets), (test_data, test_targets) = load_data()

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 200
    all_mae_histories = []
    all_scores = []

    for i in range(k):
        print('processing fold #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) *\
                num_val_samples]

        partial_train_data = np.concatenate(
                [train_data[:i * num_val_samples],
                    train_data[(i + 1) * num_val_samples:]],
                axis=0)

        partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples],
                    train_targets[(i + 1) * num_val_samples:]],
                axis=0)

        model = build_model((train_data.shape[1],))
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=1, verbose=0)
        mae_history = history.history['val_mean_absolute_error']
        all_mae_histories.append(mae_history)

    plot_mae(all_mae_histories)

def plot_mae(histories):
    '''Plots the MAE histories

    Args:
        histories: mae histories collected during training
    '''
    epochs = range(1, len(histories) + 1)
    plt.plot(epochs, histories)
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE')
    plt.legend()
    plt.savefig('housing_mae.pdf', format='pdf')
    plt.clf()

if __name__ == '__main__':
    main()
