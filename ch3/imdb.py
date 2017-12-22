from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import layers
from tensorflow.python.keras import metrics
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

def import_data():
    '''Imports the imdb dataset and returns train and test tensors using one-hot
    encoding.
    '''
    (train_data, train_labels), (test_data, test_labels) =\
        imdb.load_data(num_words=10000)

    x_train = vectorize_sequence(train_data)
    x_test = vectorize_sequence(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    return (x_train, y_train), (x_test, y_test)

def vectorize_sequence(sequences, dimension=10000):
    '''Vectorizes the sequence to use one-hot encoding

    Args:
        sequences: data to encode
        dimension: dimension of the output

    Returns:
        integer tensor of shape (samples, dimension)
    '''
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def build_model(n_layers=1, layer_size=16, input_shape=(10000,), activation='relu'):
    '''Builds a model with the given configuration

    Args:
        n_layers: the number of layers in the model, excluding the output layer
        input_shape: shape of the input data
        activation: type of activation function to be used in the hidden layers

    Returns:
        a keras model with the given configuration
    '''
    assert n_layers > 0

    model = models.Sequential()
    model.add(layers.Dense(layer_size, activation=activation, input_shape=input_shape))

    for i in range(n_layers-1):
        model.add(layers.Dense(layer_size, activation=activation))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def plot_loss_results(history):
    '''Plots the validation and training loss over each epoch

    Params:
        history: dictionary output from training
    '''
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('loss_results.pdf', format='pdf')
    plt.clf()

def plot_acc_results(history):
    '''Plots training and validation accuracy data at each epoch

    Args:
        history: history object from keras fit
    '''
    history_dict = history.history
    print('history_dict.keys(): {}'.format(history_dict.keys()))
    acc_values = history_dict['binary_accuracy']
    val_acc_values = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc_values) + 1)

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('acc_results.pdf', format='pdf')
    plt.clf()

def main():
    '''MAIN'''
    # load data
    (x_train, y_train), (x_test, y_test) = import_data()

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    # build model
    model = build_model(n_layers=5, layer_size=32)
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=[metrics.binary_accuracy])

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    # create plots
    plot_loss_results(history)
    plot_acc_results(history)


if __name__ == '__main__':
    main()
