from tensorflow.python.keras.datasets import reuters
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np


def load_data(num_words=10000):
    '''Loads the reuters dataset and returns the vectorized training and test data
    '''
    (train_data, train_labels), (test_data, test_labels) =\
            reuters.load_data(num_words=num_words)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    return (x_train, train_labels), (x_test, test_labels)

def vectorize_sequences(sequences, dimension=10000):
    '''Converts sequence data into vectors
    
    Args:
        sequences: the sequence data to be converted
        dimension: dimension of the sequences

    Returns:
        tensor of size (len(sequences), dimension)
    '''
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=64):
    '''Converts the data to one-hot encoding

    Args:
        labels: labels being encoded
        dimension: dimension of the labels
    
    Returns:
        one-hot encoding of the given labels in the given dimension
    '''
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def build_model():
    '''Constructs the model
    
    Returns:
        Keras model
    '''
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    return model


def plot_loss(history):
    '''Creates a plot of training and validation loss from the given training
    history object.

    Args:
        history: history object from model.fit()
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('reuters_loss.pdf', format='pdf')
    plt.clf()

def plot_accuracy(history):
    '''Create a plot of the training and validation accuracy from the given
    training history object.

    Args:
        history: history object from model.fit()
    '''
    print('history.history.keys() = {}'.format(history.history.keys()))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('reuters_acc.pdf', format='pdf')
    plt.clf()

def main():
    '''MAIN'''

    # setup data
    (train_data, train_labels), (test_data, test_labels) = load_data()
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    # build model
    model = build_model()

    # train the model
    x_val = train_data[:1000]
    partial_x_train = train_data[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=20,
                        batch_size=512,
                        validation_data=(x_val, y_val))

    plot_loss(history)
    plot_accuracy(history)

if __name__ == '__main__':
    main()

