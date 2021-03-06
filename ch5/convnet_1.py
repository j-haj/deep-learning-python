from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.datasets import mnist

def load_data():
    '''Returns a tuple of tuples:
        (train iamges, train labels), (test images, test labels)
    '''
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return (train_images, train_labels), (test_images, test_labels)

def build_model():
    '''Builds a Keras model'''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def main():
    '''MAIN'''
    (train_images, train_labels), (test_images, test_labels) = load_data()
    model = build_model()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy: {:.4f}'.format(test_acc))
if __name__ == '__main__':
    main()
