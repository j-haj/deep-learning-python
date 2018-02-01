from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras import preprocessing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Embedding

max_features = 1000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# Define model
model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_split=0.2)

print(history.history)
print('val accuracy: {}'.format(history.history['val_acc']))
