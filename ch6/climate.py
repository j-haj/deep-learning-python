import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GRU
from tensorflow.python.keras.optimizers import RMSprop


data_dir = os.path.join(os.getcwd(), 'data/jena_climate')
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

with open(fname, 'r') as f:
    data = f.read()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

# Constants
lookback = 1440
step = 6
delay = 144
batch_size = 128

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

train_gen = generator(float_data,
                       lookback=lookback,
                       delay=delay,
                       min_index=0,
                       max_index=200000,
                       shuffle=True,
                       step=step,
                       batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        if step % 1000 == 0:
            print('step: ', step)
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(lookback // step, float_data.shape[-1])))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    return model

def plot_data():
    temp = float_data[:, 1]
    plt.plot(range(len(temp)), temp)
    plt.title('Plot of Temperatures')
    plt.savefig('temps.pdf', format='pdf')
    plt.clf()

def plot_hist(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss.pdf', format='pdf')
    plt.clf()

def main():
    print('Naive evluation')
    #evaluate_naive_method()
    print('Building model')
    model = build_model()
    print('Fitting model')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)
    print('Creating fig')
    plot_hist(history)


if __name__ == '__main__':
    main()