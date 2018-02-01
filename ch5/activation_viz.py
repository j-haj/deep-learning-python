from tensorflow.python.keras import models
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import os

model = load_model('cats_and_dogs_small_1.h5')
print(model.summary())

img_path = os.path.join(os.getcwd(),
                        'data/train/cats_and_dogs_small/test/cats/cat.1700.jpg')
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()
