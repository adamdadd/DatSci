#! python
'''Repo: DatSci
Library: Tensorflow
Source:
-----------------------------------------
Author: Adam Dad
Title: Tutorial 1
-----------------------------------------
MIT LICENSE
'''


from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print("Tensorflow Version: ", tf.__version__)
# tf.compat.v1.enable_eager_execution()

eager_state = tf.executing_eagerly()

print("Eager (T/F): ", eager_state)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

data_shape = train_images.shape
print("Data Shape: ", data_shape)

num_train_labels = len(train_labels)
print("number of training labels: ", num_train_labels)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# View firs training image.

train_images = train_images / 255.0
test_images = test_images / 255.0

# Values must be between 0 an 1 for parsing to tf.


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# First 10 training images with respective labels.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
'''
The first layer in this network, tf.keras.layers.Flatten,
transforms the format of the images from a 2d-array (of 28 by 28 pixels),
to a 1d-array of 28 * 28 = 784 pixels.

Think of this layer as unstacking rows of pixels in the image and
lining them up.
This layer has no parameters to learn; it only reformats the data.

After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
These are densely-connected, or fully-connected, neural layers.
The first Dense layer has 128 nodes (or neurons).
The second (and last) layer is a 10-node softmax layer—this returns an array of 10 probability scores that sum to 1.
Each node contains a score that indicates the probability that the current image belongs to one of the 10 classes.
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
'''
Loss function —
This measures how accurate the model is during training.
We want to minimize this function to "steer" the model in the right direction.

Optimizer —
This is how the model is updated based on the data it sees and its loss function.

Metrics —
Used to monitor the training and testing steps.
The following example uses accuracy, the fraction of the images that are correctly classified.
'''

model.fit(train_images, train_labels, epochs=5)

# Training the model

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Accuracy of the model

predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
