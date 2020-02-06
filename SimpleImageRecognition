#image recognition without convolution

# Import modules
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Import dataset, these are the built in datasets in keras
# data = keras.datasets.fashion_mnist # Clothing
data = keras.datasets.mnist # Hand-written digits

# Load data, normally we would need to organize this using something like pandas, if it was our own data, but keras does it for us cuz its the buit in data
(x_train, y_train), (x_test, y_test) = data.load_data()

# num_train_data = len(y_train)
# num_test_data = len(y_test)
# print("Total data:", num_train_data+num_test_data)

print(x_train[0])
#show our images
plt.imshow(x_train[0], cmap=plt.cm.binary)
print(y_train[0])

# Normalize the data: map range
# convert 0-255 range -> 0-1 range, to make it easier to work with
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)


# Build model
model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# one-hot encoding - converts 2d array to 1d array
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
# each number is a confidence level
# 1 = 100% confidence, 0 = 0% confidence

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Train (fit) the model
model.fit(x_train, y_train, epochs=8)

# Predict
predictions = model.predict(x_test)

#function that makes it easier to see our predicted vs actual
def show_prediction(i):
  plt.imshow(x_test[i], cmap=plt.cm.binary)
  print("Output array:", predictions[i])
  print("Prediction: ", end="")
  print(np.argmax(predictions[i]))
  print("Actual:", y_test[i])

show_prediction(0)

