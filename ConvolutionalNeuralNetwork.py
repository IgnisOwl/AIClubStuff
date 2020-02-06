  import tensorflow as tf
from tensorflow import keras
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D #flatten the data, dense layers(for fully connected), conv2d is convolutional layers, maxpooling is the max value pooling method
from keras.models import Sequential

WEIGHTS_FILE = "weights.h5" #saves the weights so we don't have to retrain it, you can also save the entire model, not just the weights, so you don't have to run the build_model() function again

#use the built in keras handwritten digits set
data = keras.datasets.mnist

#Keras's built in function that handles the data, obviously if you are using your own dataset you will have to manage the data yourself, look at the first neural network we did where we used pandas to do this
(x_train, y_train), (x_test, y_test) = data.load_data()

#Normalize, aka convert 0-255 to 0-1, https://stackoverflow.com/questions/47435526/what-is-the-meaning-of-axis-1-in-keras-argmax
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

#reshape the data, we need to pass data with dimension of width, height, and channel:
x_train_reshaped = x_train.reshape(len(x_train), 28, 28, 1)
x_test_reshaped = x_test.reshape(len(x_test), 28, 28, 1)


def build_model():
  model = Sequential()

  #build up our layers:


  """THIS IS OUR BASIC NEURAL NETWORK, we aren't using this though, because we are doing convolution.
  model.add(Flatten())
  model.add(Dense(69, activation="relu"))
  model.add(Dense(420, activation="relu"))
  model.add(Dense(69, activation="relu"))
  model.add(Dense(10, activation="softmax"))"""


  #HyperParameter: Features that you can change about your network to improve efficiency, such as a neural network size
  #We have to add the convolution layers before the Fully Connected layers.
  model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(28, 28, 1))) #kernel_size is the size of the kernel/filter, the last value in input shape is the channel, because it's grayscale, we only have 1 channel
  model.add(MaxPooling2D())
  model.add(Conv2D(64, kernel_size=3, activation="relu", ))
  model.add(MaxPooling2D())
  #now to the fully connected layers:
  model.add(Flatten()) #remember, out flatten converts our multi-dimensional data set into a 1-dimensional list using one hot encoding.
  model.add(Dense(69, activation="relu"))
  model.add(Dense(420, activation="relu"))
  model.add(Dense(69, activation="relu"))
  model.add(Dense(10, activation="softmax")) #softmax confidence values are percentages that all add up to one.

  #optimizer will automatically adjust some hyperparameters such as training speed
  #metrics is just what information will be printed while it's training
  model.compile(
      optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"]
  )

  #return our model that we made:
  return(model)
  
 
def train():
  #Now that we have our model, we have to train it

  model = build_model() #call our function

  model.fit(x_train_reshaped, y_train, epochs=5) #remember, epochs is how many times we will run through out model

  #test it once its trained
  model.evaluate(x_test_reshaped, y_test) #this will simply test our trained model with the x/y test data that we have, so we can evaluate how well it works

  model.save_weights(WEIGHTS_FILE) #now we should save the weights so next time we train it, so if we wanna use our network without re-trianing it, just using our previously trained data, we can do that

def test():
  #now if we wan't to evaluate it later without training it, but using our old trained weightes, we can just do it like so:
  model = build_model()

  #load in the weights file, so we DO NOT run model.fit, because we do not need to train it.
  model.load_weights(WEIGHTS_FILE)
  #evaluation:
  loss, accuracy = model.evaluate(x_test_reshaped, y_test)

  print("Loss:",loss)
  print("Accuracy:",accuracy)
  
build_model()
train()
