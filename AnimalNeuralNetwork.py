#COLLAB: https://colab.research.google.com/drive/1BayozelZeZozqawPwZLoX-nndznZnC67
#NOTE: not configured to run with the GPU on non-google-collab runtime environments
#The Animal Classification Neural Network

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

#note, x is the input and y is the output

#load the data into a pandas frame
allDataFrame = pd.read_csv("https://raw.githubusercontent.com/IgnisOwl/AIClubStuff/master/AnimalClassification.csv")


#we want 90 of the rows in the data set to be dedicated to training
trainingData = 90

trainX = allDataFrame[0:trainingData] #this is the training input, from 0-100 is basically what its saying
#Remove the "class_type" column from it, because this is what the neural network is trying to figure out, or the output(Y)
trainX = trainX.drop(columns=["class_type"])
#we also have to remove the animal name because that cannot be used in neural network's computation(the name does not define any animal characteristics)
trainX = trainX.drop(columns=["animal_name"])
#correct Y values:
trainY = allDataFrame[0:trainingData][["class_type"]]

#do the same for the test data, but for the remaining values:
testX = allDataFrame[trainingData:len(allDataFrame)] #basically use values 90 - 100 for testing
testX = testX.drop(columns=["class_type"])
testX = testX.drop(columns=["animal_name"])

#intialize the sequential object as the variable model
model = keras.models.Sequential() #type of network

columnAmount = trainX.shape[1] #get the amount of columns in the training Input

model.add(keras.layers.Dense(16, activation="relu", input_shape=(columnAmount,))) #input layer, has 17 inputs, aka the amount of columns minus the output column(s), input shape specifies the number of rows and columns, first we set the amount of columns, then after we put nothing because we can have any amount of rows
model.add(keras.layers.Dense(32, activation="relu")) #hidden layer(32 neurons)
model.add(keras.layers.Dense(1)) #the output is the class_type that it guesses

#now we must compile the model:
model.compile(optimizer="adam", loss="mean_squared_error")
#optimizer controls learning rate, adjusts it throughout training
#loss is the function that tells us the error amount, or the loss

#now we gotta actually train the model

model.fit(trainX, trainY, epochs=300) #trainX is the training inputs, trainY is the training outputs, remember, epochs is the amount of times it tries all of the data in the csv and backpropigates


#Now let's test:


testY = model.predict(testX)
#this starts at index 92 on the actual github sheet if ur looking, because of the list thing where it starts at zero and the fact that it has all the column names at top
print("TEST Y:")
print(testY)
print("ROUNDED TEST Y:")
print(testY.round()) #print predicted outputs(rounded)

