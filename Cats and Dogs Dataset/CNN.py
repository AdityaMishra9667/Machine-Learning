import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy as np # To do various array operations
import matplotlib.pyplot as plt #to display the image
import os #to iterate through directories and join paths
import cv2 #opencv to do some image operations

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs1\\{}'.format(NAME))
            print(NAME)

            #A 2x64 convolutional neural network
            model = Sequential() #it's a sequential model

            #chose a random no(64) #Window size for CNN
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1)) #Dense layer with 1 node so that 1 for cat 0 for dog
            model.add(Activation('sigmoid')) #Activation function

            model.compile(loss="binary_crossentropy",
                         optimizer="adam",
                         metrics=['accuracy'])

            model.fit(np.array(X), np.array(y), batch_size=32, epochs=3, validation_split=0.3, callbacks=[tensorboard])  #pass 32 images at a time
                                                                      #Validation_split is the out of sample data
