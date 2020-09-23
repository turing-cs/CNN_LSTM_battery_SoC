#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:04:37 2019

@author: user
"""
import time
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from keras.datasets import mnist
from keras.datasets import cifar100
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout, Conv1D, MaxPooling1D, SeparableConv2D
from keras.optimizers import Adam
from keras import optimizers,regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization
from sklearn.utils import shuffle
import keras.callbacks
from keras.backend import clear_session
from keras.callbacks import ModelCheckpoint


def mobilenet(train_inputs, train_labels, val_inputs, val_labels, filepath, num_epoch):
    
    num_event = np.size(train_inputs,0)
    num_sensor = np.size(train_inputs,1)
    num_order = np.size(train_inputs,2)
#    # Reshape Inputs
    train_inputs = np.reshape(train_inputs,[num_event,num_sensor,num_order,1])
    val_inputs = np.reshape(val_inputs,[np.size(val_inputs,0),num_sensor,num_order,1])
    
    
    model = Sequential()
    model.add(SeparableConv2D(filters=16, input_shape=(num_sensor,num_order,1), kernel_size=(3,3), strides=(1,1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(SeparableConv2D(filters=16, kernel_size=(3,3), strides=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(SeparableConv2D(filters=16, kernel_size=(3,3), strides=(2,2),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=16, kernel_size=(1,1), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(units=1,activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    train_history = model.fit(train_inputs, train_labels, epochs=num_epoch, callbacks=callbacks_list, 
                              batch_size=100, validation_data=(val_inputs,val_labels))
    clear_session()
    return train_history
