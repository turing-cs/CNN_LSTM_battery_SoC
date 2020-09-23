# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:40:05 2019

@author: Ting-Han
"""
import numpy as np

from scipy import io
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D, Conv1D
from keras.layers import LSTM, TimeDistributed, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend import clear_session

Monitor_Method = 'CheckPoint'

def LeNet_Regressor(train_inputs, train_labels, val_inputs, val_labels, 
          filepath, num_epoch):
    
    # # Find Parameters
    # num_event = np.size(train_inputs,0)
    # num_order = np.size(train_inputs,1)
    # print(num_event)
    # print(num_order)
    
    train_inputs = np.expand_dims(train_inputs, axis = 2)
    val_inputs = np.expand_dims(val_inputs, axis = 2)

                     
    model = Sequential()
    	
    # LFLB1
    model.add(Conv1D(filters = 6, kernel_size = 3, strides=2, activation='relu', input_shape=(5, 1)))
    
    # #LSTM
    model.add(LSTM(units=300, activation="relu"))
    model.add(Dense(units = 80, activation="relu"))
    #FC	
    model.add(Dense(units=1,activation='linear'))
    
    #Model compilation	
    model.compile(optimizer='adam', loss='MAE', metrics=['mae','mape','acc'])
    print(model.summary())
    savebestmodel = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    earlystop = EarlyStopping(monitor='val_loss', patience=1000, verbose=1, mode='auto')
    train_history = model.fit(train_inputs, train_labels,  
                              epochs=num_epoch, callbacks=[earlystop, savebestmodel], 
                              batch_size = 10000, validation_data=(val_inputs,val_labels))
#                              
#                              
#    clear_session()
#    train_history = 0
    return train_history