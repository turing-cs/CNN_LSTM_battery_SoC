# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:30:40 2020

@author: SHAOJIM
"""

from data_training import training
from data_predict import predict
from make_pic import find_csvfile
from make_pic import make_pic
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

result_filename ="20200922_1_LeNet_LSTM_simpo_column80"
column = 80

#%%
# traing
training_data_filename  = './panasonic_v9_original/train_panasonic_v9'
num_epochs = 200000

training(result_filename, training_data_filename, num_epochs, column)


#%%

testing_data_filename = './test_data'
model_name = "LeNetLSTM"
weight = "best_weight.hdf5"

# testing 
#predict(result_filename, testing_data_filename, model_name, weight, column)

#first_battery, second_battery = find_csvfile(result_filename)
#make_pic(first_battery, model_name + '_first_battery', result_filename)
#make_pic(second_battery, model_name + '_second_battery', result_filename)
