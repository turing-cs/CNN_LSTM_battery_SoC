# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:39:11 2019

@author: SHAOJIM
"""
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt # ?¯è???æ¨¡??
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
import os
from read_excel import read_excel
from two_to_three_d import convert_data
from os import walk


def predict(result_filename, testing_data_filename, model_name, weight, column):
    
    path = "./result/" + result_filename

    newdirs = []
    for root, dirs, files in walk(testing_data_filename):
        if files is not None:
            for i in range(len(files)):
                newdirs.append(root + '/' + files[i])
                print(root + '/' + files[i])

    model = load_model(path + '/' + weight)
    print(model.summary())
    
    
    for i in range(len(newdirs)):
        name_tmp = newdirs[i].split('/')
        print(name_tmp)
        name = name_tmp[2] + '_' + name_tmp[3] + '_' + name_tmp[4] 
        x_newtest, y_newtest = read_excel(newdirs[i])
        train_inputs, d1 = convert_data(x_newtest, y_newtest, column)
        num_event = np.size(train_inputs,0)
        num_order = np.size(train_inputs,1)
        num_sensor = np.size(train_inputs,2)
        train_inputs = np.reshape(train_inputs,[num_event, 1, num_order, num_sensor,1])
        Y_pred = model.predict(train_inputs)
        MAE = mean_absolute_error(d1, Y_pred)
        MAE = '%.4f' % (MAE)
        print(MAE)
        
        # plot a picture
        output = []
        for m in range(len(Y_pred)): 
            output.append(Y_pred[m][0])
    #        d1[m] = d1[m]/100
        nlist = np.arange(0,len(Y_pred))
        fig, ax = plt.subplots()
        ax.plot(nlist, output, label='predict')
        ax.plot(nlist, d1, label='true')
        ax.set(xlabel='time step', ylabel='SOC(%)', title= name + "\nmean absolute error =" + str(MAE))
        ax.grid()
        ax.legend()
        
        fig.savefig(path + "/" + model_name + '_' + name + ".png")
        #plt.show()
        
        temp = []
        temp.append(['predict', 'real'])
        
        for m in range(len(output)):
            temp.append([output[m], d1[m]])
        
        with open(path + "/" + model_name + '_' + name + ".csv", 'w', newline='') as csvfile:
    
            writer = csv.writer(csvfile)
            writer.writerows(temp)

