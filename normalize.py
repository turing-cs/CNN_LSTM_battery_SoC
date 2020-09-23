# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:29:18 2019

@author: SHAOJIM
"""

from sklearn import preprocessing
import numpy as np

def normalize_min_max(data, max_num, min_num):
    z_voltage = data
    z_voltage_diff = []
    z_voltage = np.array(z_voltage, dtype=float)
    z_voltage = z_voltage.reshape(-1, 1)
    z_voltage_minmax = np.array([[max_num], [min_num]])
    z_voltage_min_max_scaler = preprocessing.MinMaxScaler()
    z_voltage_train_min_max = z_voltage_min_max_scaler.fit_transform(z_voltage_minmax)
    z_voltage = z_voltage_min_max_scaler.transform(z_voltage)
    z_voltage = z_voltage.flatten()
    highest_z_voltage = max(z_voltage)
    #print(len(voltage))
    for i in range(len(z_voltage)):
        z_voltage_diff.append(highest_z_voltage - z_voltage[i])
        
    return z_voltage
