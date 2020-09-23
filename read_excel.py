#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:18:05 2019

@author: user
"""

import pandas as pd
import numpy as np
from normalize import normalize_min_max

def read_excel(filename):

#    file_path = '/home/user/Desktop/DATA1/u103011142/soc_predict_v2'
#    filename = 'constant_loading/5degC/0.1C.xlsx'
    
    df = pd.read_csv(filename, index_col=0)
    try:
        voltage = df['Voltage'].tolist()
    except:
        voltage = df['Vpack'].tolist()
        
    current = df['Current'].tolist()
    temperature = df['Temperature'].tolist()
    soc = df['SOC'].tolist()
    
    avg = 20
    
    z_voltage = normalize_min_max(voltage, 17400, 12000)
    z_current = normalize_min_max(current, 0, -1500)
    z_temperature = normalize_min_max(temperature, 45, 0)
    
    z_voltage_avg = []
    z_current_avg = []
    
    for i in range(len(z_voltage)-avg+1):
        z_voltage_avg.append(sum(z_voltage[i:i+avg]))
    for i in range(len(z_current)-avg+1):
        z_current_avg.append(sum(z_current[i:i+avg]))
    
    z_voltage = z_voltage[avg:]
    z_current = z_current[avg:]
    z_temperature = z_temperature[avg:]
    
    x_train = []
    y_train = []
    
    if float(soc[10]) > 50:
        soc_tmp = 1
    else:
        soc_tmp = 100
        
    soc = soc[avg:]
        
    for i in range(19, len(soc)):
        x_train.append([z_voltage[i], z_current[i], z_temperature[i], np.mean(z_voltage[i-19:i+1]), np.mean(z_current[i-19:i+1])])
#        x_train.append([z_battery_1[i], z_battery_2[i], z_battery_3[i], z_battery_4[i], z_current[i], z_temperature[i]])
        y_train.append(float(soc[i])*soc_tmp)
    
        
    x_newtrain = np.array(x_train,dtype=float)
    y_newtrain = np.array(y_train,dtype=float)
    return x_newtrain, y_newtrain