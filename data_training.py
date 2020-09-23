import numpy as np
import os
import matplotlib.pyplot as plt
from read_excel import read_excel
from LeNet_LSTM import LeNet_Regressor
from os import walk
import pandas as pd

def training(result_filename, training_data_filename, num_epochs, column):
    path = "./result/" + result_filename
    if not os.path.isdir(path):
        os.mkdir(path)
    
    newdirs = []
    for root, dirs, files in walk(training_data_filename):
        if files is not None:
            for i in range(len(files)):
                newdirs.append(root + '/' + files[i])
                print(root + '/' + files[i])
    
    x_train, y_train = read_excel(training_data_filename + '/' + files[0])
    
    
    for i in range(1, len(newdirs)):
        a1, b1 = read_excel(newdirs[i])
    
        x_train = np.vstack((x_train, a1))
    
        y_train = np.hstack((y_train, b1))
        print(newdirs[i])
    
    x_tem_train = []
    y_tem_train = []
    x_tem_validation = []
    y_tem_validation = []
    #x_newtest = np.array(x_test,dtype=float)
    #y_newtest = np.array(y_test,dtype=float)
    for i in range(len(x_train)):
        if i%5 == 0:
            x_tem_validation.append(x_train[i])
            y_tem_validation.append(y_train[i])
        else:
            x_tem_train.append(x_train[i])
            y_tem_train.append(y_train[i])
        
    x_new_train = np.array(x_tem_train,dtype=float)
    y_new_train = np.array(y_tem_train,dtype=float)
    x_new_validation = np.array(x_tem_validation,dtype=float)
    y_new_validation = np.array(y_tem_validation,dtype=float)
    #LeNet_Regressor(train_inputs, train_labels, val_inputs, val_labels, 
    #          filepath, num_epoch, SensorType):
    
    seqModel = LeNet_Regressor(x_new_train, y_new_train, x_new_validation, y_new_validation, path + '/best_weight.hdf5', num_epochs)
    
    hist_df = pd.DataFrame(seqModel.history) 
    # or save to csv: 
    hist_csv_file = path + '/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
#    train_loss = seqModel.history['loss']
#    val_loss   = seqModel.history['val_loss']
#    train_acc  = seqModel.history['acc']
#    val_acc    = seqModel.history['val_acc']
#    xc         = range(len(train_loss))
#    
#    plt.figure()
#    plt.plot(xc, train_loss, label='train_loss')
#    plt.plot(xc, val_loss, label='val_loss')
#    plt.savefig(path + '/history.png' )
