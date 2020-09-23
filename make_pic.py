# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:29:02 2020

@author: SHAOJIM
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from matplotlib.ticker import NullFormatter 
import matplotlib.ticker as ticker
from os import walk
#%%
def caculate_mean_absolute_error(predict, real):
    mae = mean_absolute_error(predict, real)
    mae = '%.4f' % (mae)
    return mae

#%%
def read_csv(filename):
    print(filename)
    df = pd.read_csv(filename)
    predict = df['predict'].tolist()
    real = df['real'].tolist()
    return predict, real

def find_csvfile(result_filename):
    first_battery = []
    second_battery = []
    for root, dirs, files in walk('result/' + result_filename):
        print(files)
            
    for i in range(len(files)):
        if files[i].split('.')[-1] == 'csv' and len(files[i].split('_')) > 1 and files[i].split('_')[1] == 'dynamic':
            first_battery.append(files[i])
            
        elif files[i].split('.')[-1] == 'csv' and len(files[i].split('_')) > 1 and files[i].split('_')[1] == 'sec':
            second_battery.append(files[i])
    
    return first_battery, second_battery
        
#%%
def make_pic(battery, pic_name, result_filename):
    path = "./result/" + result_filename + "/"
    for i in range(len(battery)):
    # A_predict_B_3_transfer
        if battery[i].split('_')[-2] == '5degC' and battery[i].split('_')[-1] == 'dynamic1.xlsx.csv':
            predict1, real1 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '5degC' and battery[i].split('_')[-1] == 'dynamic2.xlsx.csv':
            predict2, real2 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '5degC' and battery[i].split('_')[-1] == 'dynamic3.xlsx.csv':
            predict3, real3 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '25degC' and battery[i].split('_')[-1] == 'dynamic1.xlsx.csv':
            predict4, real4 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '25degC' and battery[i].split('_')[-1] == 'dynamic2.xlsx.csv':
            predict5, real5 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '25degC' and battery[i].split('_')[-1] == 'dynamic3.xlsx.csv':
            predict6, real6 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '40degC' and battery[i].split('_')[-1] == 'dynamic1.xlsx.csv':
            predict7, real7 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '40degC' and battery[i].split('_')[-1] == 'dynamic2.xlsx.csv':
            predict8, real8 = read_csv(path + battery[i])
        elif battery[i].split('_')[-2] == '40degC' and battery[i].split('_')[-1] == 'dynamic3.xlsx.csv':
            predict9, real9 = read_csv(path + battery[i])

    mae1 = caculate_mean_absolute_error(predict1, real1)
    mae2 = caculate_mean_absolute_error(predict2, real2)
    mae3 = caculate_mean_absolute_error(predict3, real3)
    mae4 = caculate_mean_absolute_error(predict4, real4)
    mae5 = caculate_mean_absolute_error(predict5, real5)
    mae6 = caculate_mean_absolute_error(predict6, real6)
    mae7 = caculate_mean_absolute_error(predict7, real7)
    mae8 = caculate_mean_absolute_error(predict8, real8)
    mae9 = caculate_mean_absolute_error(predict9, real9)
    

    #%%
    # whole figure
    fig= plt.figure(constrained_layout=True, figsize=(12,8))
    #plt.xticks(fontsize=20)
    spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    bbox = dict(boxstyle="round", fc="0.8")
    tick_spacing = 5
    #plt.title("Test",fontweight="bold")
    #plt.rcParams["title.labelweight"] = "bold"
    #%%
    f_ax1 = fig.add_subplot(spec[0, 0])
    #nlist = range(len(predict1))
    nlist = np.linspace(0, len(predict1)/100, num=len(predict1))
    f_ax1.plot(nlist, predict1, label='predict')
    f_ax1.plot(nlist, real1, label='real')
    f_ax1.set(ylabel='SOC(%)')
    f_ax1.set_xlim(0, max(nlist))
    f_ax1.set_xticks(np.round(np.linspace(0, len(predict1)/100, num=7)))
    f_ax1.set_ylim(0, 100)
    #f_ax1.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #f_ax1.set_title("D1 5℃ MAE = " + str(mae1), fontweight="bold")
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f_ax1.set_yticklabels([0,20,40,60,80,100])
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax1.grid()
    f_ax1.annotate("Dynamic 1\nTmp. = 5℃\nMAE = " + str(mae1), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax1.legend()
    
    #%%
    f_ax2 = fig.add_subplot(spec[0, 1])
    nlist = np.linspace(0, len(predict2)/100, num=len(predict2))
    f_ax2.plot(nlist, predict2, label='predict')
    f_ax2.plot(nlist, real2, label='real')
    f_ax2.set_xlim(0, max(nlist)) 
    f_ax2.set_xticks(np.round(np.linspace(0, len(predict2)/100, num=7)))
    f_ax2.set_ylim(0, 100)
    f_ax2.set_title(pic_name, fontweight="bold")
    #f_ax2.grid()
    #f_ax2.get_xaxes().set_ticks([])
    f_ax2.axes.get_yaxis().set_major_formatter(NullFormatter())
    #f_ax2.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax2.get_yaxis().set_visible(False) 
    #f_ax2.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax2.grid()
    f_ax2.annotate("Dynamic 2\nTmp. = 5℃\nMAE = " + str(mae2), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax2.legend()
    #f2_ax2.set_title('gs[0,1]')
    
    #%%
    f_ax3 = fig.add_subplot(spec[0, 2])
    nlist = np.linspace(0, len(predict3)/100, num=len(predict3))
    f_ax3.plot(nlist, predict3, label='predict')
    f_ax3.plot(nlist, real3, label='real')
    f_ax3.set_xlim(0, max(nlist)) 
    f_ax3.set_xticks(np.round(np.linspace(0, len(predict3)/100, num=7)))
    f_ax3.set_ylim(0, 100)
    #f_ax3.set_title("D3 5℃ MAE = " + str(mae3), fontweight="bold")
    f_ax3.axes.get_yaxis().set_major_formatter(NullFormatter())
    #f_ax3.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax3.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax3.grid()
    f_ax3.annotate("Dynamic 3\nTmp. = 5℃\nMAE = " + str(mae3), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax3.legend()
    #f2_ax3.set_title('gs[0,2]')
    
    #%%
    f_ax4 = fig.add_subplot(spec[1, 0])
    nlist = np.linspace(0, len(predict4)/100, num=len(predict4))
    f_ax4.plot(nlist, predict4, label='predict')
    f_ax4.plot(nlist, real4, label='real')
    f_ax4.set(ylabel='SOC(%)')
    f_ax4.set_xlim(0, max(nlist)) 
    f_ax4.set_xticks(np.round(np.linspace(0, len(predict4)/100, num=7)))
    f_ax4.set_ylim(0, 100)
    #f_ax4.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax4.set_title("D1 25℃ MAE = " + str(mae4), fontweight="bold")
    #f_ax4.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax4.grid()
    f_ax4.annotate("Dynamic 1\nTmp. = 20℃\nMAE = " + str(mae4), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax4.legend()
    #f2_ax4.set_title('gs[1,0]')
    
    #%%
    f_ax5 = fig.add_subplot(spec[1, 1])
    nlist = np.linspace(0, len(predict5)/100, num=len(predict5))
    f_ax5.plot(nlist, predict5, label='predict')
    f_ax5.plot(nlist, real5, label='real')
    f_ax5.set_xlim(0, max(nlist)) 
    f_ax5.set_xticks(np.round(np.linspace(0, len(predict5)/100, num=7)))
    f_ax5.set_ylim(0, 100)
    #f_ax5.set_title("D2 25℃ MAE = " + str(mae5), fontweight="bold")
    f_ax5.axes.get_yaxis().set_major_formatter(NullFormatter())
    #f_ax5.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax5.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax5.grid()
    f_ax5.annotate("Dynamic 2\nTmp. = 20℃\nMAE = " + str(mae5), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax5.legend()
    #f2_ax5.set_title('gs[1,1]')
    
    #%%
    f_ax6 = fig.add_subplot(spec[1, 2])
    nlist = np.linspace(0, len(predict6)/100, num=len(predict6))
    f_ax6.plot(nlist, predict6, label='predict')
    f_ax6.plot(nlist, real6, label='real')
    f_ax6.set_xlim(0, max(nlist)) 
    f_ax6.set_xticks(np.round(np.linspace(0, len(predict6)/100, num=7)))
    f_ax6.set_ylim(0, 100)
    #f_ax6.set_title("D3 25℃ MAE = " + str(mae6), fontweight="bold")
    f_ax6.axes.get_yaxis().set_major_formatter(NullFormatter())
    #f_ax6.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax6.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax6.grid()
    f_ax6.annotate("Dynamic 3\nTmp. = 20℃\nMAE = " + str(mae6), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax6.legend()
    #f2_ax6.set_title('gs[1,2]')
    
    #%%
    f_ax7 = fig.add_subplot(spec[2, 0])
    nlist = np.linspace(0, len(predict7)/100, num=len(predict7))
    f_ax7.plot(nlist, predict7, label='predict')
    f_ax7.plot(nlist, real7, label='real')
    f_ax7.set(ylabel='SOC(%)')
    f_ax7.set_xlim(0, max(nlist)) 
    f_ax7.set_xticks(np.round(np.linspace(0, len(predict7)/100, num=7)))
    f_ax7.set_ylim(0, 100)
    #f_ax7.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax7.set_title("D1 40℃ MAE = " + str(mae7), fontweight="bold")
    #f_ax7.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax7.grid()
    f_ax7.annotate("Dynamic 1\nTmp. = 40℃\nMAE = " + str(mae7), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax7.legend()
    #f2_ax7.set_title('gs[2,0]')
    
    #%%
    f_ax8 = fig.add_subplot(spec[2, 1])
    nlist = np.linspace(0, len(predict8)/100, num=len(predict8))
    f_ax8.plot(nlist, predict8, label='predict')
    f_ax8.plot(nlist, real8, label='real')
    f_ax8.set_xlim(0, max(nlist)) 
    f_ax8.set_xticks(np.round(np.linspace(0, len(predict8)/100, num=7)))
    f_ax8.set_ylim(0, 100)
    #f_ax8.set_title("D2 40℃ MAE = " + str(mae8), fontweight="bold")
    f_ax8.axes.get_yaxis().set_major_formatter(NullFormatter())
    #f_ax8.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax8.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax8.grid()
    f_ax8.annotate("Dynamic 2\nTmp. = 40℃\nMAE = " + str(mae8), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax8.legend()
    #f2_ax8.set_title('gs[2,1]')
    
    #%%
    f_ax9 = fig.add_subplot(spec[2, 2])
    nlist = np.linspace(0, len(predict9)/100, num=len(predict9))
    f_ax9.plot(nlist, predict9, label='predict')
    f_ax9.plot(nlist, real9, label='real')
    f_ax9.set_xlim(0, max(nlist)) 
    f_ax9.set_xticks(np.round(np.linspace(0, len(predict9)/100, num=7)))
    f_ax9.set_ylim(0, 100)
    #f_ax9.set_title("D3 40℃ MAE = " + str(mae9), fontweight="bold")
    f_ax9.axes.get_yaxis().set_major_formatter(NullFormatter())
    #f_ax9.axes.get_xaxis().set_major_formatter(NullFormatter())
    #f_ax9.set_yticklabels([0,20,40,60,80,100])
    #plt.setp(f2_ax1.xaxis.get_majorticklabels(), fontsize=5)
    #f2_ax1.set_xticklabels(labels, fontsize=5)
    #f2_ax1.set_yticklabels(labels, fontsize=5)
    #f2_ax1.set_xlabel(xlabel='time step', fontsize=10)
    f_ax9.grid()
    f_ax9.annotate("Dynamic 3\nTmp. = 40℃\nMAE = " + str(mae9), xy=(8,10), xycoords='axes points', bbox=bbox)
    f_ax9.legend()
    #f2_ax9.set_title('gs[2,2]')
    
    #%%
    fig.savefig(path + pic_name + 'png', dpi=500)
