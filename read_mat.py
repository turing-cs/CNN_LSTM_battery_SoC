

from scipy.io import loadmat
import pandas as pd

from os import walk

mypath = "./battery_mat"
for root, dirs, files in walk(mypath):
    print(files)
  
for i in range(len(files)):
    filename = files[i]
    print('start file ........' + filename)
    annots = loadmat('./battery_mat/'+ filename)
    
    columnlist = ['TimeStamp', 'Voltage', 'Current', 'Ah', 'Wh', 'Power', 'Battery_Temp_degC', 'Time', 'Chamber_Temp_degC']
    df_empty = pd.DataFrame(columns=columnlist)
    
    
    for i in range(len(annots['meas']['Voltage'].item())):
    #for i in range(1, 100):
        data = [[row.flat[i] for row in line] for line in annots['meas'][0]]
        df_train = pd.DataFrame(data, columns=columnlist)
        df_empty = pd.concat([df_empty, df_train])
        
    df_empty.to_csv('./battery/' + filename + '.csv')
    print('complete file ........' + filename)

'''   
#data2 = [[row.flat[1] for row in line] for line in annots['meas'][0]]
#aa = annots['meas'].tolist()
#print(len(annots['meas']['Voltage'].item()))
    columns = ['TimeStamp', 'Voltage', 'Current', 'Ah', 'Wh', 'Power', 'Battery_Temp_degC', 'Time', 'Chamber_Temp_degC']
    df_train = pd.DataFrame(data, columns=columns)
#df_train2 = pd.DataFrame(data2, columns=columns)
    df_train.append(df_train, ignore_index=True, sort=False)
#print(df_train)

data = [[row.flat[0] for row in line] for line in annots['meas'][0]]
df_train = pd.DataFrame(data, columns=columnlist)
aa = pd.concat([df_train, df_train])
df_empty.append(df_train)
df_empty.append(df_train)
df_empty.append(df_train)
'''
