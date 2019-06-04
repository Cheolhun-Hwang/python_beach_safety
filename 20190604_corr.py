import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

## Set Params
print("Set Params \n")
file_type = 'avg'
# type = 'drowning'
type = 'drifting'
type_optimal = 'gap'

load_path = './data/20190523/dataset/'
load_drift_file = file_type+'_drifting_normal_3.csv'
load_drown_file = file_type+'_drowning_normal_3.csv'

k_list = ['h',  'dth',  'cd',  'wt',  'wh',  'cs',  'ws',  'tv',  'th']

print("\nLoad Data")

def get_drowning_time(path, file):
    dataset = pd.read_csv(path + file)
    features = dataset[k_list]
    drowning = dataset['drowning']
    k_list.append('drowning')
    data = dataset[k_list]
    return features, drowning,data
def get_drowning_time_accident(path, file):
    data = pd.read_csv(path + file)
    dataset = data[data['drowning'] > 0]
    dataset = dataset[((0.4166 <= dataset.h) & (dataset.h <= 0.84))]
    features = dataset[k_list]
    drowning = dataset['drowning']
    return features, drowning, dataset
def get_drifting_time(path, file):
    dataset = pd.read_csv(path + file)
    features = dataset[k_list]
    drifting = dataset['drifting']
    k_list.append('drifting')
    data = dataset[k_list]
    return features, drifting, data
def get_drifting_time_accident(path, file):
    data = pd.read_csv(path + file)
    dataset = data[data['drifting'] > 0]
    dataset = dataset[((0.4166 <= dataset.h) & (dataset.h <= 0.84))]
    features = dataset[k_list]
    drifting = dataset['drifting']

    return features, drifting, dataset
def showData(feature, label, data):
    print("\nShow Up Database :")
    print("\nLength : " + str(len(feature)))
    print(feature.head())
    print("\nLables : ")
    print(label.head())
    print(data.head())
    print("\n\n")

if(type == 'drifting'):
    af, al, ad = get_drifting_time_accident(load_path, load_drift_file)
else :
    af, al, ad = get_drowning_time_accident(load_path, load_drift_file)

showData(af, al, ad)
f_columns = list(af.columns.values)

corr = af.corr(method = 'pearson')
print(corr)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 15))
sns.heatmap(data=af.corr(), annot=True,
            fmt='.2f', linewidths=.5, cmap='Blues')
plt.show()