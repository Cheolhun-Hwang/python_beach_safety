import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

accid_type = 'drowning'

load_path = 'D:/workspace/python/SaftyBeach/data/20190523/drowning/feature_all/2019-05-24-11-28/'
# load_drift_file = file_type+'_drifting_normal.csv'
# load_drown_file = file_type+'_drowning_normal.csv'

load_file = 'avg_drowning_10_gap_26_ps.csv'

def get_drowning_time(path, file):
    dataset = pd.read_csv(path + file)


    hour = dataset['hour']
    prob = dataset['p('+accid_type+'|pred)*(p(dist_s)*p(pred|s))']
    classify = dataset[accid_type]

    temp_dataset = pd.concat([hour, prob, classify], axis=1)

    return temp_dataset, dataset
def get_drifting_time(path, file):
    dataset = pd.read_csv(path + file)

    hour = dataset['hour']
    prob = dataset['p('+accid_type+'|pred)*(p(dist_s)*p(pred|s))']
    classify = dataset[accid_type]
    temp_dataset = pd.concat([hour, prob, classify], axis=1)

    return temp_dataset, dataset

data, origin = get_drifting_time(load_path,load_file)
# data, origin = get_drowning_time(load_path+load_drown_file)

plt.xlim(0, 1)
plt.ylim(0, 0.0004)
plt.scatter(data.hour,
            data['p('+accid_type+'|pred)*(p(dist_s)*p(pred|s))'],
            c=data[accid_type])
plt.xlabel('hour')
plt.ylabel('B prob result')
plt.colorbar()
plt.show()


# groups = data.groupby(accid_type)
# fig, ax = plt.subplots()
# for name, group in groups:
#     ax.plot(groups.hour,
#             group['p('+accid_type+'|pred)*(p(dist_s)*p(pred|s))'],
#             marker='o',
#             linestyle='',
#             label=name)
# ax.legend(fontsize=12, loc='upper left') # legend position
# plt.xlabel('hour', fontsize=14)
# plt.ylabel('B result', fontsize=14)
# plt.show()
