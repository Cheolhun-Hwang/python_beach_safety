import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

load_path = 'C:/Users/qewqs/hooney/Data/csv/daecheon_beach/20190509/'
load_drift_file = 'drifting/normal_drifting.csv'
load_drown_file = 'drowning/normal_drowning.csv'

save_drift_path = 'drifting/save/'
save_drown_path = 'drowning/save/'

def get_drowning_time(path, file):
    data = pd.read_csv(path + file)

    dataset = data[data['drowning'] > 0]

    print("data : \n")
    print(dataset.head())

    wind_dir = dataset['wind_dir']
    current_dir = dataset['current_dir']
    wind_speed = dataset['wind_speed']
    current_speed = dataset['current_speed']
    wave_height = dataset['wave_height']
    water_temp = dataset['water_temp']
    danger_depth = dataset['danger_depth']
    tide_variation = dataset['tide_variation']
    hour = dataset['hour']
    drowning = dataset['drowning']

    temp_dataset = pd.concat([hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation], axis=1)

    return temp_dataset, np.array(drowning)

def get_drifting_time(path, file):
    data = pd.read_csv(path + file)

    dataset = data[data['drifting'] > 0]
    print("data : \n")
    print(dataset.head())

    wind_dir = dataset['wind_dir']
    current_dir = dataset['current_dir']
    wind_speed = dataset['wind_speed']
    current_speed = dataset['current_speed']
    wave_height = dataset['wave_height']
    water_temp = dataset['water_temp']
    danger_depth = dataset['danger_depth']
    tide_variation = dataset['tide_variation']
    hour = dataset['hour']
    drifting = dataset['drifting']

    temp_dataset = pd.concat([hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation], axis=1)


    return temp_dataset, np.array(drifting)

def get_iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    data_df = pd.DataFrame(data['data'], columns=data['feature_names'])
    return data_df, data['target']

def silhouette_socore(data, type):
    from sklearn.metrics import silhouette_score
    Sum_of_squared_distances = []
    K = range(2, 16, 1)
    max = 0
    max_score = 0
    for n_clusters in K:
        clusterer = KMeans(n_clusters=n_clusters, n_init=123)
        preds = clusterer.fit_predict(data)
        centers = clusterer.cluster_centers_

        score = silhouette_score(data, preds, metric='euclidean')
        Sum_of_squared_distances.append(score)
        # print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

        if(score > max_score):
            max_score = score
            max = n_clusters
    # print(max)
    if(type == True):
        plt.plot(K, Sum_of_squared_distances, '-o')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    return max

def gap_statistic(data, type):
    from gap_statistic import OptimalK

    array = []
    for num in range (1, 10):
        optimalK = OptimalK(parallel_backend='rust')
        n_clusters = optimalK(np.array(data), cluster_array=np.arange(2, 11))
        array.append(n_clusters)

    set_dic = {}
    for num in array:
        if num in set_dic:
            set_dic[num] += 1
        else:
            set_dic[num] = 1

    # print(set_dic)

    best_key = 0
    best_value = 0

    for key, value in set_dic.items():
        if(value > best_value):
            best_value = value
            best_key = key

    if(type == True):
        plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
        plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
                    optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
        plt.grid(True)
        plt.xlabel('Cluster Count')
        plt.ylabel('Gap Value')
        plt.title('Gap Values by Cluster Count')
        plt.show()

    return best_key

def bestK(list):
    set_dic = {}
    for num in list:
        if num in set_dic:
             set_dic[num] += 1
        else:
             set_dic[num] = 1

    print(set_dic)

    best_key = 0
    best_value = 0

    for key, value in set_dic.items():
        if(value > best_value):
            best_value = value
            best_key = key

    return best_key, best_value

def findK(list, type, len):
    optiK=[]
    for num in range(0, len, 1):
        print("Loop : " + str(num));
        if(type == False):
            gap = gap_statistic(list, False)
            optiK.append(gap)
        else:
            sil = silhouette_socore(list, False)
            optiK.append(sil)
    print(bestK(optiK))

# drifting_features, drifting_classify = get_drifting_time(load_path, load_drift_file)
# findK(drifting_features, False, 1000)
# findK(drifting_features, True, 1000)

# drowning_features, drowning_classify = get_drowning_time(load_path, load_drown_file)
# findK(drowning_features, False, 1000)
# findK(drowning_features, True, 1000)

#Temp
f, c  = get_iris_data()

print(f.head())
print(c)

# findK(f, False, 1000)
findK(f, True, 1000)