import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

load_path = 'D:/data/csv/daecheon_beach/20190502/'
save_path = 'drowning/'

drowning_load = 'normal_1.csv'
drowning_save = 'cluster_drowning.csv'

def getElbow(data):
    Sum_of_squared_distances = []
    K = range(2,11)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        Sum_of_squared_distances.append(km.inertia_)

    plt.plot(K, Sum_of_squared_distances, '-o')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

def hierarchy(dara, type):
    from scipy.cluster.hierarchy import linkage, dendrogram
    mergers = linkage(dara, method='complete')
    dendrogram(mergers, leaf_rotation=90, leaf_font_size=6)
    plt.show()

def silhouette_socore(data, type):
    from sklearn.metrics import silhouette_score
    Sum_of_squared_distances = []
    K = range(2, 16, 1)
    max = 0
    max_score = 0
    for n_clusters in K:
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(data)
        centers = clusterer.cluster_centers_

        score = silhouette_score(data, preds, metric='euclidean')
        Sum_of_squared_distances.append(score)
        print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

        if(score > max_score):
            max_score = score
            max = n_clusters
    print(max)
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
    for num in range (1, 101):
        optimalK = OptimalK(parallel_backend='rust')
        n_clusters = optimalK(np.array(data), cluster_array=np.arange(2, 11))
        array.append(n_clusters)

    set_dic = {}
    for num in array:
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

def get_drowning_time(path, file):
    dataset = pd.read_csv(path + file)

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

def get_drowning_all(path, file):
    dataset = pd.read_csv(path + file)

    wind_dir = dataset['wind_dir']
    current_dir = dataset['current_dir']
    wind_speed = dataset['wind_speed']
    current_speed = dataset['current_speed']
    wave_height = dataset['wave_height']
    water_temp = dataset['water_temp']
    danger_depth = dataset['danger_depth']
    tide_variation = dataset['tide_variation']
    daily = dataset['daily']
    hour = dataset['hour']
    drowning = dataset['drowning']

    temp_dataset = pd.concat([daily, hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation], axis=1)


    return temp_dataset, np.array(drowning)

def test_set_all():
    testset = pd.read_csv(load_path + 'normal.csv')

    wind_dir = testset['wind_dir']
    current_dir = testset['current_dir']
    wind_speed = testset['wind_speed']
    current_speed = testset['current_speed']
    wave_height = testset['wave_height']
    water_temp = testset['water_temp']
    danger_depth = testset['danger_depth']
    tide_variation = testset['tide_variation']
    daily = testset['daily']
    hour = testset['hour']
    drowning = testset['drowning']

    temp_dataset = pd.concat([daily, hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation], axis=1)

    return temp_dataset, drowning

# features_data, classify = get_drowning_all(load_path, drowning_load)
features_data, classify = get_drowning_time(load_path, drowning_load)

# print(features_data.head())
# print(classify)

# optimal_k = gap_statistic(features_data, False)
# optimal_k = silhouette_socore(features_data, False)
optimal_k = 10

# print("optimal : " + str(optimal_k))

# optimal_k_all = [8, 9, 10]
# optimal_k_time = [6, 7, 8, 9]

# test_feature, test_classify = test_set_all()

# for optimal_k in optimal_k_all :
#     kmeans_model_1 = KMeans(n_clusters=optimal_k)
#     distances_1 = kmeans_model_1.fit(features_data)
#     # labels_1 = distances_1.labels_
#     # features_data['cluster'] = labels_1
#
#     pred = kmeans_model_1.predict(test_feature)
#
#     test_feature['prediction'] = 1 if pred > 0 else 0
#
#     crs = pd.crosstab(test_classify, test_feature['prediction'])
#     tp = crs[0][0]
#     tn = crs[0][1]
#     fp = crs[1][0]
#     fn = crs[1][1]
#     print(crs)
#     TPR = tp / (tp + fn)
#     TNR = tn / (tn + fp)
#     ACC = (tp + tn) / (tp + tn + fp + fn)
#
#     print("\n\n Confusion Matrix " + str(optimal_k))
#     print("TPR : " + str(TPR))
#     print("TNR : " + str(TNR))
#     print("ACC : " + str(ACC))
#
#     test_feature['drowning'] = test_classify
#
#     test_feature.to_csv(load_path+save_path+"All_"+str(optimal_k)+"_"+drowning_save)


kmeans_model_1 = KMeans(n_clusters=optimal_k)
distances_1 = kmeans_model_1.fit(features_data)
labels_1 = distances_1.labels_
features_data['cluster'] = labels_1
print(kmeans_model_1.cluster_centers_)

features_data.to_csv(load_path+save_path+"time_"+str(optimal_k)+"_"+drowning_save)