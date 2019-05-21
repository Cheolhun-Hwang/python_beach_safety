import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#############################################################################################################################################

## Set Params
print("Set Params \n")
load_path = 'E:/data/csv/daecheon_beach/20190509/'
load_drift_file = 'drifting/normal_drifting.csv'
load_drown_file = 'drowning/normal_drowning.csv'

save_drift_path = 'drifting/save/'
save_drown_path = 'drowning/save/'

features_length = 9
optimal_k = 10

type = 'drifting' # 'drowning'

#############################################################################################################################################

## Load Data
print("Load Data \n")

def get_drowning_time(path, file):
    dataset = pd.read_csv(path + file)
    dataset = dataset[(0.4166 <= dataset.hour) & (dataset.hour <= 0.84)]

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

    return temp_dataset, drowning, dataset
def get_drowning_time_accident(path, file):
    data = pd.read_csv(path + file)
    dataset = data[data['drifting'] > 0]
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
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

    return temp_dataset, drowning, dataset
def get_drifting_time(path, file):
    dataset = pd.read_csv(path + file)
    dataset = dataset[(0.4166 <= dataset.hour) & (dataset.hour <= 0.84)]

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

    return temp_dataset, drifting, dataset
def get_drifting_time_accident(path, file):
    data = pd.read_csv(path + file)

    dataset = data[data['drifting'] > 0]
    dataset = dataset[(0.4166 <= dataset.hour) & (dataset.hour <= 0.84)]


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

    return temp_dataset, drifting, dataset

def showData(feature, label, data):
    print("Show Up Database : \n")
    print(feature.head())
    print(label.head())
    print(data.head())
    print("\n\n")

if(type == 'drifting'):
    print("type : drifting")
    ac, lb, ad = get_drifting_time_accident(load_path, load_drift_file)
else:
    print("type : drowning")
    ac, lb, ad = get_drowning_time_accident(load_path, load_drift_file)

## Data K-means
print("Data K-means \n")
def dataKmenas(featrues):
    kmeans_model_1 = KMeans(n_clusters=optimal_k, n_init=123)
    distances_1 = kmeans_model_1.fit(featrues)
    labels_1 = distances_1.labels_
    print(kmeans_model_1.cluster_centers_)

    return labels_1, kmeans_model_1.cluster_centers_

labels, centers = dataKmenas(ac)

temp_data = ac
temp_data['cluster'] = labels+1
print("Temp_data Length : " + str(len(temp_data)))

print("Centroid : \n")
print("size : " + str(len(centers)))
print(centers)

print("Cluster Result : \n")
print(temp_data.head())


## Data Split Data
print("Data Split Data \n")
clusters = []
for num in range(1, len(centers)+1, 1):
    dim = temp_data[temp_data['cluster'] == num]
    clusters.append(dim)

print("clusters : \n")
for index in range(0, len(clusters), 1):
    print("\nclusters " + str(index+1) )
    print(clusters[index].head())
    print("Length : " + str(len(clusters[index])))

#############################################################################################################################################

## Summarize Clusters
print("Summarize \n")

def calc(list):
    min = np.min(list)
    max = np.max(list)
    mean = np.mean(list)
    std = np.std(list)
    return min, max, mean, std, len(list)

clusters_summarize = []

for index in range(0, len(clusters), 1):
    print("\nclusters " + str(index+1) )
    temp_cluster = clusters[index]
    data_columns = list(temp_cluster.columns.values)

    feature_summarize = []
    for feature in data_columns[0:len(data_columns)-1]:
        print("Feature : " + feature)
        c_min, c_max, c_mean, c_std, c_size = calc(np.array(temp_cluster[feature]))
        cluster_dic = {'min':c_min, 'max':c_max, 'mean':c_mean, 'std':c_std, 'size':c_size}
        print("Min : " + str(cluster_dic['min']))
        print("Max : " + str(cluster_dic['max']))
        print("Mean : " + str(cluster_dic['mean']))
        print("Std : " + str(cluster_dic['std']))
        print("Size : " + str(cluster_dic['size']))
        feature_summarize.append(cluster_dic)
    clusters_summarize.append(feature_summarize)
len("Cluster Summarize : " + str(len(clusters_summarize)))

with open(load_path+'cluster_summarize.txt', 'w') as f:
    for item in clusters_summarize:
        f.write("%s\n" % item)

#############################################################################################################################################

## Test Data set
print("Test Dataset Load... \n")

if(type == 'drifting'):
    print("type : drifting")
    f, c, d = get_drifting_time(load_path, load_drift_file)

else:
    print("type : drowning")
    f, c, d = get_drowning_time(load_path, load_drift_file)


## P(S) 구하기!
print("P(mobS) \n")
print("\nAdding Prediction Column : ")
d['prediction'] = 0

for index in range(0, len(clusters_summarize), 1):
    feature_summarize = clusters_summarize[index]
    print("cluster : " + str((index+1)))
    print(str(feature_summarize[0]['min']) + " <= " + str(d.hour) + " <= " + str(
        feature_summarize[0]['max']) + " = " + str(
        ((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))))
    print(str(feature_summarize[1]['min']) + " <= " + str(d.wind_dir) + " <= " + str(
        feature_summarize[1]['max']) + " = " + str(
        ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max']))))
    print(str(feature_summarize[2]['min']) + " <= " + str(d.current_dir) + " <= " + str(
        feature_summarize[2]['max']) + " = " + str(
        ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))))
    print(str(feature_summarize[3]['min']) + " <= " + str(d.wind_speed) + " <= " + str(
        feature_summarize[3]['max']) + " = " + str(
        ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max']))))
    print(str(feature_summarize[4]['min']) + " <= " + str(d.current_speed) + " <= " + str(
        feature_summarize[4]['max']) + " = " + str(
        ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))))
    print(str(feature_summarize[5]['min']) + " <= " + str(d.wave_height) + " <= " + str(
        feature_summarize[5]['max']) + " = " + str(
        ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max']))))
    print(str(feature_summarize[6]['min']) + " <= " + str(d.water_temp) + " <= " + str(
        feature_summarize[6]['max']) + " = " + str(
        ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max']))))
    print(str(feature_summarize[7]['min']) + " <= " + str(d.danger_depth) + " <= " + str(
        feature_summarize[7]['max']) + " = " + str(
        ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max']))))
    print(str(feature_summarize[8]['min']) + " <= " + str(d.tide_variation) + " <= " + str(
        feature_summarize[8]['max']) + " = " + str(
        ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max']))))
    print(str((((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max'])) &
          ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max'])) &
          ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max'])) &
          ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max'])) &
          ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max'])) &
          ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max'])) &
          ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max'])) &
          ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max'])) &
          ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max'])))))



    d.loc[(((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max'])) &
          ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max'])) &
          ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max'])) &
          ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max'])) &
          ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max'])) &
          ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max'])) &
          ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max'])) &
          ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max'])) &
          ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max'])))
    , 'prediction'] = 1

d.to_csv(load_path+"time_drifting_prediction.csv")
print("\nAdding Cluster Column : ")
d['cluster'] = 0

for index in range(0, len(clusters_summarize), 1):
    feature_summarize = clusters_summarize[index]
    print("cluster : " + str((index+1)))
    print(str(feature_summarize[0]['min']) + " <= " + str(d.hour) + " <= " + str(
        feature_summarize[0]['max']) + " = " + str(
        ((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))))
    print(str(feature_summarize[1]['min']) + " <= " + str(d.wind_dir) + " <= " + str(
        feature_summarize[1]['max']) + " = " + str(
        ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max']))))
    print(str(feature_summarize[2]['min']) + " <= " + str(d.current_dir) + " <= " + str(
        feature_summarize[2]['max']) + " = " + str(
        ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))))
    print(str(feature_summarize[3]['min']) + " <= " + str(d.wind_speed) + " <= " + str(
        feature_summarize[3]['max']) + " = " + str(
        ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max']))))
    print(str(feature_summarize[4]['min']) + " <= " + str(d.current_speed) + " <= " + str(
        feature_summarize[4]['max']) + " = " + str(
        ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))))
    print(str(feature_summarize[5]['min']) + " <= " + str(d.wave_height) + " <= " + str(
        feature_summarize[5]['max']) + " = " + str(
        ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max']))))
    print(str(feature_summarize[6]['min']) + " <= " + str(d.water_temp) + " <= " + str(
        feature_summarize[6]['max']) + " = " + str(
        ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max']))))
    print(str(feature_summarize[7]['min']) + " <= " + str(d.danger_depth) + " <= " + str(
        feature_summarize[7]['max']) + " = " + str(
        ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max']))))
    print(str(feature_summarize[8]['min']) + " <= " + str(d.tide_variation) + " <= " + str(
        feature_summarize[8]['max']) + " = " + str(
        ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max']))))
    print(str((((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max'])) &
               ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max'])) &
               ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max'])) &
               ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max'])) &
               ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max'])) &
               ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max'])) &
               ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max'])) &
               ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max'])) &
               ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max'])))))

    d.loc[(((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max'])) &
          ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max'])) &
          ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max'])) &
          ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max'])) &
          ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max'])) &
          ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max'])) &
          ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max'])) &
          ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max'])) &
          ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max'])))
    , 'cluster'] = (index+1)

d.to_csv(load_path+"time_drifting_cluster.csv")

print("\nAdding P(s) Column : ")
import scipy as sp
import scipy.stats

d['p(mobS)'] = 0

for index in range(0, len(d), 1):
    data_columns = list(d.columns.values)
    if (int(d[index:index+1]['prediction']) == 1) :
        # 범위에 들어왔을 때, 확인
        print("Length : " + str(d[index:index+1]))
        print("Index : " + str(index))
        for index2 in range(0, len(clusters_summarize), 1):
            ps = []
            feature_summarize = clusters_summarize[index2]
            print("Features Length : " + str(len(feature_summarize)))
            for index3 in range(0, len(feature_summarize), 1):
                print("Feature : " + str(feature_summarize[index3]))
                print("Mean : " + str(feature_summarize[index3]['mean']))
                print("STD : " + str(feature_summarize[index3]['std']))
                print("Feature : " + data_columns[index3])
                print("Value : " + str(float(d[index:index + 1][data_columns[index3]])))
                rv = sp.stats.norm(loc=feature_summarize[index3]['mean'], scale=feature_summarize[index3]['std'])
                prob = rv.cdf(float(d[index:index + 1][data_columns[index3]]))
                print("Result : ")
                print(prob)
                ps.append(prob)

            print("Prob List : ")
            print(ps)
            add = 0
            for p in ps:
                print(p)
                add += p

            print("Length " + str(len(ps)))
            d.loc[index:index+1, len(data_columns)-1:len(data_columns)] = add / len(ps)
    else:
        d.loc[index:index + 1, len(data_columns) - 1:len(data_columns)] = 0


## P(S) 구하기!
print("P(S) \n")

def matrixBasian(datas, cluster_id, acc, res):
    result = datas[datas[res] > 0]

    total_length = len(datas) # 4331
    pred_length = len(result) # 745
    accident_p = len(result[result[acc] == cluster_id]) # 101

    print(str(accident_p) + " / " + str(pred_length) + " / " + str(total_length))

    p_sp = accident_p / pred_length
    p_p = pred_length / total_length
    p_res = ((p_sp*p_p) / ((p_sp*p_p)+((1-p_sp)*(1-p_p))))
    print("result : " + str(p_res))
    return p_res


d['p(pred|s)'] = 0
d['p(mobS)*p(pred|s)'] = 0
for index in range(0, len(d), 1):
    data_columns = list(d.columns.values)
    cluster = int(d[index:index + 1]['cluster'])
    print(cluster)
    print(d[index:index + 1]['p(mobS)'])
    if(cluster > 0):
        calc = matrixBasian(d, cluster, 'cluster', 'prediction')
        d.loc[index:index + 1, len(data_columns) - 2:len(data_columns) - 1] = calc
        d.loc[index:index + 1, len(data_columns) - 1:len(data_columns)] = float(d[index:index + 1]['p(mobS)']) * calc
    else:
        d.loc[index:index + 1, len(data_columns) - 2:len(data_columns)-1] = 0
        d.loc[index:index + 1, len(data_columns) - 1:len(data_columns)] = 0

# ## predict all (S) 구하기!
# print("P(drifting) \n")
#
#
# def matrixBasianDrifting(datas, acc, res):
#     result = datas[datas[res] > 0]
#
#     total_length = len(datas) # 4331
#     pred_length = len(result) # 745
#     accident_p = len(result[result[acc]> 0]) # 101
#     print(str(accident_p) + " / " + str(pred_length) + " / " + str(total_length))
#
#     p_sp = accident_p / pred_length
#     p_p = pred_length / total_length
#     p_res = ((p_sp*p_p) / ((p_sp*p_p)+((1-p_sp)*(1-p_p))))
#     print("result : " + str(p_res))
#     return p_res
#
# d['prob(drifting|pred)'] = matrixBasianDrifting(d, 'prediction', 'drifting')
#
# d['prob(drifting)'] = d['p(mobS)*prob(pred|s)']*d['prob(drifting|pred)']

temp_data.to_csv(load_path+"time_drifting_cluster.csv")
d.to_csv(load_path+"time_drifting_ps.csv")
print(d)
