import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

########################################################################################################################

## Set Params
print("Set Params \n")
file_type = 'median'

load_path = 'D:/workspace/python/SaftyBeach/data/20190523/dataset/'
load_drift_file = file_type+'_drifting_normal.csv'
load_drown_file = file_type+'_drowning_normal.csv'

# load_drift_file = file_type+'_normal_2.csv'
# load_drown_file = file_type+'_normal_2.csv'

save_path = 'D:/workspace/python/SaftyBeach/data/20190523/'
save_drift_path = 'drifting/save/'
save_drown_path = 'drowning/save/'

features_length = 10

optimal_k_range = 50

# type = 'drowning'
type = 'drifting'

type_optimal = 'gap'
optimal_bash = 100


save_file_name = file_type+"_"+type+"_"+str(features_length)+"_"+type_optimal+"_"+str(optimal_k_range)+"_"

########################################################################################################################

## Load Data
print("\nLoad Data")

def combinationFeature(dataset):
    wind_dir = dataset['wind_dir']
    current_dir = dataset['current_dir']
    wind_speed = dataset['wind_speed']
    current_speed = dataset['current_speed']
    wave_height = dataset['wave_height']
    water_temp = dataset['water_temp']
    tide_height = dataset['tide_height']
    dif_tide_height = dataset['dif_tide_height']
    tide_variation = dataset['tide_variation']
    ## combine :2
    return  pd.concat([wind_dir, current_dir, wind_speed, current_speed,
                       wave_height, water_temp, tide_height, dif_tide_height, tide_variation], axis=1)



def get_drowning_time(path, file):
    dataset = pd.read_csv(path + file)


    hour = dataset['hour']
    drowning = dataset['drowning']

    temp_dataset = pd.concat([hour,
                              combinationFeature(dataset)], axis=1)

    return temp_dataset, drowning, dataset
def get_drowning_time_accident(path, file):
    data = pd.read_csv(path + file)
    dataset = data[data['drowning'] > 0]
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
    hour = dataset['hour']
    drowning = dataset['drowning']

    temp_dataset = pd.concat([hour,
                              combinationFeature(dataset)], axis=1)

    return temp_dataset, drowning, dataset
def get_drifting_time(path, file):
    dataset = pd.read_csv(path + file)

    hour = dataset['hour']
    drifting = dataset['drifting']
    temp_dataset = pd.concat([hour,
                              combinationFeature(dataset)], axis=1)

    return temp_dataset, drifting, dataset
def get_drifting_time_accident(path, file):
    data = pd.read_csv(path + file)

    dataset = data[data['drifting'] > 0]
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
    hour = dataset['hour']
    drifting = dataset['drifting']

    temp_dataset = pd.concat([hour,
                              combinationFeature(dataset)], axis=1)

    return temp_dataset, drifting, dataset

def showData(feature, label, data):
    print("Show Up Database : \n")
    print("Length : " + str(len(feature)))
    print(feature.head())
    print(label.head())
    print(data.head())
    print("\n\n")

if(type == 'drifting'):
    print("type : drifting")
    ac, lb, ad = get_drifting_time_accident(load_path, load_drift_file)
else:
    print("type : drowning")
    ac, lb, ad = get_drowning_time_accident(load_path, load_drown_file)

showData(ac, lb, ad)

f_columns = list(ac.columns.values)

print(str(f_columns))
########################################################################################################################
## Data K-means
print("\nOptimal K")

def silhouette_socore(data):
    from sklearn.metrics import silhouette_score
    Sum_of_squared_distances = []
    K = range(2, 11, 1)
    max = 0
    max_score = 0
    for n_clusters in K:
        clusterer = KMeans(n_clusters=n_clusters, n_init=123)
        preds = clusterer.fit_predict(data)

        score = silhouette_score(data, preds, metric='euclidean')
        Sum_of_squared_distances.append(score)
        # print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

        if(score > max_score):
            max_score = score
            max = n_clusters

    return max

def gap_statistic(data):
    from gap_statistic import OptimalK
    optimalK = OptimalK(parallel_backend='rust')
    n_clusters = optimalK(np.array(data), cluster_array=np.arange(1, optimal_k_range+1))

    return n_clusters

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

    return best_key, best_value, set_dic

def findK(list, type, len):
    optiK=[]
    for num in range(0, len, 1):
        print("Loop : " + str(num));
        if(type == 'gap'):
            gap = gap_statistic(list)
            optiK.append(gap)
        else:
            sil = silhouette_socore(list)
            optiK.append(sil)
    best_key, best_value, k_all = bestK(optiK)
    print('Best Key : ' + str(best_key) + '\nRate : ' + str(best_value) + ' / ' + str(optimal_bash)
          + ' = ' + str(best_value/optimal_bash))
    return best_key, best_value, k_all

optimal_k, pick_value, pick_list = findK(ac, type_optimal, optimal_bash)

print("Optimal K is : " + str(optimal_k))

## Data K-means
print("\nData K-means \n")
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

########################################################################################################################

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

########################################################################################################################

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
        cluster_dic = {'name':feature,'min':c_min, 'max':c_max, 'mean':c_mean, 'std':c_std, 'size':c_size}
        print("name : " + feature)
        print("Min : " + str(cluster_dic['min']))
        print("Max : " + str(cluster_dic['max']))
        print("Mean : " + str(cluster_dic['mean']))
        print("Std : " + str(cluster_dic['std']))
        print("Size : " + str(cluster_dic['size']))
        feature_summarize.append(cluster_dic)
    clusters_summarize.append(feature_summarize)


print("Cluster Summarize : " + str(len(clusters_summarize)))

with open(save_path+save_file_name+'cluster_summarize.txt', 'w') as f:
    for item in clusters_summarize:
        f.write("%s\n" % item)

########################################################################################################################

## Test Data set
print("\nTest Dataset Load... \n")

if(type == 'drifting'):
    print("type : drifting")
    f, c, d = get_drifting_time(load_path, load_drift_file)

else:
    print("type : drowning")
    f, c, d = get_drowning_time(load_path, load_drown_file)

showData(f,c,d)

########################################################################################################################

## P(S) 구하기!
print("P(dist_s) \n")
print("\nAdding Prediction Column : ")
d['prediction'] = 0

for index in range(0, len(clusters_summarize), 1):
    feature_summarize = clusters_summarize[index]
    print("cluster : " + str((index+1)))
    # print(str(feature_summarize[0]['min']) + " <= d.hour <= " + str(
    #     feature_summarize[0]['max']) + " = " + str(
    #     ((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))))
    # print(str(feature_summarize[1]['min']) + " <= d.wind_dir <= " + str(
    #     feature_summarize[1]['max']) + " = " + str(
    #     ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max']))))
    # print(str(feature_summarize[2]['min']) + " <= d.current_dir <= " + str(
    #     feature_summarize[2]['max']) + " = " + str(
    #     ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))))
    # print(str(feature_summarize[3]['min']) + " <= d.wind_speed <= " + str(
    #     feature_summarize[3]['max']) + " = " + str(
    #     ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max']))))
    # print(str(feature_summarize[4]['min']) + " <= d.current_speed <= " + str(
    #     feature_summarize[4]['max']) + " = " + str(
    #     ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))))
    # print(str(feature_summarize[5]['min']) + " <= d.wave_height <= " + str(
    #     feature_summarize[5]['max']) + " = " + str(
    #     ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max']))))
    # print(str(feature_summarize[6]['min']) + " <= d.water_temp <= " + str(
    #     feature_summarize[6]['max']) + " = " + str(
    #     ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max']))))
    # print(str(feature_summarize[7]['min']) + " <= d.danger_depth <= " + str(
    #     feature_summarize[7]['max']) + " = " + str(
    #     ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max']))))
    # print(str(feature_summarize[8]['min']) + " <= d.tide_variation <= " + str(
    #     feature_summarize[8]['max']) + " = " + str(
    #     ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max']))))

    d.loc[(((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))
           & ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max']))
           & ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))
           & ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max']))
           & ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))
           & ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max']))
           & ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max']))
           & ((feature_summarize[7]['min'] <= d.tide_height) & (d.tide_height <= feature_summarize[7]['max']))
           & ((feature_summarize[8]['min'] <= d.dif_tide_height) & (d.dif_tide_height <= feature_summarize[8]['max']))
           & ((feature_summarize[9]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[9]['max']))
           )
    , 'prediction'] = 1

d.to_csv(save_path+save_file_name+"prediction.csv")
print("\nAdding Cluster Column : ")
d['cluster'] = 0

for index in range(0, len(clusters_summarize), 1):
    feature_summarize = clusters_summarize[index]
    print("cluster : " + str((index+1)))
    # print(str(feature_summarize[0]['min']) + " <= d.hour <= " + str(
    #     feature_summarize[0]['max']) + " = " + str(
    #     ((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))))
    # print(str(feature_summarize[1]['min']) + " <= d.wind_dir <= " + str(
    #     feature_summarize[1]['max']) + " = " + str(
    #     ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max']))))
    # print(str(feature_summarize[2]['min']) + " <= d.current_dir <= " + str(
    #     feature_summarize[2]['max']) + " = " + str(
    #     ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))))
    # print(str(feature_summarize[3]['min']) + " <= d.wind_speed <= " + str(
    #     feature_summarize[3]['max']) + " = " + str(
    #     ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max']))))
    # print(str(feature_summarize[4]['min']) + " <= d.current_speed <= " + str(
    #     feature_summarize[4]['max']) + " = " + str(
    #     ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))))
    # print(str(feature_summarize[5]['min']) + " <= d.wave_height <= " + str(
    #     feature_summarize[5]['max']) + " = " + str(
    #     ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max']))))
    # print(str(feature_summarize[6]['min']) + " <= d.water_temp <= " + str(
    #     feature_summarize[6]['max']) + " = " + str(
    #     ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max']))))
    # print(str(feature_summarize[7]['min']) + " <= d.danger_depth <= " + str(
    #     feature_summarize[7]['max']) + " = " + str(
    #     ((feature_summarize[7]['min'] <= d.danger_depth) & (d.danger_depth <= feature_summarize[7]['max']))))
    # print(str(feature_summarize[8]['min']) + " <= d.tide_variation <= " + str(
    #     feature_summarize[8]['max']) + " = " + str(
    #     ((feature_summarize[8]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[8]['max']))))

    d.loc[(((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))
           & ((feature_summarize[1]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[1]['max']))
           & ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))
           & ((feature_summarize[3]['min'] <= d.wind_speed) & (d.wind_speed <= feature_summarize[3]['max']))
           & ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))
           & ((feature_summarize[5]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[5]['max']))
           & ((feature_summarize[6]['min'] <= d.water_temp) & (d.water_temp <= feature_summarize[6]['max']))
           & ((feature_summarize[7]['min'] <= d.tide_height) & (d.tide_height <= feature_summarize[7]['max']))
           & ((feature_summarize[8]['min'] <= d.dif_tide_height) & (d.dif_tide_height <= feature_summarize[8]['max']))
           & ((feature_summarize[9]['min'] <= d.tide_variation) & (d.tide_variation <= feature_summarize[9]['max']))
           )
    , 'cluster'] = (index+1)

d.to_csv(save_path+save_file_name+"cluster.csv")

########################################################################################################################

print("\nAdding P(s) Column : ")
import scipy as sp
import scipy.stats

d['p(dist_s)'] = 0

for index in range(0, len(d), 1):
    data_columns = list(d.columns.values)
    if (int(d[index:index+1]['prediction']) == 1) :
        # 범위에 들어왔을 때, 확인
        print("\n")
        print("row : " + str(index+1))
        print("index : " + str(index))
        print(str(d[index:index + 1]))

        cluster_num = int(d[index:index+1]['cluster'])
        print("Cluster : " + str(cluster_num))
        ps = []
        feature_summarize = clusters_summarize[cluster_num-1]
        print("Features Length : " + str(len(feature_summarize)))
        for index3 in range(0, len(feature_summarize), 1):
            print("Feature : " + data_columns[index3])
            print(str(feature_summarize[index3]))
            print("Mean : " + str(feature_summarize[index3]['mean']))
            print("STD : " + str(feature_summarize[index3]['std']))
            print("Value : " + str(float(d[index:index + 1][data_columns[index3]])))
            rv = sp.stats.norm(loc=feature_summarize[index3]['mean'], scale=feature_summarize[index3]['std'])
            prob = rv.cdf(float(d[index:index + 1][data_columns[index3]]))
            print("Result : ")
            print(prob)
            ps.append(prob)

        print("\nProb List : ")
        print(ps)
        add = 0
        for p in ps:
            add += p

        print("Prob Length : " + str(len(ps)))
        res = add / len(ps)
        print("Prob : " + str(res))
        print("Origin : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))
        d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)] = res
        print("Modify : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))
    else:
        print("\n")
        d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)] = 0
        print("Non Select : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))

########################################################################################################################

## P(S) 구하기!
print("P(S) \n")

def matrixBasian(datas, cluster_id):
    result = datas[datas['prediction'] > 0]

    total_length = len(datas) # 4331
    pred_length = len(result) # 745
    accident_p = len(result[result['cluster'] == cluster_id]) # 101

    print(str(accident_p) + " / " + str(pred_length) + " / " + str(total_length))

    p_sp = accident_p / pred_length
    p_p = pred_length / total_length
    p_res = ((p_sp*p_p) / ((p_sp*p_p)+((1-p_sp)*(1-p_p))))
    print("result : " + str(p_res))
    return p_sp, p_p, p_res

d['p(s|pred)'] = 0
d['p(pred)'] = 0
d['p(pred|s)'] = 0

for index in range(0, len(d), 1):
    data_columns = list(d.columns.values)
    if (float(d[index:index+1]['p(dist_s)']) > 0) :
        print("\n")
        print("row : " + str(index + 1))
        print("index : " + str(index))
        print(str(d[index:index + 1]))

        cluster_num = int(d[index:index + 1]['cluster'])
        print("Cluster : " + str(cluster_num))

        p_sp, p_p, p_res = matrixBasian(d, cluster_num)
        d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns) - 2] = p_sp
        d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns) - 1] = p_p
        d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)] = p_res
        print("Select p(s|pred) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns) - 2]))
        print("Select p(pred) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns) - 1]))
        print("Select p(pred|s) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))
    else:
        print("\n")
        d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns)-2] = 0
        d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns)-1] = 0
        d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)] = 0
        print("Non Select p(s|pred) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns)-2]))
        print("Non Select p(pred) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns)-1]))
        print("Non Select p(pred|s) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))


d['p(dist_s)*p(pred|s)'] = d['p(dist_s)'] * d['p(pred|s)']

########################################################################################################################

def matrixBasianDrifting(datas):
    result = datas[datas[type] > 0]

    total_length = len(datas) # 4331
    pred_length = len(result) # 745
    accident_p = len(result[result['prediction'] > 0]) # 101

    print(str(accident_p) + " / " + str(pred_length) + " / " + str(total_length))

    p_sp = accident_p / pred_length
    p_p = pred_length / total_length
    p_res = ((p_sp*p_p) / ((p_sp*p_p)+((1-p_sp)*(1-p_p))))
    print("result : " + str(p_res))
    return p_sp, p_p, p_res

d['p(pred|'+type+')'] = 0
d['p('+type+')'] = 0
d['p('+type+'|pred)'] = 0

for index in range(0, len(d), 1):
    data_columns = list(d.columns.values)
    if (float(d[index:index+1]['p(dist_s)']) > 0) :
        print("\n")
        print("row : " + str(index + 1))
        print("index : " + str(index))
        print(str(d[index:index + 1]))

        p_sp, p_p, p_res = matrixBasianDrifting(d)
        d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns) - 2] = p_sp
        d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns) - 1] = p_p
        d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)] = p_res
        print("Select p(pred|"+type+") : \n" + str(d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns) - 2]))
        print("Select p("+type+") : \n" + str(d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns) - 1]))
        print("Select p("+type+"|pred) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))
    else:
        print("\n")
        d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns)-2] = 0
        d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns)-1] = 0
        d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)] = 0
        print("Non Select p(pred|drift) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 3:len(data_columns)-2]))
        print("Non Select p(drift) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 2:len(data_columns)-1]))
        print("Non Select p(drift|pred) : \n" + str(d.iloc[index:index + 1, len(data_columns) - 1:len(data_columns)]))


d['p('+type+'|pred)*(p(dist_s)*p(pred|s))'] = d['p('+type+'|pred)'] * d['p(dist_s)*p(pred|s)']
########################################################################################################################

temp_data.to_csv(save_path+save_file_name+"cluster.csv")
d.to_csv(save_path+save_file_name+"ps.csv", index=False)
print(d)

########################################################################################################################
## Result
print("Result \n")
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(np.array(d[type]), np.array(d['prediction'])).ravel()

## 정밀도
precision = ((tp) / (tp+fp))

# 재현율
recall = ((tp)/(tp+fn))

f_measure = 2*((precision*recall) / (precision+recall))

# TPR
tpr = ((tp)/(tp+fn))

# FNR
fnr = ((fn)/(fn+tp))

print("precision : " + str(precision))
print("recall : " + str(recall))
print("f-measure : " + str(f_measure))

print("TPR : " + str(tpr))
print("FNR : " + str(fnr))

measure_str = "precision : " + str(precision) + "\n"
measure_str += "recall : " + str(recall) + "\n"
measure_str += "f_measure : " + str(f_measure) + "\n"
measure_str += "TPR : " + str(tpr) + "\n"
measure_str += "FNR : " + str(fnr) + "\n"
measure_str += "Features : " + str(f_columns) + "\n"
measure_str += "Optimal K : " + str(optimal_k) + "/ Pick : "
               # + str(pick_value) + ' / ' + str(optimal_bash) + "\n"
# measure_str += str(pick_list) + "\n"
measure_str += "type : " + type + "\n"
measure_str += "optimal method : " + type_optimal + "\n"
measure_str += "merge type : " + file_type + "\n"

with open(save_path+save_file_name+'measure.txt', 'w') as f:
    f.write("%s\n" %measure_str)

