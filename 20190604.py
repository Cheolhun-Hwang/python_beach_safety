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
file_type = 'avg'

load_path = './data/20190523/dataset/'
load_drift_file = file_type+'_drifting_normal.csv'
load_drown_file = file_type+'_drowning_normal.csv'

# load_drift_file = file_type+'_normal_2.csv'
# load_drown_file = file_type+'_normal_2.csv'

save_path = './data/20190605/'
save_drift_path = save_path+'drifting/'
save_drown_path = save_path+'drowning/'

# type = 'drowning'
type = 'drifting'
type_optimal = 'gap'

save_file_name = file_type+"_"+type+"_"+type_optimal+"_"


## ---------------------------------------------------------------- 교체부분
k_list = ['hour' , 'dif_tide_height' , 'current_dir' , 'wave_height' , 'current_speed' , 'wind_dir']
optimalK = 41
## -------------------------------------------------------------------------

########################################################################################################################
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
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
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
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
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

print("\nColumns : ")
print(str(f_columns))
########################################################################################################################
print("Optimal K is : " + str(optimalK))
########################################################################################################################
## Data K-means
print("\nData K-means \n")
def dataKmenas(featrues):
    kmeans_model_1 = KMeans(n_clusters=optimalK, n_init=123)
    distances_1 = kmeans_model_1.fit(featrues)
    labels_1 = distances_1.labels_
    print(kmeans_model_1.cluster_centers_)

    return labels_1, kmeans_model_1.cluster_centers_

labels, centers = dataKmenas(af)

temp_data = af
temp_data['cluster'] = labels+1
print("Temp_data Length : " + str(len(temp_data)))

print("Centroid : \n")
print("size : " + str(len(centers)))
print(centers)

print("Cluster Result : \n")
print(temp_data.head())
########################################################################################################################

## Data Split Data
print("\nData Split Data")
clusters = []
for num in range(1, len(centers)+1, 1):
    dim = temp_data[temp_data['cluster'] == num]
    clusters.append(dim)

print("\nclusters :")
for index in range(0, len(clusters), 1):
    print("\nclusters " + str(index+1) )
    print(clusters[index].head())
    print("Length : " + str(len(clusters[index])))

########################################################################################################################

## Summarize Clusters
print("\nSummarize")

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
        print("\nFeature : " + feature)
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


print("\nCluster Summarize : " + str(len(clusters_summarize)))

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
print("\nP(s)")
print("\nAdding Prediction Column : ")
d['prediction'] = 0

for index in range(0, len(clusters_summarize), 1):
    feature_summarize = clusters_summarize[index]
    print("cluster : " + str((index+1)))
    d.loc[(((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))
           & ((feature_summarize[1]['min'] <= d.dif_tide_height) & (d.dif_tide_height <= feature_summarize[1]['max']))
           & ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))
           & ((feature_summarize[3]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[3]['max']))
           & ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))
           & ((feature_summarize[5]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[5]['max']))
           ), 'prediction'] = 1

print("\n prediction!!")
print(d)

print("\nAdding Cluster Column : ")
d['cluster'] = 0

for index in range(0, len(clusters_summarize), 1):
    feature_summarize = clusters_summarize[index]
    print("cluster : " + str((index+1)))
    d.loc[(((feature_summarize[0]['min'] <= d.hour) & (d.hour <= feature_summarize[0]['max']))
           & ((feature_summarize[1]['min'] <= d.dif_tide_height) & (d.dif_tide_height <= feature_summarize[1]['max']))
           & ((feature_summarize[2]['min'] <= d.current_dir) & (d.current_dir <= feature_summarize[2]['max']))
           & ((feature_summarize[3]['min'] <= d.wave_height) & (d.wave_height <= feature_summarize[3]['max']))
           & ((feature_summarize[4]['min'] <= d.current_speed) & (d.current_speed <= feature_summarize[4]['max']))
           & ((feature_summarize[5]['min'] <= d.wind_dir) & (d.wind_dir <= feature_summarize[5]['max']))
           ), 'cluster'] = (index+1)

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
    if (float(d[index:index+1]['prediction']) > 0) :
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
    if (float(d[index:index+1]['prediction']) > 0) :
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
measure_str += "Optimal K : " + str(optimalK) + "/ Pick : "
measure_str += "type : " + type + "\n"
measure_str += "optimal method : " + type_optimal + "\n"
measure_str += "merge type : " + file_type + "\n"

with open(save_path+save_file_name+'measure.txt', 'w') as f:
    f.write("%s\n" %measure_str)