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
save_path = './data/20190605/'
save_drift_path = save_path+'drifting/'
save_drown_path = save_path+'drowning/'
type_optimal = 'gap'

## ---------------------------------------------------------------- 교체부분

type = 'drowning'
# type = 'drifting'

# k_list = ['hour',  'wind_dir',  'current_dir']
# optimalK = 4

## -------------------------------------------------------------------------
save_file_name = file_type+"_"+type+"_"+type_optimal+"_"


########################################################################################################################
print("\nLoad Data")

def get_drowning_time(path, file, clist):
    dataset = pd.read_csv(path + file)
    features = dataset[clist]
    drowning = dataset['drowning']
    clist.append('drowning')
    data = dataset[clist]
    return features, drowning, data
def get_drowning_time_accident(path, file, clist):
    data = pd.read_csv(path + file)
    dataset = data[data['drowning'] > 0]
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
    features = dataset[clist]
    drowning = dataset['drowning']
    return features, drowning, dataset
def get_drifting_time(path, file, clist):
    dataset = pd.read_csv(path + file)
    features = dataset[clist]
    drifting = dataset['drifting']
    clist.append('drifting')
    data = dataset[clist]
    return features, drifting, data
def get_drifting_time_accident(path, file, clist):
    data = pd.read_csv(path + file)
    dataset = data[data['drifting'] > 0]
    dataset = dataset[((0.4166 <= dataset.hour) & (dataset.hour <= 0.84))]
    features = dataset[clist]
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
def loadData(combination_columns):
    if(type == 'drifting'):
        af, al, ad = get_drifting_time_accident(load_path, load_drift_file, combination_columns)
    else :
        af, al, ad = get_drowning_time_accident(load_path, load_drown_file, combination_columns)

    showData(af, al, ad)
    f_columns = list(af.columns.values)
    return af, al, ad, f_columns

########################################################################################################################
## Data K-means
def dataKmenas(featrues, op_k):
    kmeans_model_1 = KMeans(n_clusters=op_k, n_init=123)
    distances_1 = kmeans_model_1.fit(featrues)
    labels_1 = distances_1.labels_
    print(kmeans_model_1.cluster_centers_)

    return labels_1, kmeans_model_1.cluster_centers_

def loadKmeans(af, op_k):
    print("\nData K-means \n")
    print("Optimal K is : " + str(op_k))
    labels, centers = dataKmenas(af, op_k)

    temp_data = af
    temp_data['cluster'] = labels+1
    print("Temp_data Length : " + str(len(temp_data)))

    print("Centroid : \n")
    print("size : " + str(len(centers)))
    print(centers)

    print("Cluster Result : \n")
    print(temp_data.head())
    return temp_data, centers
########################################################################################################################
def splitData(temp, cent):
    ## Data Split Data
    print("\nData Split Data")
    clusters = []
    for num in range(1, len(cent)+1, 1):
        dim = temp[temp['cluster'] == num]
        clusters.append(dim)

    print("\nclusters :")
    for index in range(0, len(clusters), 1):
        print("\nclusters " + str(index+1) )
        print(clusters[index].head())
        print("Length : " + str(len(clusters[index])))

    return clusters

########################################################################################################################
## Summarize Clusters
def calc(list):
    min = np.min(list)
    max = np.max(list)
    mean = np.mean(list)
    std = np.std(list)
    return min, max, mean, std, len(list)

def summarized(cluster_list, row):
    print("\nSummarize")
    clusters_summarize = []

    for index in range(0, len(cluster_list), 1):
        print("\nclusters " + str(index+1) )
        temp_cluster = cluster_list[index]
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

    with open(save_path+save_file_name+'cluster_summarize_'+str(row)+'.txt', 'w') as f:
        for item in clusters_summarize:
            f.write("%s\n" % item)
    return clusters_summarize

########################################################################################################################
def loadTestSet(clist):
    ## Test Data set
    print("\nTest Dataset Load... \n")

    if(type == 'drifting'):
        print("type : drifting")
        f, c, d = get_drifting_time(load_path, load_drift_file, clist)

    else:
        print("type : drowning")
        f, c, d = get_drowning_time(load_path, load_drown_file, clist)

    showData(f,c,d)
    return f,c,d

########################################################################################################################
def labelPredCluster2(featrues, dataset, cluster):
    dataset.loc[(
                  ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                          dataset[featrues[0]['name']] <= featrues[0]['max']))
                  & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                  dataset[featrues[1]['name']] <= featrues[1]['max']))
          ), 'prediction'] = 1
    dataset.loc[(
                  ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                          dataset[featrues[0]['name']] <= featrues[0]['max']))
                  & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                  dataset[featrues[1]['name']] <= featrues[1]['max']))
          ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster3(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster4(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster5(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                        dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster6(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                        dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster7(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                        dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster8(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                        dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                        & ((featrues[7]['min'] <= dataset[featrues[7]['name']]) & (
                        dataset[featrues[7]['name']] <= featrues[7]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                        & ((featrues[7]['min'] <= dataset[featrues[7]['name']]) & (
                        dataset[featrues[7]['name']] <= featrues[7]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster9(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                        dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                        & ((featrues[7]['min'] <= dataset[featrues[7]['name']]) & (
                        dataset[featrues[7]['name']] <= featrues[7]['max']))
                        & ((featrues[8]['min'] <= dataset[featrues[8]['name']]) & (
                        dataset[featrues[8]['name']] <= featrues[8]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                        & ((featrues[7]['min'] <= dataset[featrues[7]['name']]) & (
                        dataset[featrues[7]['name']] <= featrues[7]['max']))
                        & ((featrues[8]['min'] <= dataset[featrues[8]['name']]) & (
                        dataset[featrues[8]['name']] <= featrues[8]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset
def labelPredCluster10(featrues, dataset, cluster):
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                        dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                        & ((featrues[7]['min'] <= dataset[featrues[7]['name']]) & (
                        dataset[featrues[7]['name']] <= featrues[7]['max']))
                        & ((featrues[8]['min'] <= dataset[featrues[8]['name']]) & (
                        dataset[featrues[8]['name']] <= featrues[8]['max']))
                        & ((featrues[9]['min'] <= dataset[featrues[9]['name']]) & (
                        dataset[featrues[9]['name']] <= featrues[9]['max']))
                ), 'prediction'] = 1
    dataset.loc[(
                        ((featrues[0]['min'] <= dataset[featrues[0]['name']]) & (
                                dataset[featrues[0]['name']] <= featrues[0]['max']))
                        & ((featrues[1]['min'] <= dataset[featrues[1]['name']]) & (
                        dataset[featrues[1]['name']] <= featrues[1]['max']))
                        & ((featrues[2]['min'] <= dataset[featrues[2]['name']]) & (
                        dataset[featrues[2]['name']] <= featrues[2]['max']))
                        & ((featrues[3]['min'] <= dataset[featrues[3]['name']]) & (
                        dataset[featrues[3]['name']] <= featrues[3]['max']))
                        & ((featrues[4]['min'] <= dataset[featrues[4]['name']]) & (
                        dataset[featrues[4]['name']] <= featrues[4]['max']))
                        & ((featrues[5]['min'] <= dataset[featrues[5]['name']]) & (
                        dataset[featrues[5]['name']] <= featrues[5]['max']))
                        & ((featrues[6]['min'] <= dataset[featrues[6]['name']]) & (
                        dataset[featrues[6]['name']] <= featrues[6]['max']))
                        & ((featrues[7]['min'] <= dataset[featrues[7]['name']]) & (
                        dataset[featrues[7]['name']] <= featrues[7]['max']))
                        & ((featrues[8]['min'] <= dataset[featrues[8]['name']]) & (
                        dataset[featrues[8]['name']] <= featrues[8]['max']))
                        & ((featrues[9]['min'] <= dataset[featrues[9]['name']]) & (
                        dataset[featrues[9]['name']] <= featrues[9]['max']))
                ), 'cluster'] = (cluster + 1)
    return dataset

def probPred(cluster_sumarize_list, dataset):
    ## P(S) 구하기!
    print("\nP(s)")
    print("\nAdding Prediction Column : ")
    print("\nAdding Cluster Column : ")
    dataset['prediction'] = 0
    dataset['cluster'] = 0
    for index in range(0, len(cluster_sumarize_list), 1):
        feature_summarize = cluster_sumarize_list[index]
        print("cluster : " + str((index+1)))
        if(len(feature_summarize) == 2):
            dataset = labelPredCluster2(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 3):
            dataset = labelPredCluster3(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 4):
            dataset = labelPredCluster4(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 5):
            dataset = labelPredCluster5(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 6):
            dataset = labelPredCluster6(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 7):
            dataset = labelPredCluster7(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 8):
            dataset = labelPredCluster8(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 9):
            dataset = labelPredCluster9(feature_summarize, dataset, index)
        elif (len(feature_summarize) == 10):
            dataset = labelPredCluster10(feature_summarize, dataset, index)

    print("\n prediction!!")
    print(dataset)
    return dataset


########################################################################################################################
def matrixBasian(datas, cluster_id):
    result = datas[datas['prediction'] > 0]

    total_length = len(datas) # 4331
    pred_length = len(result) # 745
    accident_p = len(result[result['cluster'] == cluster_id]) # 101

    print(str(accident_p) + " / " + str(pred_length) + " / " + str(total_length))

    p_sp = accident_p / pred_length
    print("result : " + str(p_sp))
    return p_sp
def probType(dataset):
    dataset['p(s|pred)'] = 0

    for index in range(0, len(dataset), 1):
        data_columns = list(dataset.columns.values)
        if (float(dataset[index:index+1]['prediction']) > 0) :
            print("\n")
            print("row : " + str(index + 1))
            print("index : " + str(index))
            print(str(dataset[index:index + 1]))

            cluster_num = int(dataset[index:index + 1]['cluster'])
            print("Cluster : " + str(cluster_num))

            p_sp = matrixBasian(dataset, cluster_num)
            dataset.iloc[index, len(data_columns)-1] = p_sp
            print("Select p(s|pred) : \n" + str(dataset.iloc[index, len(data_columns)-1]))
    return dataset


########################################################################################################################

def matrixBasianDrifting(datas):
    type_list = datas[datas[type] > 0] # drifting drowning
    type_non_list = datas[datas[type] < 1] # ~drifting ~drowning
    p_pred_type = type_list.loc[
        (type_list['prediction'] > 0)
    ]
    p_pred_non_type = type_non_list.loc[
        (type_non_list['prediction'] > 0)
    ]
    return len(datas), len(type_list), len(type_non_list), len(p_pred_type), len(p_pred_non_type)
def lastProb(d):
    total_num, type_num, none_type_num, p_pred_type_num, p_pred_non_type_num = matrixBasianDrifting(d)

    d['p('+type+')'] = type_num/total_num
    d['p(~'+type+')'] = none_type_num/total_num
    d['p(pred|'+type+')'] = p_pred_type_num/type_num
    d['p(pred|~'+type+')'] = p_pred_non_type_num/none_type_num

    d['p('+type+'|pred)'] = d['prediction'] * ( (d['p('+type+')']*d['p(pred|'+type+')'])/
                            (d['p('+type+')']*d['p(pred|'+type+')']+d['p(~'+type+')']*d['p(pred|~'+type+')']) )

    d['p('+type+'|pred)*p(s|pred)'] = d['p('+type+'|pred)']*d['p(s|pred)']
    return d

########################################################################################################################

def resultDataset(d, row):
    d.to_csv(save_path + save_file_name + "_" + str(row) +"_ps.csv", index=False)

########################################################################################################################
## Result
def result(d, f_columns, op_k, row):
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
    measure_str += "Optimal K : " + str(op_k) + "/ Pick : "
    measure_str += "type : " + type + "\n"
    measure_str += "optimal method : " + type_optimal + "\n"
    measure_str += "merge type : " + file_type + "\n"

    with open(save_path+save_file_name+'measure_'+str(row)+'.txt', 'w') as f:
        f.write("%s\n" %measure_str)

def startFunction():
    print("\nLoad Columns")
    dataset = pd.read_csv("./data/20190605/drowning_True_save_optimal_k.csv")

    print("\nStart Operation")
    for index in range(0, len(dataset), 1):
        combination_features = str(dataset.iloc[index, 0]).replace('\'', '').replace('[', '').replace(']', '')
        optimal_k = dataset.iloc[index, 1]
        print("\nFeatures : " + combination_features)
        print("\nOptimal K : " + str(optimal_k))

        train_features, train_lables, train_dataset, train_columns = loadData(combination_features.split())

        temp_data, cluster_centers = loadKmeans(train_features, optimal_k)
        split_clusters_list = splitData(temp_data, cluster_centers)
        cluster_summarize = summarized(split_clusters_list, index)

        test_features, test_labels, test_dataset = loadTestSet(combination_features.split())

        result_dataset = probPred(cluster_summarize, test_dataset)
        result_dataset = probType(result_dataset)
        result_dataset = lastProb(result_dataset)

        resultDataset(result_dataset, index)
        result(result_dataset, combination_features.split(), optimal_k, index)



startFunction()

