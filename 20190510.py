import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

load_path = 'C:/Users/qewqs/hooney/Data/csv/daecheon_beach/20190509/'
load_drift_file = 'drifting/normal_1.csv'
load_drown_file = 'drowning/normal_1.csv'

save_drift_path = 'drifting/save/'
save_drown_path = 'drowning/save/'

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

    return temp_dataset, dataset

def get_drifting_time(path, file):
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
    drifting = dataset['drifting']

    temp_dataset = pd.concat([hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation], axis=1)


    return temp_dataset, dataset

# drifting_features, drifting_datase = get_drifting_time(load_path, load_drift_file)
# drowning_features, drowning_dataset = get_drowning_time(load_path, load_drown_file)

def situation_time_10(data):
    print("\n\nsituation_time_10")

    print("\n\n Data : ")
    print(data.head())

    data['prediction'] = 0
    data['cluster_type'] = 0

    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791667) &
              (0.621943 <= data.wind_dir) & (data.wind_dir <= 0.996945) &
              (0.022036 <= data.current_dir) & (data.current_dir <= 0.488973) &
              (0.056604 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
              (0.062636 <= data.current_speed) & (data.current_speed <= 0.725935) &
              (0.03694 <= data.wave_height) & (data.wave_height <= 0.258332) &
              (0.301303 <= data.water_temp) & (data.water_temp <= 0.758840) &
              (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.509092) &
              (0.028697 <= data.tide_variation) & (data.tide_variation <= 0.557764)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.791668) &
             (0.557221 <= data.wind_dir) & (data.wind_dir <= 0.971112) &
             (0.250602 <= data.current_dir) & (data.current_dir <= 0.938844) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.40567) &
             (0.049593 <= data.current_speed) & (data.current_speed <= 0.323617) &
             (0.054544 <= data.wave_height) & (data.wave_height <= 0.266277) &
             (0.238932 <= data.water_temp) & (data.water_temp <= 0.749559) &
             (0.315151 <= data.danger_depth) & (data.danger_depth <= 0.593940) &
             (0.272260 <= data.tide_variation) & (data.tide_variation <= 0.618838)
    , 'prediction'] = 1
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.331666 <= data.wind_dir) & (data.wind_dir <= 0.810557) &
             (0.078355 <= data.current_dir) & (data.current_dir <= 0.136367) &
             (0.188678 <= data.wind_speed) & (data.wind_speed <= 0.566039) &
             (0.258671 <= data.current_speed) & (data.current_speed <= 0.624724) &
             (0.162462 <= data.wave_height) & (data.wave_height <= 0.632259) &
             (0.406102 <= data.water_temp) & (data.water_temp <= 0.657853) &
             (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.442424) &
             (0.036792 <= data.tide_variation) & (data.tide_variation <= 0.288448)
    , 'prediction'] = 1
    data.loc[(0.416666 <= data.hour) & (data.hour <= 0.583334) &
             (0.1475 <= data.wind_dir) & (data.wind_dir <= 0.6325) &
             (0.496134 <= data.current_dir) & (data.current_dir <= 0.580163) &
             (0.160378 <= data.wind_speed) & (data.wind_speed <= 0.443397) &
             (0.357933 <= data.current_speed) & (data.current_speed <= 0.516680) &
             (0.033431 <= data.wave_height) & (data.wave_height <= 0.214077) &
             (0.448253 <= data.water_temp) & (data.water_temp <= 0.637247) &
             (0.630302 <= data.danger_depth) & (data.danger_depth <= 0.812122) &
             (0.065488 <= data.tide_variation) & (data.tide_variation <= 0.437087)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.833332) &
             (0.111943 <= data.wind_dir) & (data.wind_dir <= 0.569723) &
             (0.101666 <= data.current_dir) & (data.current_dir <= 0.358172) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.069298 <= data.current_speed) & (data.current_speed <= 0.371440) &
             (0.06392 <= data.wave_height) & (data.wave_height <= 0.503813) &
             (0.266885 <= data.water_temp) & (data.water_temp <= 0.628407) &
             (0.272726 <= data.danger_depth) & (data.danger_depth <= 0.612122) &
             (0.183222 <= data.tide_variation) & (data.tide_variation <= 0.605593)
    , 'prediction'] = 1
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.529166 <= data.wind_dir) & (data.wind_dir <= 0.998057) &
             (0.303031 <= data.current_dir) & (data.current_dir <= 0.592038) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.490567) &
             (0.044058 <= data.current_speed) & (data.current_speed <= 0.480296) &
             (0.036363 <= data.wave_height) & (data.wave_height <= 0.293843) &
             (0.352293 <= data.water_temp) & (data.water_temp <= 0.793865) &
             (0.406060 <= data.danger_depth) & (data.danger_depth <= 0.8) &
             (0.055922 <= data.tide_variation) & (data.tide_variation <= 0.393673)
    , 'prediction'] = 1
    data.loc[(0.541666 <= data.hour) & (data.hour <= 0.791668) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.355834) &
             (0.07169 <= data.current_dir) & (data.current_dir <= 0.187524) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.320756) &
             (0.215571 <= data.current_speed) & (data.current_speed <= 0.521994) &
             (0.040149 <= data.wave_height) & (data.wave_height <= 0.254546) &
             (0.262300 <= data.water_temp) & (data.water_temp <= 0.662273) &
             (0.145454 <= data.danger_depth) & (data.danger_depth <= 0.466668) &
             (0.036055 <= data.tide_variation) & (data.tide_variation <= 0.522444)
    , 'prediction'] = 1
    data.loc[(0.541666 <= data.hour) & (data.hour <= 0.833334) &
             (0.094443 <= data.wind_dir) & (data.wind_dir <= 0.781112) &
             (0.416573 <= data.current_dir) & (data.current_dir <= 0.578797) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.537737) &
             (0.115497 <= data.current_speed) & (data.current_speed <= 0.463027) &
             (0.077418 <= data.wave_height) & (data.wave_height <= 0.194136) &
             (0.102091 <= data.water_temp) & (data.water_temp <= 0.424279) &
             (0.557575 <= data.danger_depth) & (data.danger_depth <= 0.769698) &
             (0.063281 <= data.tide_variation) & (data.tide_variation <= 0.313467)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.644168) &
             (0.274652 <= data.current_dir) & (data.current_dir <= 0.60082) &
             (0.311320 <= data.wind_speed) & (data.wind_speed <= 0.688680) &
             (0.04258 <= data.current_speed) & (data.current_speed <= 0.328562) &
             (0.146040 <= data.wave_height) & (data.wave_height <= 0.606453) &
             (0.273460 <= data.water_temp) & (data.water_temp <= 0.651997) &
             (0.369696 <= data.danger_depth) & (data.danger_depth <= 0.757577) &
             (0.042677 <= data.tide_variation) & (data.tide_variation <= 0.353202)
    , 'prediction'] = 1
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.440557) &
             (0.418471 <= data.current_dir) & (data.current_dir <= 0.932617) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.462265) &
             (0.04523 <= data.current_speed) & (data.current_speed <= 0.369890) &
             (0.028151 <= data.wave_height) & (data.wave_height <= 0.322582) &
             (0.289094 <= data.water_temp) & (data.water_temp <= 0.658792) &
             (0.430302 <= data.danger_depth) & (data.danger_depth <= 0.69698) &
             (0.049300 <= data.tide_variation) & (data.tide_variation <= 0.520972)
    , 'prediction'] = 1



    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791667) &
             (0.621943 <= data.wind_dir) & (data.wind_dir <= 0.996945) &
             (0.022036 <= data.current_dir) & (data.current_dir <= 0.488973) &
             (0.056604 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.062636 <= data.current_speed) & (data.current_speed <= 0.725935) &
             (0.03694 <= data.wave_height) & (data.wave_height <= 0.258332) &
             (0.301303 <= data.water_temp) & (data.water_temp <= 0.758840) &
             (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.509092) &
             (0.028697 <= data.tide_variation) & (data.tide_variation <= 0.557764)
    , 'cluster_type'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.791668) &
             (0.557221 <= data.wind_dir) & (data.wind_dir <= 0.971112) &
             (0.250602 <= data.current_dir) & (data.current_dir <= 0.938844) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.40567) &
             (0.049593 <= data.current_speed) & (data.current_speed <= 0.323617) &
             (0.054544 <= data.wave_height) & (data.wave_height <= 0.266277) &
             (0.238932 <= data.water_temp) & (data.water_temp <= 0.749559) &
             (0.315151 <= data.danger_depth) & (data.danger_depth <= 0.593940) &
             (0.272260 <= data.tide_variation) & (data.tide_variation <= 0.618838)
    , 'cluster_type'] = 2
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.331666 <= data.wind_dir) & (data.wind_dir <= 0.810557) &
             (0.078355 <= data.current_dir) & (data.current_dir <= 0.136367) &
             (0.188678 <= data.wind_speed) & (data.wind_speed <= 0.566039) &
             (0.258671 <= data.current_speed) & (data.current_speed <= 0.624724) &
             (0.162462 <= data.wave_height) & (data.wave_height <= 0.632259) &
             (0.406102 <= data.water_temp) & (data.water_temp <= 0.657853) &
             (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.442424) &
             (0.036792 <= data.tide_variation) & (data.tide_variation <= 0.288448)
    , 'cluster_type'] = 3
    data.loc[(0.416666 <= data.hour) & (data.hour <= 0.583334) &
             (0.1475 <= data.wind_dir) & (data.wind_dir <= 0.6325) &
             (0.496134 <= data.current_dir) & (data.current_dir <= 0.580163) &
             (0.160378 <= data.wind_speed) & (data.wind_speed <= 0.443397) &
             (0.357933 <= data.current_speed) & (data.current_speed <= 0.516680) &
             (0.033431 <= data.wave_height) & (data.wave_height <= 0.214077) &
             (0.448253 <= data.water_temp) & (data.water_temp <= 0.637247) &
             (0.630302 <= data.danger_depth) & (data.danger_depth <= 0.812122) &
             (0.065488 <= data.tide_variation) & (data.tide_variation <= 0.437087)
    , 'cluster_type'] = 4
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.833332) &
             (0.111943 <= data.wind_dir) & (data.wind_dir <= 0.569723) &
             (0.101666 <= data.current_dir) & (data.current_dir <= 0.358172) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.069298 <= data.current_speed) & (data.current_speed <= 0.371440) &
             (0.06392 <= data.wave_height) & (data.wave_height <= 0.503813) &
             (0.266885 <= data.water_temp) & (data.water_temp <= 0.628407) &
             (0.272726 <= data.danger_depth) & (data.danger_depth <= 0.612122) &
             (0.183222 <= data.tide_variation) & (data.tide_variation <= 0.605593)
    , 'cluster_type'] = 5
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.529166 <= data.wind_dir) & (data.wind_dir <= 0.998057) &
             (0.303031 <= data.current_dir) & (data.current_dir <= 0.592038) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.490567) &
             (0.044058 <= data.current_speed) & (data.current_speed <= 0.480296) &
             (0.036363 <= data.wave_height) & (data.wave_height <= 0.293843) &
             (0.352293 <= data.water_temp) & (data.water_temp <= 0.793865) &
             (0.406060 <= data.danger_depth) & (data.danger_depth <= 0.8) &
             (0.055922 <= data.tide_variation) & (data.tide_variation <= 0.393673)
    , 'cluster_type'] = 6
    data.loc[(0.541666 <= data.hour) & (data.hour <= 0.791668) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.355834) &
             (0.07169 <= data.current_dir) & (data.current_dir <= 0.187524) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.320756) &
             (0.215571 <= data.current_speed) & (data.current_speed <= 0.521994) &
             (0.040149 <= data.wave_height) & (data.wave_height <= 0.254546) &
             (0.262300 <= data.water_temp) & (data.water_temp <= 0.662273) &
             (0.145454 <= data.danger_depth) & (data.danger_depth <= 0.466668) &
             (0.036055 <= data.tide_variation) & (data.tide_variation <= 0.522444)
    , 'cluster_type'] = 7
    data.loc[(0.541666 <= data.hour) & (data.hour <= 0.833334) &
             (0.094443 <= data.wind_dir) & (data.wind_dir <= 0.781112) &
             (0.416573 <= data.current_dir) & (data.current_dir <= 0.578797) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.537737) &
             (0.115497 <= data.current_speed) & (data.current_speed <= 0.463027) &
             (0.077418 <= data.wave_height) & (data.wave_height <= 0.194136) &
             (0.102091 <= data.water_temp) & (data.water_temp <= 0.424279) &
             (0.557575 <= data.danger_depth) & (data.danger_depth <= 0.769698) &
             (0.063281 <= data.tide_variation) & (data.tide_variation <= 0.313467)
    , 'cluster_type'] = 8
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.644168) &
             (0.274652 <= data.current_dir) & (data.current_dir <= 0.60082) &
             (0.311320 <= data.wind_speed) & (data.wind_speed <= 0.688680) &
             (0.04258 <= data.current_speed) & (data.current_speed <= 0.328562) &
             (0.146040 <= data.wave_height) & (data.wave_height <= 0.606453) &
             (0.273460 <= data.water_temp) & (data.water_temp <= 0.651997) &
             (0.369696 <= data.danger_depth) & (data.danger_depth <= 0.757577) &
             (0.042677 <= data.tide_variation) & (data.tide_variation <= 0.353202)
    , 'cluster_type'] = 9
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.440557) &
             (0.418471 <= data.current_dir) & (data.current_dir <= 0.932617) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.462265) &
             (0.04523 <= data.current_speed) & (data.current_speed <= 0.369890) &
             (0.028151 <= data.wave_height) & (data.wave_height <= 0.322582) &
             (0.289094 <= data.water_temp) & (data.water_temp <= 0.658792) &
             (0.430302 <= data.danger_depth) & (data.danger_depth <= 0.69698) &
             (0.049300 <= data.tide_variation) & (data.tide_variation <= 0.520972)
    , 'cluster_type'] = 10

    data.to_csv(load_path + save_drift_path+"drifting_result2.csv")

    crs = pd.crosstab(data['drifting'], data['prediction'])
    tp = crs[0][0]
    tn = crs[0][1]
    fp = crs[1][0]
    fn = crs[1][1]
    print(crs)
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    ACC = (tp + tn) / (tp + tn + fp + fn)
    cluster_2_accuracy = len(data[data['drifting'] == data['prediction']]) / len(data)
    print('K=2 KMeans -> {0:.4f}%'.format(cluster_2_accuracy * 100))
    print("\n\n Confusion Matrix")
    print("TPR : " + str(TPR))
    print("TNR : " + str(TNR))
    print("ACC : " + str(ACC))

def situation_Time_6(data):
    save_file = 'time_6_validation.csv'
    print("\n\n"+save_file)

    data['prediction'] = 0
    data['cluster_type'] = 0

    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.513057) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.955279) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.235850) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.343617) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.170675) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.601283) &
             (0.362415 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.455483)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.108903 <= data.tide_variation) & (data.tide_variation <= 0.362768)
    , 'prediction'] = 1
    data.loc[(0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145454 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'prediction'] = 1
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.680277 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.349058) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.267272 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.336480 <= data.current_dir) & (data.current_dir <= 0.472084) &
             (0.0 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.130553 <= data.current_speed) & (data.current_speed <= 0.204060) &
             (0.066275 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.582056 <= data.water_temp) & (data.water_temp <= 0.638407) &
             (0.389261 <= data.danger_depth) & (data.danger_depth <= 0.597316) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.643705) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.716982) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.856306) &
             (0.133801 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.52349 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.238412)
    , 'prediction'] = 1




    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.513057) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.955279) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.235850) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.343617) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.170675) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.601283) &
             (0.362415 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.455483)
    , 'cluster_type'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.108903 <= data.tide_variation) & (data.tide_variation <= 0.362768)
    , 'cluster_type'] = 2
    data.loc[(0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145454 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'cluster_type'] = 3
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.680277 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.349058) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.267272 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'cluster_type'] = 4
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.336480 <= data.current_dir) & (data.current_dir <= 0.472084) &
             (0.0 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.130553 <= data.current_speed) & (data.current_speed <= 0.204060) &
             (0.066275 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.582056 <= data.water_temp) & (data.water_temp <= 0.638407) &
             (0.389261 <= data.danger_depth) & (data.danger_depth <= 0.597316) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'cluster_type'] = 5
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.643705) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.716982) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.856306) &
             (0.133801 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.52349 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.238412)
    , 'cluster_type'] = 6

    data.to_csv(load_path + save_drown_path + "drowning_result2.csv")

    crs = pd.crosstab(data['drowning'], data['prediction'])
    tp = crs[0][0]
    tn = crs[0][1]
    fp = crs[1][0]
    fn = crs[1][1]
    print(crs)
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    ACC = (tp + tn) / (tp + tn + fp + fn)
    cluster_2_accuracy = len(data[data['drowning'] == data['prediction']]) / len(data)
    print('K=2 KMeans -> {0:.4f}%'.format(cluster_2_accuracy * 100))
    print("\n\n Confusion Matrix")
    print("TPR : " + str(TPR))
    print("TNR : " + str(TNR))
    print("ACC : " + str(ACC))

test_drifting_dataset = pd.read_csv('C:/Users/qewqs/hooney/Data/csv/daecheon_beach/20190509/drifting/' + 'normal_drifting.csv')
test_drowning_dataset = pd.read_csv('C:/Users/qewqs/hooney/Data/csv/daecheon_beach/20190509/drowning/' + 'normal_drowning.csv')
# data_columns = list(test_dataset.columns.values)

situation_time_10(test_drifting_dataset)
situation_Time_6(test_drowning_dataset)