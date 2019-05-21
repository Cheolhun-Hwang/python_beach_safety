import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

path = 'D:/data/csv/daecheon_beach/20190422/'
save_path = 'D:/data/csv/daecheon_beach/20190422/save/'

load_file = 'normal.csv'

dataset = pd.read_csv(path + load_file)
data_columns = list(dataset.columns.values)

def situation_3():
    save_file = '3_validation.csv'
    data = dataset
    print("\n\n3_validation")
    print("\n\n Data : ")
    print(data.head())

    data['prediction'] = 0

    data.loc[ (0.415385 <= data.daily) & (data.daily <= 0.969232) &
              (0.416666 <= data.hour) & (data.hour <= 0.791668) &
              (0.0 <= data.wind_dir) & (data.wind_dir <= 0.998057) &
              (0.274652 <= data.current_dir) & (data.current_dir <= 0.938844) &
              (0.0377359 <= data.wind_speed) & (data.wind_speed <= 0.688680) &
              (0.044058 <= data.current_speed) & (data.current_speed <= 0.516680) &
              (0.028151 <= data.wave_height) & (data.wave_height <= 0.606453) &
              (0.424277 <= data.water_temp) & (data.water_temp <= 0.793865) &
              (0.357575 <= data.danger_depth) & (data.danger_depth <= 0.812122) &
              (0.042677 <= data.tide_variation) & (data.tide_variation <= 0.589405)
    , 'prediction'] = 1

    data.loc[(0.030768 <= data.daily) & (data.daily <= 0.507693) &
             (0.541666 <= data.hour) & (data.hour <= 0.833334) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.924723) &
             (0.07169 <= data.current_dir) & (data.current_dir <= 0.678149) &
             (0.028302 <= data.wind_speed) & (data.wind_speed <= 0.603775) &
             (0.04259 <= data.current_speed) & (data.current_speed <= 0.463027) &
             (0.054544 <= data.wave_height) & (data.wave_height <= 0.632259) &
             (0.102091 <= data.water_temp) & (data.water_temp <= 0.570070) &
             (0.157575 <= data.danger_depth) & (data.danger_depth <= 0.769698) &
             (0.052243 <= data.tide_variation) & (data.tide_variation <= 0.522444)
    , 'prediction'] = 1

    data.loc[(0.33846 <= data.daily) & (data.daily <= 1.0) &
             (0.458333 <= data.hour) & (data.hour <= 0.0833334) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.996944) &
             (0.022037 <= data.current_dir) & (data.current_dir <= 0.0488973) &
             (0.028302 <= data.wind_speed) & (data.wind_speed <= 0.566039) &
             (0.062637 <= data.current_speed) & (data.current_speed <= 0.725935) &
             (0.03695 <= data.wave_height) & (data.wave_height <= 0.585338) &
             (0.406103 <= data.water_temp) & (data.water_temp <= 0.758840) &
             (0.133333 <= data.danger_depth) & (data.danger_depth <= 0.509092) &
             (0.028698 <= data.tide_variation) & (data.tide_variation <= 0.618838)
    , 'prediction'] =1

    data.to_csv(save_path + save_file)

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

def situation_2():
    save_file = '2_validation.csv'
    data = dataset
    print("\n\n2_validation")
    print("\n\n Data : ")
    print(data.head())

    data['prediction'] = 0

    data.loc[ (0.458332 <= data.hour) & (data.hour <= 0.833334) &
              (0.0 <= data.wind_dir) & (data.wind_dir <= 0.996945) &
              (0.022036 <= data.current_dir) & (data.current_dir <= 0.488973) &
              (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.566039) &
              (0.049593 <= data.current_speed) & (data.current_speed <= 0.725935) &
              (0.03694 <= data.wave_height) & (data.wave_height <= 0.632259) &
              (0.238932 <= data.water_temp) & (data.water_temp <= 0.758840) &
              (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.509092) &
              (0.028697 <= data.tide_variation) & (data.tide_variation <= 0.618838)
    , 'prediction'] = 1

    data.loc[(0.416666 <= data.hour) & (data.hour <= 0.833335) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.998057) &
             (0.274654 <= data.current_dir) & (data.current_dir <= 0.938844) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.688680) &
             (0.04259 <= data.current_speed) & (data.current_speed <= 0.516680) &
             (0.028151 <= data.wave_height) & (data.wave_height <= 0.606453) &
             (0.102091 <= data.water_temp) & (data.water_temp <= 0.793865) &
             (0.357575 <= data.danger_depth) & (data.danger_depth <= 0.812122) &
             (0.042677 <= data.tide_variation) & (data.tide_variation <= 0.589405)
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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

def situation_time_10():
    save_file = 'time_10_validation.csv'
    data = dataset
    print("\n\nsituation_time_10")

    print("\n\n Data : ")
    print(data.head())

    data['prediction'] = 0

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

    data.to_csv(save_path + save_file)

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

def situation_daily_10():
    save_file = 'daily_10_validation.csv'
    data = dataset
    print("\n\ndaily_10_validation")
    print("\n\n Data : ")
    print(data.head())

    data['prediction'] = 0

    data.loc[(0.276922 <= data.daily) & (data.daily <= 0.430770) &
             (0.541666 <= data.hour) & (data.hour <= 0.791668) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.658334) &
             (0.07168 <= data.current_dir) & (data.current_dir <= 0.629330) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.537737) &
             (0.180737 <= data.current_speed) & (data.current_speed <= 0.377566) &
             (0.067448 <= data.wave_height) & (data.wave_height <= 0.266277) &
             (0.259041 <= data.water_temp) & (data.water_temp <= 0.424334) &
             (0.157575 <= data.danger_depth) & (data.danger_depth <= 0.690910) &
             (0.081677 <= data.tide_variation) & (data.tide_variation <= 0.522444)
    , 'prediction'] = 1
    data.loc[(0.461537 <= data.daily) & (data.daily <= 0.923078) &
             (0.5 <= data.hour) & (data.hour <= 0.791668) &
             (0.336943 <= data.wind_dir) & (data.wind_dir <= 0.658334) &
             (0.274652 <= data.current_dir) & (data.current_dir <= 0.856112) &
             (0.264152 <= data.wind_speed) & (data.wind_speed <= 0.688680) &
             (0.058892 <= data.current_speed) & (data.current_speed <= 0.441403) &
             (0.137242 <= data.wave_height) & (data.wave_height <= 0.606453) &
             (0.424277 <= data.water_temp) & (data.water_temp <= 0.651997) &
             (0.369696 <= data.danger_depth) & (data.danger_depth <= 0.8) &
             (0.042677 <= data.tide_variation) & (data.tide_variation <= 0.54600)
    , 'prediction'] = 1
    data.loc[(0.476922 <= data.daily) & (data.daily <= 0.892309) &
             (0.458333 <= data.hour) & (data.hour <= 0.791668) &
             (0.525277 <= data.wind_dir) & (data.wind_dir <= 0.950279) &
             (0.078518 <= data.current_dir) & (data.current_dir <= 0.481297) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.367926) &
             (0.044058 <= data.current_speed) & (data.current_speed <= 0.291219) &
             (0.058064 <= data.wave_height) & (data.wave_height <= 0.19825) &
             (0.484247 <= data.water_temp) & (data.water_temp <= 0.783590) &
             (0.29696 <= data.danger_depth) & (data.danger_depth <= 0.612122) &
             (0.071375 <= data.tide_variation) & (data.tide_variation <= 0.618838)
    , 'prediction'] = 1
    data.loc[(0.2 <= data.daily) & (data.daily <= 0.923078) &
             (0.458332 <= data.hour) & (data.hour <= 0.833334) &
             (0.448332 <= data.wind_dir) & (data.wind_dir <= 0.913057) &
             (0.038934 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.566039) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.518436) &
             (0.03694 <= data.wave_height) & (data.wave_height <= 0.632259) &
             (0.301303 <= data.water_temp) & (data.water_temp <= 0.60886) &
             (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.448486) &
             (0.036791 <= data.tide_variation) & (data.tide_variation <= 0.605593)
    , 'prediction'] = 1
    data.loc[(0.538461 <= data.daily) & (data.daily <= 0.969232) &
             (0.416666 <= data.hour) & (data.hour <= 0.583334) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.63251) &
             (0.496133 <= data.current_dir) & (data.current_dir <= 0.580163) &
             (0.037735 <= data.wind_speed) & (data.wind_speed <= 0.330190) &
             (0.357933 <= data.current_speed) & (data.current_speed <= 0.516680) &
             (0.028151 <= data.wave_height) & (data.wave_height <= 0.214077) &
             (0.448253 <= data.water_temp) & (data.water_temp <= 0.637247) &
             (0.630302 <= data.danger_depth) & (data.danger_depth <= 0.812122) &
             (0.065488 <= data.tide_variation) & (data.tide_variation <= 0.437087)
    , 'prediction'] = 1
    data.loc[(0.030768 <= data.daily) & (data.daily <= 0.415386) &
             (0.583332 <= data.hour) & (data.hour <= 0.833334) &
             (0.334443 <= data.wind_dir) & (data.wind_dir <= 0.924723) &
             (0.188703 <= data.current_dir) & (data.current_dir <= 0.578797) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.603775) &
             (0.04258 <= data.current_speed) & (data.current_speed <= 0.463027) &
             (0.054544 <= data.wave_height) & (data.wave_height <= 0.479766) &
             (0.102091 <= data.water_temp) & (data.water_temp <= 0.449305) &
             (0.315151 <= data.danger_depth) & (data.danger_depth <= 0.769468) &
             (0.052243 <= data.tide_variation) & (data.tide_variation <= 0.507727)
    , 'prediction'] = 1
    data.loc[(0.6 <= data.daily) & (data.daily <= 1.0) &
             (0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.626666 <= data.wind_dir) & (data.wind_dir <= 0.996945) &
             (0.022036 <= data.current_dir) & (data.current_dir <= 0.488973) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.154833 <= data.current_speed) & (data.current_speed <= 0.725935) &
             (0.05044 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.610838 <= data.water_temp) & (data.water_temp <= 0.758840) &
             (0.133332 <= data.danger_depth) & (data.danger_depth <= 0.406062) &
             (0.028697 <= data.tide_variation) & (data.tide_variation <= 0.557764)
    , 'prediction'] = 1
    data.loc[(0.615384 <= data.daily) & (data.daily <= 0.969232) &
             (0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.078818 <= data.current_dir) & (data.current_dir <= 0.358172) &
             (0.028301 <= data.wind_speed) & (data.wind_speed <= 0.424529) &
             (0.069298 <= data.current_speed) & (data.current_speed <= 0.624724) &
             (0.040148 <= data.wave_height) & (data.wave_height <= 0.453373) &
             (0.542390 <= data.water_temp) & (data.water_temp <= 0.662273) &
             (0.145454 <= data.danger_depth) & (data.danger_depth <= 0.593940) &
             (0.036055 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'prediction'] = 1
    data.loc[(0.430768 <= data.daily) & (data.daily <= 0.907693) &
             (0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.630832 <= data.wind_dir) & (data.wind_dir <= 0.998057) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.938844) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.490567) &
             (0.04679 <= data.current_speed) & (data.current_speed <= 0.480296) &
             (0.036363 <= data.wave_height) & (data.wave_height <= 0.202934) &
             (0.469191 <= data.water_temp) & (data.water_temp <= 0.793865) &
             (0.49697 <= data.danger_depth) & (data.danger_depth <= 0.739395) &
             (0.055922 <= data.tide_variation) & (data.tide_variation <= 0.526123)
    , 'prediction'] = 1
    data.loc[(0.430768 <= data.daily) & (data.daily <= 0.784616) &
             (0.458332 <= data.hour) & (data.hour <= 0.791668) &
             (0.003055 <= data.wind_dir) & (data.wind_dir <= 0.530557) &
             (0.436295 <= data.current_dir) & (data.current_dir <= 0.932617) &
             (0.056603 <= data.wind_speed) & (data.wind_speed <= 0.462265) &
             (0.04524 <= data.current_speed) & (data.current_speed <= 0.237270) &
             (0.029325 <= data.wave_height) & (data.wave_height <= 0.293843) &
             (0.438475 <= data.water_temp) & (data.water_temp <= 0.658792) &
             (0.430302 <= data.danger_depth) & (data.danger_depth <= 0.69697) &
             (0.049300 <= data.tide_variation) & (data.tide_variation <= 0.33040)
    , 'prediction'] = 1
    data.to_csv(save_path + save_file)

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

