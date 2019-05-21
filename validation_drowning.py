import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

path = 'D:/data/csv/daecheon_beach/20190502/'
save_path = 'D:/data/csv/daecheon_beach/20190502/drowning/'

load_file = 'normal_range_time.csv'

dataset = pd.read_csv(path + load_file)
data_columns = list(dataset.columns.values)

def situation_ALL_8():
    save_file = 'ALL_8_validation.csv'
    data = dataset
    print("\n\n"+save_file)

    data['prediction'] = 0

    data.loc[(0.138461 <= data.daily) & (data.daily <= 0.615386) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.680279) &
             (0.072221 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.132074 <= data.wind_speed) & (data.wind_speed <= 0.386793) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.601725) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.103016 <= data.tide_variation) & (data.tide_variation <= 0.362768)
    , 'prediction'] = 1
    data.loc[(0.723076 <= data.daily) & (data.daily <= 0.907693) &
             (0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145454 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'prediction'] = 1
    data.loc[(0.692307 <= data.daily) & (data.daily <= 0.969232) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.127360 <= data.current_dir) & (data.current_dir <= 0.472084) &
             (0.0 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.130553 <= data.current_speed) & (data.current_speed <= 0.371440) &
             (0.066275 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.574402 <= data.water_temp) & (data.water_temp <= 0.638407) &
             (0.268455 <= data.danger_depth) & (data.danger_depth <= 0.597316) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'prediction'] = 1
    data.loc[(0.738461 <= data.daily) & (data.daily <= 1.0) &
             (0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.780832 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.264152) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.063342 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.652603 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[((0.169231 - (0.169231*0.03)) <= data.daily) & (data.daily <= (0.169231 + (0.169231*0.03))) &
             ((0.75 - (0.75*0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75*0.03))) &
             ((0.450833 - (0.450833*0.03)) <= data.wind_dir) & (data.wind_dir <= (0.450833 + (0.450833*0.03))) &
             ((0.643704 - (0.643704*0.03)) <= data.current_dir) & (data.current_dir <= (0.643704 + (0.643704*0.03))) &
             ((0.716981 - (0.716981*0.03)) <= data.wind_speed) & (data.wind_speed <= (0.716981 + (0.716981*0.03))) &
             ((0.20893 - (0.20893*0.03)) <= data.current_speed) & (data.current_speed <= (0.20893 + (0.20893*0.03))) &
             ((0.856305 - (0.856305*0.03)) <= data.wave_height) & (data.wave_height <= (0.856305 + (0.856305*0.03))) &
             ((0.133802 - (0.133802*0.03)) <= data.water_temp) & (data.water_temp <= (0.133802 + (0.133802*0.03))) &
             ((0.52349 - (0.52349*0.03)) <= data.danger_depth) & (data.danger_depth <= (0.52349 + (0.52349*0.03))) &
             ((0.238411 - (0.238411*0.03)) <= data.tide_variation) & (data.tide_variation <= (0.238411 + (0.238411*0.03)))
    , 'prediction'] = 1
    data.loc[(0.230768 <= data.daily) & (data.daily <= 0.584616) &
             (0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.513057) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.955279) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.235850) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.343617) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.170675) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.601283) &
             (0.362415 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.455483)
    , 'prediction'] = 1
    data.loc[(0.892307 <= data.daily) & (data.daily <= 0.953847) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.332221 <= data.wind_dir) & (data.wind_dir <= 0.334445) &
             (0.075693 <= data.current_dir) & (data.current_dir <= 0.094546) &
             (0.292452 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.40583 <= data.current_speed) & (data.current_speed <= 0.4453) &
             (0.090695 <= data.wave_height) & (data.wave_height <= 0.120822) &
             (0.64829 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.09396 <= data.danger_depth) & (data.danger_depth <= 0.234900) &
             (0.15011 <= data.tide_variation) & (data.tide_variation <= 0.24209)
    , 'prediction'] = 1
    data.loc[(0.923076 <= data.daily) & (data.daily <= 0.923078) &
             (0.5 <= data.hour) & (data.hour <= 0.708334) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.60081) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.613209) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.583579) &
             (0.555317 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.563757 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.217808)
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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

def situation_ALL_9():
    save_file = 'ALL_9_validation.csv'
    data = dataset
    print("\n\n"+save_file)

    data['prediction'] = 0

    data.loc[(0.230768 <= data.daily) & (data.daily <= 0.338463) &
             (0.5 <= data.hour) & (data.hour <= 0.541668) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.497779) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.708738) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.169812) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.171085) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.154680) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.283496) &
             (0.449663 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.179545)
    , 'prediction'] = 1
    data.loc[(0.538461 <= data.daily) & (data.daily <= 0.969232) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.311666 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.54718) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.4453) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.474118 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.09396 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.15011 <= data.tide_variation) & (data.tide_variation <= 0.362768)
    , 'prediction'] = 1
    data.loc[(0.723076 <= data.daily) & (data.daily <= 0.907693) &
             (0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145454 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'prediction'] = 1
    data.loc[((0.169231 - (0.169231 * 0.03)) <= data.daily) & (data.daily <= (0.169231 + (0.169231 * 0.03))) &
             ((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.450833 - (0.450833 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.450833 + (0.450833 * 0.03))) &
             ((0.643704 - (0.643704 * 0.03)) <= data.current_dir) & (data.current_dir <= (0.643704 + (0.643704 * 0.03))) &
             ((0.716981 - (0.716981 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.716981 + (0.716981 * 0.03))) &
             ((0.20893 - (0.20893 * 0.03)) <= data.current_speed) & (data.current_speed <= (0.20893 + (0.20893 * 0.03))) &
             ((0.856305 - (0.856305 * 0.03)) <= data.wave_height) & (data.wave_height <= (0.856305 + (0.856305 * 0.03))) &
             ((0.133802 - (0.133802 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.133802 + (0.133802 * 0.03))) &
             ((0.52349 - (0.52349 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.52349 + (0.52349 * 0.03))) &
             ((0.238411 - (0.238411 * 0.03)) <= data.tide_variation) & (data.tide_variation <= (0.238411 + (0.238411 * 0.03)))
    , 'prediction'] = 1
    data.loc[(0.738461 <= data.daily) & (data.daily <= 1.0) &
             (0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.780832 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.264152) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.063342 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.652603 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04414 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[((0.584615 - (0.584615 * 0.03)) <= data.daily) & (data.daily <= (0.584615 + (0.584615 * 0.03))) &
             ((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.513056 - (0.513056 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.513056 + (0.513056 * 0.03))) &
             ((0.955278 - (0.955278 * 0.03)) <= data.current_dir) & (data.current_dir <= (0.955278 + (0.955278 * 0.03))) &
             ((0.235849 - (0.235849 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.235849 + (0.235849 * 0.03))) &
             ((0.343616 - (0.343616 * 0.03)) <= data.current_speed) & (data.current_speed <= (0.343616 + (0.343616 * 0.03))) &
             ((0.170674 - (0.170674 * 0.03)) <= data.wave_height) & (data.wave_height <= (0.170674 + (0.170674 * 0.03))) &
             ((0.601282 - (0.601282 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.601282 + (0.601282 * 0.03))) &
             ((0.362416 - (0.362416 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.362416 + (0.362416 * 0.03))) &
             ((0.455482 - (0.455482 * 0.03)) <= data.tide_variation) & (data.tide_variation <= (0.455482 + (0.455482 * 0.03)))
    , 'prediction'] = 1
    data.loc[(0.138461 <= data.daily) & (data.daily <= 0.461539) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.680279) &
             (0.072221 <= data.current_dir) & (data.current_dir <= 0.121668) &
             (0.132074 <= data.wind_speed) & (data.wind_speed <= 0.386793) &
             (0.273357 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.270382) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.368814) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.328860) &
             (0.103016 <= data.tide_variation) & (data.tide_variation <= 0.214129)
    , 'prediction'] = 1
    data.loc[(0.923076 <= data.daily) & (data.daily <= 0.923078) &
             (0.5 <= data.hour) & (data.hour <= 0.708334) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.60081) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.613209) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.583579) &
             (0.555317 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.563757 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.217808)
    , 'prediction'] = 1
    data.loc[(0.692307 <= data.daily) & (data.daily <= 0.969232) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.336480 <= data.current_dir) & (data.current_dir <= 0.472084) &
             (0.0 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.130553 <= data.current_speed) & (data.current_speed <= 0.204060) &
             (0.066275 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.582056 <= data.water_temp) & (data.water_temp <= 0.638407) &
             (0.389261 <= data.danger_depth) & (data.danger_depth <= 0.597316) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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

def situation_ALL_10():
    save_file = 'ALL_10_validation.csv'
    data = dataset
    print("\n\n"+save_file)

    data['prediction'] = 0

    data.loc[(0.923076 <= data.daily) & (data.daily <= 0.923078) &
             (0.5 <= data.hour) & (data.hour <= 0.708334) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.60081) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.613209) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.583579) &
             (0.555317 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.563757 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.217808)
    , 'prediction'] = 1
    data.loc[(0.723076 <= data.daily) & (data.daily <= 0.969232) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.328055 <= data.wind_dir) & (data.wind_dir <= 0.334445) &
             (0.075693 <= data.current_dir) & (data.current_dir <= 0.155582) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.335886 <= data.current_speed) & (data.current_speed <= 0.4453) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.574402 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.09396 <= data.danger_depth) & (data.danger_depth <= 0.328860) &
             (0.15011 <= data.tide_variation) & (data.tide_variation <= 0.305373)
    , 'prediction'] = 1
    data.loc[(0.538461 <= data.daily) & (data.daily <= 0.615386) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.311666 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.367926) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.273735) &
             (0.179471 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.474118 <= data.water_temp) & (data.water_temp <= 0.601725) &
             (0.194630 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.217806 <= data.tide_variation) & (data.tide_variation <= 0.362768)
    , 'prediction'] = 1
    data.loc[(0.738461 <= data.daily) & (data.daily <= 1.0) &
             (0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.780832 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.264152) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.063342 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.652603 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[(0.230768 <= data.daily) & (data.daily <= 0.338463) &
             (0.5 <= data.hour) & (data.hour <= 0.541668) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.497779) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.708738) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.169812) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.171085) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.154680) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.283496) &
             (0.449663 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.179545)
    , 'prediction'] = 1
    data.loc[(0.230768 <= data.daily) & (data.daily <= 0.338463) &
             (0.5 <= data.hour) & (data.hour <= 0.541668) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.497779) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.708738) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.169812) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.171085) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.154680) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.283496) &
             (0.449663 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.179545)
    , 'prediction'] = 1
    data.loc[(0.723076 <= data.daily) & (data.daily <= 0.907693) &
             (0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145454 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'prediction'] = 1
    data.loc[((0.169231 - (0.169231 * 0.03)) <= data.daily) & (data.daily <= (0.169231 + (0.169231 * 0.03))) &
             ((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.450833 - (0.450833 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.450833 + (0.450833 * 0.03))) &
             ((0.643704 - (0.643704 * 0.03)) <= data.current_dir) & (data.current_dir <= (0.643704 + (0.643704 * 0.03))) &
             ((0.716981 - (0.716981 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.716981 + (0.716981 * 0.03))) &
             ((0.20893 - (0.20893 * 0.03)) <= data.current_speed) & (data.current_speed <= (0.20893 + (0.20893 * 0.03))) &
             ((0.856305 - (0.856305 * 0.03)) <= data.wave_height) & (data.wave_height <= (0.856305 + (0.856305 * 0.03))) &
             ((0.133802 - (0.133802 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.133802 + (0.133802 * 0.03))) &
             ((0.52349 - (0.52349 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.52349 + (0.52349 * 0.03))) &
             ((0.238411 - (0.238411 * 0.03)) <= data.tide_variation) & (data.tide_variation <= (0.238411 + (0.238411 * 0.03)))
    , 'prediction'] = 1
    data.loc[((0.584615 - (0.584615 * 0.03)) <= data.daily) & (data.daily <= (0.584615 + (0.584615 * 0.03))) &
             ((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.513056 - (0.513056 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.513056 + (0.513056 * 0.03))) &
             ((0.955278 - (0.955278 * 0.03)) <= data.current_dir) & (
                         data.current_dir <= (0.955278 + (0.955278 * 0.03))) &
             ((0.235849 - (0.235849 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.235849 + (0.235849 * 0.03))) &
             ((0.343616 - (0.343616 * 0.03)) <= data.current_speed) & (
                         data.current_speed <= (0.343616 + (0.343616 * 0.03))) &
             ((0.170674 - (0.170674 * 0.03)) <= data.wave_height) & (
                         data.wave_height <= (0.170674 + (0.170674 * 0.03))) &
             ((0.601282 - (0.601282 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.601282 + (0.601282 * 0.03))) &
             ((0.362416 - (0.362416 * 0.03)) <= data.danger_depth) & (
                         data.danger_depth <= (0.362416 + (0.362416 * 0.03))) &
             ((0.455482 - (0.455482 * 0.03)) <= data.tide_variation) & (
                         data.tide_variation <= (0.455482 + (0.455482 * 0.03)))
    , 'prediction'] = 1
    data.loc[(0.138461 <= data.daily) & (data.daily <= 0.461539) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.680279) &
             (0.072221 <= data.current_dir) & (data.current_dir <= 0.121668) &
             (0.132074 <= data.wind_speed) & (data.wind_speed <= 0.386793) &
             (0.273357 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.270382) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.368814) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.328860) &
             (0.103016 <= data.tide_variation) & (data.tide_variation <= 0.214129)
    , 'prediction'] = 1
    data.loc[(0.692307 <= data.daily) & (data.daily <= 0.969232) &
             (0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.336480 <= data.current_dir) & (data.current_dir <= 0.472084) &
             (0.0 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.130553 <= data.current_speed) & (data.current_speed <= 0.204060) &
             (0.066275 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.582056 <= data.water_temp) & (data.water_temp <= 0.638407) &
             (0.389261 <= data.danger_depth) & (data.danger_depth <= 0.597316) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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

def situation_Time_6():
    save_file = 'time_6_validation.csv'
    data = dataset
    print("\n\n"+save_file)

    data['prediction'] = 0

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

    data.to_csv(save_path + save_file)

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

def situation_Time_7():
    save_file = 'time_7_validation.csv'
    data = dataset
    print("\n\n" + save_file)

    data['prediction'] = 0

    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.780832 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.264152) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.063342 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.652603 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[(0.541666 <= data.hour) & (data.hour <= 0.75) &
             (0.0 <= data.wind_dir) & (data.wind_dir <= 0.409723) &
             (0.336480 <= data.current_dir) & (data.current_dir <= 0.601163) &
             (0.0 <= data.wind_speed) & (data.wind_speed <= 0.377359) &
             (0.130553 <= data.current_speed) & (data.current_speed <= 0.204060) &
             (0.066275 <= data.wave_height) & (data.wave_height <= 0.177127) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.638407) &
             (0.389261 <= data.danger_depth) & (data.danger_depth <= 0.597316) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.467256)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.75) &
             (0.497777 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.955279) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.343617) &
             (0.145454 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.283494 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.455483)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.708334) &
             (0.485555 <= data.wind_dir) & (data.wind_dir <= 0.680279) &
             (0.072221 <= data.current_dir) & (data.current_dir <= 0.092478) &
             (0.132074 <= data.wind_speed) & (data.wind_speed <= 0.386793) &
             (0.379704 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.270382) &
             (0.267272 <= data.water_temp) & (data.water_temp <= 0.368814) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.087249) &
             (0.103016 <= data.tide_variation) & (data.tide_variation <= 0.214129)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.4453) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.093696 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.362728)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.708334) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.60082) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.613209) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.583579) &
             (0.555317 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.563757 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.217808)
    , 'prediction'] = 1
    data.loc[((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.450833 - (0.450833 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.450833 + (0.450833 * 0.03))) &
             ((0.643704 - (0.643704 * 0.03)) <= data.current_dir) & (
                         data.current_dir <= (0.643704 + (0.643704 * 0.03))) &
             ((0.716981 - (0.716981 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.716981 + (0.716981 * 0.03))) &
             ((0.20893 - (0.20893 * 0.03)) <= data.current_speed) & (
                         data.current_speed <= (0.20893 + (0.20893 * 0.03))) &
             ((0.856305 - (0.856305 * 0.03)) <= data.wave_height) & (
                         data.wave_height <= (0.856305 + (0.856305 * 0.03))) &
             ((0.133802 - (0.133802 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.133802 + (0.133802 * 0.03))) &
             ((0.52349 - (0.52349 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.52349 + (0.52349 * 0.03))) &
             ((0.238411 - (0.238411 * 0.03)) <= data.tide_variation) & (
                         data.tide_variation <= (0.238411 + (0.238411 * 0.03)))
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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

def situation_Time_9():
    save_file = 'time_9_validation.csv'
    data = dataset
    print("\n\n" + save_file)

    data['prediction'] = 0

    data.loc[(0.625 <= data.hour) & (data.hour <= 0.708334) &
             (0.485555 <= data.wind_dir) & (data.wind_dir <= 0.680279) &
             (0.072221 <= data.current_dir) & (data.current_dir <= 0.092478) &
             (0.132074 <= data.wind_speed) & (data.wind_speed <= 0.386793) &
             (0.379704 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.270382) &
             (0.267272 <= data.water_temp) & (data.water_temp <= 0.368814) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.087249) &
             (0.103016 <= data.tide_variation) & (data.tide_variation <= 0.214129)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.708334) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.60082) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.613209) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.583579) &
             (0.555317 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.563757 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.217808)
    , 'prediction'] = 1
    data.loc[(0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145154 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'prediction'] = 1
    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.780832 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.264152) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.063342 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.652603 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.4453) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.09396 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.362768)
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
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.541668) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.497779) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.708738) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.169812) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.171085) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.154680) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.283496) &
             (0.449663 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.179545)
    , 'prediction'] = 1
    data.loc[((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.450833 - (0.450833 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.450833 + (0.450833 * 0.03))) &
             ((0.643704 - (0.643704 * 0.03)) <= data.current_dir) & (data.current_dir <= (0.643704 + (0.643704 * 0.03))) &
             ((0.716981 - (0.716981 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.716981 + (0.716981 * 0.03))) &
             ((0.20893 - (0.20893 * 0.03)) <= data.current_speed) & (data.current_speed <= (0.20893 + (0.20893 * 0.03))) &
             ((0.856305 - (0.856305 * 0.03)) <= data.wave_height) & (data.wave_height <= (0.856305 + (0.856305 * 0.03))) &
             ((0.133802 - (0.133802 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.133802 + (0.133802 * 0.03))) &
             ((0.52349 - (0.52349 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.52349 + (0.52349 * 0.03))) &
             ((0.238411 - (0.238411 * 0.03)) <= data.tide_variation) & (data.tide_variation <= (0.238411 + (0.238411 * 0.03)))
    , 'prediction'] = 1
    data.loc[((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.513056 - (0.513056 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.513056 + (0.513056 * 0.03))) &
             ((0.955278 - (0.955278 * 0.03)) <= data.current_dir) & (data.current_dir <= (0.955278 + (0.955278 * 0.03))) &
             ((0.235849 - (0.235849 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.235849 + (0.235849 * 0.03))) &
             ((0.343616 - (0.343616 * 0.03)) <= data.current_speed) & (data.current_speed <= (0.343616 + (0.343616 * 0.03))) &
             ((0.170674 - (0.170674 * 0.03)) <= data.wave_height) & (data.wave_height <= (0.170674 + (0.170674 * 0.03))) &
             ((0.601282 - (0.601282 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.601282 + (0.601282 * 0.03))) &
             ((0.362416 - (0.362416 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.362416 + (0.362416 * 0.03))) &
             ((0.455482 - (0.455482 * 0.03)) <= data.tide_variation) & (data.tide_variation <= (0.455482 + (0.455482 * 0.03)))
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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

def situation_Time_10():
    save_file = 'time_10_validation.csv'
    data = dataset
    print("\n\n" + save_file)

    data['prediction'] = 0

    data.loc[(0.458332 <= data.hour) & (data.hour <= 0.75) &
             (0.780832 <= data.wind_dir) & (data.wind_dir <= 0.990834) &
             (0.049976 <= data.current_dir) & (data.current_dir <= 0.125603) &
             (0.066037 <= data.wind_speed) & (data.wind_speed <= 0.264152) &
             (0.18524 <= data.current_speed) & (data.current_speed <= 0.446643) &
             (0.063342 <= data.wave_height) & (data.wave_height <= 0.185338) &
             (0.652603 <= data.water_temp) & (data.water_temp <= 0.689674) &
             (0.087247 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.04415 <= data.tide_variation) & (data.tide_variation <= 0.373805)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.541668) &
             (0.383610 <= data.wind_dir) & (data.wind_dir <= 0.497779) &
             (0.601161 <= data.current_dir) & (data.current_dir <= 0.708738) &
             (0.075471 <= data.wind_speed) & (data.wind_speed <= 0.169812) &
             (0.081233 <= data.current_speed) & (data.current_speed <= 0.171085) &
             (0.102852 <= data.wave_height) & (data.wave_height <= 0.154680) &
             (0.275057 <= data.water_temp) & (data.water_temp <= 0.283496) &
             (0.449663 <= data.danger_depth) & (data.danger_depth <= 0.530202) &
             (0.139072 <= data.tide_variation) & (data.tide_variation <= 0.179545)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.708334) &
             (0.485555 <= data.wind_dir) & (data.wind_dir <= 0.680279) &
             (0.072221 <= data.current_dir) & (data.current_dir <= 0.092478) &
             (0.132074 <= data.wind_speed) & (data.wind_speed <= 0.386793) &
             (0.379704 <= data.current_speed) & (data.current_speed <= 0.520222) &
             (0.062756 <= data.wave_height) & (data.wave_height <= 0.270382) &
             (0.267272 <= data.water_temp) & (data.water_temp <= 0.368814) &
             (0.04698 <= data.danger_depth) & (data.danger_depth <= 0.087249) &
             (0.103016 <= data.tide_variation) & (data.tide_variation <= 0.214129)
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
    data.loc[((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.450833 - (0.450833 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.450833 + (0.450833 * 0.03))) &
             ((0.643704 - (0.643704 * 0.03)) <= data.current_dir) & (
                         data.current_dir <= (0.643704 + (0.643704 * 0.03))) &
             ((0.716981 - (0.716981 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.716981 + (0.716981 * 0.03))) &
             ((0.20893 - (0.20893 * 0.03)) <= data.current_speed) & (
                         data.current_speed <= (0.20893 + (0.20893 * 0.03))) &
             ((0.856305 - (0.856305 * 0.03)) <= data.wave_height) & (
                         data.wave_height <= (0.856305 + (0.856305 * 0.03))) &
             ((0.133802 - (0.133802 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.133802 + (0.133802 * 0.03))) &
             ((0.52349 - (0.52349 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.52349 + (0.52349 * 0.03))) &
             ((0.238411 - (0.238411 * 0.03)) <= data.tide_variation) & (
                         data.tide_variation <= (0.238411 + (0.238411 * 0.03)))
    , 'prediction'] = 1
    data.loc[(0.583332 <= data.hour) & (data.hour <= 0.75) &
             (0.758055 <= data.wind_dir) & (data.wind_dir <= 0.904445) &
             (0.315416 <= data.current_dir) & (data.current_dir <= 0.753797) &
             (0.141508 <= data.wind_speed) & (data.wind_speed <= 0.358492) &
             (0.11417 <= data.current_speed) & (data.current_speed <= 0.199189) &
             (0.145154 <= data.wave_height) & (data.wave_height <= 0.191790) &
             (0.657243 <= data.water_temp) & (data.water_temp <= 0.748123) &
             (0.33557 <= data.danger_depth) & (data.danger_depth <= 0.557048) &
             (0.114053 <= data.tide_variation) & (data.tide_variation <= 0.146432)
    , 'prediction'] = 1
    data.loc[(0.5 <= data.hour) & (data.hour <= 0.708334) &
             (0.438610 <= data.wind_dir) & (data.wind_dir <= 0.563890) &
             (0.207013 <= data.current_dir) & (data.current_dir <= 0.60082) &
             (0.5 <= data.wind_speed) & (data.wind_speed <= 0.613209) &
             (0.195571 <= data.current_speed) & (data.current_speed <= 0.234835) &
             (0.381231 <= data.wave_height) & (data.wave_height <= 0.583579) &
             (0.555317 <= data.water_temp) & (data.water_temp <= 0.557474) &
             (0.563757 <= data.danger_depth) & (data.danger_depth <= 0.610739) &
             (0.214863 <= data.tide_variation) & (data.tide_variation <= 0.217808)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.291388 <= data.wind_dir) & (data.wind_dir <= 0.556945) &
             (0.074166 <= data.current_dir) & (data.current_dir <= 0.221339) &
             (0.113207 <= data.wind_speed) & (data.wind_speed <= 0.367926) &
             (0.166198 <= data.current_speed) & (data.current_speed <= 0.371440) &
             (0.085576 <= data.wave_height) & (data.wave_height <= 0.408799) &
             (0.244843 <= data.water_temp) & (data.water_temp <= 0.601725) &
             (0.194630 <= data.danger_depth) & (data.danger_depth <= 0.342283) &
             (0.138336 <= data.tide_variation) & (data.tide_variation <= 0.362768)
    , 'prediction'] = 1
    data.loc[(0.625 <= data.hour) & (data.hour <= 0.75) &
             (0.332221 <= data.wind_dir) & (data.wind_dir <= 0.334445) &
             (0.075693 <= data.current_dir) & (data.current_dir <= 0.094546) &
             (0.292452 <= data.wind_speed) & (data.wind_speed <= 0.54717) &
             (0.40583 <= data.current_speed) & (data.current_speed <= 0.4453) &
             (0.090695 <= data.wave_height) & (data.wave_height <= 0.120822) &
             (0.64829 <= data.water_temp) & (data.water_temp <= 0.673653) &
             (0.09396 <= data.danger_depth) & (data.danger_depth <= 0.234900) &
             (0.15011 <= data.tide_variation) & (data.tide_variation <= 0.24209)
    , 'prediction'] = 1
    data.loc[((0.75 - (0.75 * 0.03)) <= data.hour) & (data.hour <= (0.75 + (0.75 * 0.03))) &
             ((0.513056 - (0.513056 * 0.03)) <= data.wind_dir) & (data.wind_dir <= (0.513056 + (0.513056 * 0.03))) &
             ((0.955278 - (0.955278 * 0.03)) <= data.current_dir) & (data.current_dir <= (0.955278 + (0.955278 * 0.03))) &
             ((0.235849 - (0.235849 * 0.03)) <= data.wind_speed) & (data.wind_speed <= (0.235849 + (0.235849 * 0.03))) &
             ((0.343616 - (0.343616 * 0.03)) <= data.current_speed) & (data.current_speed <= (0.343616 + (0.343616 * 0.03))) &
             ((0.170674 - (0.170674 * 0.03)) <= data.wave_height) & (data.wave_height <= (0.170674 + (0.170674 * 0.03))) &
             ((0.601282 - (0.601282 * 0.03)) <= data.water_temp) & (data.water_temp <= (0.601282 + (0.601282 * 0.03))) &
             ((0.362416 - (0.362416 * 0.03)) <= data.danger_depth) & (data.danger_depth <= (0.362416 + (0.362416 * 0.03))) &
             ((0.455482 - (0.455482 * 0.03)) <= data.tide_variation) & (data.tide_variation <= (0.455482 + (0.455482 * 0.03)))
    , 'prediction'] = 1

    data.to_csv(save_path + save_file)

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


situation_ALL_10()
situation_Time_6()