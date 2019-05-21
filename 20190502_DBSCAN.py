import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

load_path = 'D:/data/csv/daecheon_beach/20190502/'
save_path = 'drowning/'

drowning_save = 'cluster_drowning.csv'

def get_drowning_time():
    dataset = pd.read_csv('D:/data/csv/daecheon_beach/20190502/' + 'normal_1.csv')

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

def get_drowning_all():
    dataset = pd.read_csv('D:/data/csv/daecheon_beach/20190502/' + 'normal_1.csv')

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

def get_drifting_all():
    dataset = pd.read_csv('D:/data/csv/daecheon_beach/20190422/' + 'normal_1.csv')

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

    min_max_scale = MinMaxScaler()

    sub_daily = daily / 65
    sub_hour = (hour + 1) / 24

    sub_wind_dir = wind_dir / 360

    sub_current_dir = current_dir / 360

    x_scaled = min_max_scale.fit_transform(wind_speed)
    sub_wind_speed = pd.DataFrame(x_scaled, columns=['wind_speed'])

    x_scaled = min_max_scale.fit_transform(current_speed)
    sub_current_speed = pd.DataFrame(x_scaled, columns=['current_speed'])

    x_scaled = min_max_scale.fit_transform(wave_height)
    sub_wave_height = pd.DataFrame(x_scaled, columns=['wave_height'])

    x_scaled = min_max_scale.fit_transform(water_temp)
    sub_water_temp = pd.DataFrame(x_scaled, columns=['water_temp'])

    x_scaled = min_max_scale.fit_transform(danger_depth)
    sub_danger_depth = pd.DataFrame(x_scaled, columns=['danger_depth'])

    x_scaled = min_max_scale.fit_transform(tide_variation)
    sub_tide_variation = pd.DataFrame(x_scaled, columns=['tide_variation'])

    temp_dataset = pd.concat([daily, hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation], axis=1)

    return temp_dataset, np.array(drowning)


features_data, classify = get_drowning_all()
# features_data, classify = get_drowning_time()

eps = 0.5
min_sample = 5

epss = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for ep in epss:
    for ms in range(1, 11, 1):
        dbscan = DBSCAN(eps=ep, min_samples=ms) #기본값이다.
        cluster = dbscan.fit_predict(features_data)
        features_data["cluster"] = cluster

        features_data.to_csv(load_path+save_path+"dbscan_"+str(ep)+"_"+str(ms)+"_drowning.csv")

# path = 'D:/data/csv/daecheon_beach/20190502/'
# save_path = 'D:/data/csv/daecheon_beach/20190502/save/'
#
# load_file = 'normal.csv'
#
# dataset = pd.read_csv(path + load_file)