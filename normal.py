import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def drifting():
    path = 'C:/Users/qewqs/hooney/Data/csv/daecheon_beach/20190417/'
    save_path = 'C:/Users/qewqs/hooney/Data/csv/daecheon_beach/20190509/drifting/'

    load_file = 'only_numberic.csv'
    save_file = 'normal.csv'
    origin_file = 'origin.csv'

    print("\n\n### Data Set...")
    dataset = pd.read_csv(path + load_file)
    data_columns = list(dataset.columns.values)

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
    drifting = dataset['drifting']

    temp_dataset = pd.concat([daily, hour,
                              wind_dir, current_dir, wind_speed, current_speed,
                              wave_height, water_temp, danger_depth, tide_variation,
                              drifting], axis=1)

    temp_dataset.to_csv(save_path + origin_file)

    ## 정규화
    min_max_scale = MinMaxScaler()

    sub_daily = dataset['daily'] / 65
    sub_hour = (dataset['hour'] + 1) / 24

    sub_wind_dir = wind_dir / 360

    sub_current_dir = current_dir / 360

    X = pd.concat([wind_speed, current_speed, wave_height, water_temp, danger_depth, tide_variation], axis=1)
    x_scaled = min_max_scale.fit_transform(X)
    df_normalized = pd.DataFrame(x_scaled, columns=['wind_speed','current_speed', 'wave_height', 'water_temp','danger_depth','tide_variation'])

    # x_scaled = min_max_scale.fit_transform(wind_speed)
    # sub_wind_speed = pd.DataFrame(x_scaled, columns=['wind_speed'])
    #
    # x_scaled = min_max_scale.fit_transform(current_speed)
    # sub_current_speed = pd.DataFrame(x_scaled, columns=['current_speed'])
    #
    # x_scaled = min_max_scale.fit_transform(wave_height)
    # sub_wave_height = pd.DataFrame(x_scaled, columns=['wave_height'])
    #
    # x_scaled = min_max_scale.fit_transform(water_temp)
    # sub_water_temp = pd.DataFrame(x_scaled, columns=['water_temp'])
    #
    # x_scaled = min_max_scale.fit_transform(danger_depth)
    # sub_danger_depth = pd.DataFrame(x_scaled, columns=['danger_depth'])
    #
    # x_scaled = min_max_scale.fit_transform(tide_variation)
    # sub_tide_variation = pd.DataFrame(x_scaled, columns=['tide_variation'])

    sub_dataset = pd.concat([sub_daily, sub_hour,
                             sub_wind_dir, sub_current_dir, df_normalized,
                             drifting], axis=1)

    sub_dataset.to_csv(save_path + save_file)

def drowning():
    path = 'D:/data/csv/daecheon_beach/20190502/'
    save_path = 'D:/data/csv/daecheon_beach/20190502/'

    load_file = 'only_numberic.csv'
    save_file = 'normal.csv'
    origin_file = 'origin.csv'

    print("\n\n### Data Set...")
    dataset = pd.read_csv(path + load_file)
    data_columns = list(dataset.columns.values)

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
                              wave_height, water_temp, danger_depth, tide_variation,
                              drowning], axis=1)

    temp_dataset.to_csv(save_path + origin_file)

    ## 정규화
    min_max_scale = MinMaxScaler()

    sub_daily = dataset['daily'] / 65
    sub_hour = (dataset['hour'] + 1) / 24

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

    sub_dataset = pd.concat([sub_daily, sub_hour,
                             sub_wind_dir, sub_current_dir, sub_wind_speed, sub_current_speed,
                             sub_wave_height, sub_water_temp, sub_danger_depth, sub_tide_variation,
                             drowning], axis=1)

    sub_dataset.to_csv(save_path + save_file)


drifting()