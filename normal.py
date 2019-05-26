import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def normalize():
    path = 'D:/workspace/python/SaftyBeach/data/20190523/dataset/'
    save_path = 'D:/workspace/python/SaftyBeach/data/20190523/dataset/'
    # file_type = 'drifting'
    file_type = 'drowning'
    # file_num = 'avg'
    file_num = 'median'
    load_file = file_num+'_'+file_type+'_data_set.csv'
    save_file = file_num+'_'+file_type+'_normal.csv'

    print("\n\n### Data Set...")
    dataset = pd.read_csv(path + load_file)
    data_columns = list(dataset.columns.values)

    # Features
    wind_dir = dataset['wind_dir']
    current_dir = dataset['current_dir']
    wind_speed = dataset['wind_speed']
    current_speed = dataset['current_speed']
    wave_height = dataset['wave_height']
    water_temp = dataset['water_temp']
    tide_height = dataset['tide_height']
    dif_tide_height = dataset['dif_tide_height']
    tide_variation = dataset['tide_variation']
    hour = dataset['hour']
    year = dataset.year
    month = dataset.month
    day = dataset.day

    # Class
    if(file_type == 'drifting'):
        classify = dataset['drifting']
        classify_num = dataset['drifting_num']
    else:
        classify = dataset['drowning']
        classify_num = dataset['drowning_num']

    ## 정규화
    min_max_scale = MinMaxScaler()

    sub_hour = (hour + 1) / 24

    sub_wind_dir = (wind_dir) / 360

    sub_current_dir = (current_dir) / 360

    X = pd.concat([wind_speed, current_speed, wave_height, water_temp, tide_height, dif_tide_height, tide_variation], axis=1)
    x_scaled = min_max_scale.fit_transform(X)
    df_normalized = pd.DataFrame(x_scaled, columns=['wind_speed','current_speed', 'wave_height', 'water_temp', 'tide_height', 'dif_tide_height', 'tide_variation'])

    sub_dataset = pd.concat([year, month, day, sub_hour,
                             sub_wind_dir, sub_current_dir, df_normalized,
                             classify, classify_num], axis=1)

    sub_dataset.to_csv(save_path + save_file, index=False)

normalize()