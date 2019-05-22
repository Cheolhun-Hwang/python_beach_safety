import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def normalize():
    path = 'E:/workspace/python/SaftyBeach/data/'
    save_path = 'E:/workspace/python/SaftyBeach/data/'

    load_file = 'origin.csv'
    save_file = 'normal.csv'

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
    danger_depth = dataset['tide_height']
    tide_variation = dataset['dif_tide_height']
    hour = dataset['hour']

    # Class
    drifting = dataset['drifting']
    drifting_num = dataset['drifting_num']
    drowning = dataset['drowning']
    drowning_num = dataset['drowning_num']

    ## 정규화
    min_max_scale = MinMaxScaler()

    sub_hour = (dataset['hour'] + 1) / 24

    sub_wind_dir = wind_dir / 360

    sub_current_dir = current_dir / 360

    X = pd.concat([wind_speed, current_speed, wave_height, water_temp, danger_depth, tide_variation], axis=1)
    x_scaled = min_max_scale.fit_transform(X)
    df_normalized = pd.DataFrame(x_scaled, columns=['wind_speed','current_speed', 'wave_height', 'water_temp','tide_height','dif_tide_height'])

    sub_dataset = pd.concat([sub_hour,
                             sub_wind_dir, sub_current_dir, df_normalized,
                             drifting, drifting_num, drowning, drowning_num], axis=1)

    sub_dataset.to_csv(save_path + save_file, index=False)

normalize()