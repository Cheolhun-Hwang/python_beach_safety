import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

path = 'E:/workspace/python/SaftyBeach/data/20190523/dataset/'
load_file = 'median_data_set.csv'