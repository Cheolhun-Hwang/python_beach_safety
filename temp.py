import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

list = [1, 5, 9, 13, 17, 21]
list1 = [2, 6, 10, 14, 18, 22]
list2 = [3, 7, 11, 15, 19, 23]
list3 = [4, 8, 12, 16, 20, 24]

total_list = [list, list1, list2, list3]

dataframe = pd.DataFrame(total_list, columns=['index', 'index2', 'index3', 'index4', 'index5', 'index6'])

print(dataframe)

print("\nMinMax : ")

min_max_scale = MinMaxScaler()
x_scaled = min_max_scale.fit_transform(dataframe)

print(x_scaled)
