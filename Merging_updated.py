# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021

@author: Malek
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing 
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
import time
from datetime import datetime
label_encoder = preprocessing.LabelEncoder()
from sklearn.ensemble import RandomForestRegressor

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

df4 = pd.read_csv('C:/Users/Malek/Desktop/Vessel prediction/Experiments/Data/stop_times.txt', error_bad_lines=False, low_memory=False)
df3 = pd.read_csv('C:/Users/Malek/Desktop/Vessel prediction/Experiments/Data/trips.txt', error_bad_lines=False, low_memory=False)
df2 = pd.read_csv('C:/Users/Malek/Desktop/Vessel prediction/Experiments/Data/routes.txt', error_bad_lines=False, low_memory=False)
df1 = pd.read_excel('C:/Users/Malek/Desktop/Vessel prediction/Experiments/Data/data-updated.xlsx',sheet_name=0)

#df5 = pd.read_excel('C:/Users/Malek/Desktop/Vessel prediction/Experiments/Weather.xlsx',sheet_name=0)

df3['trip_id'] = df3['trip_id'].astype(str)
data = pd.merge(df1, df3, left_on=['Vehicle Route ID'],  right_on=['route_id'])
#data = pd.concat([df1.set_index('Vehicle Route ID'),df2.set_index('route_id')], axis=1, join='inner')
#data = pd.merge(df1, df2, left_on=['Vehicle Route ID'],  right_on=['route_id']).merge(df3, left_on=['Vehicle Trip ID'], right_on=['trip_id'])
#data.to_excel("data_merged11.xlsx") 
data.drop_duplicates(['Vehicle Timestamp', 'Vehicle ID'], inplace=True)
data.to_excel("data_merged13.xlsx") 
                         

