# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021

@author: Malek
"""
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing 
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
import time
from datetime import datetime
label_encoder = preprocessing.LabelEncoder()
from sklearn.ensemble import RandomForestRegressor

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

data = pd.read_excel('data_merged13.xlsx',sheet_name=0)
Weather = pd.read_excel('Weather-info.xlsx',sheet_name=0)
data = pd.merge(data, Weather, left_on=['Start'],  right_on=['Time'])

print("aaaa")
#print(Weather)
#for col in data.columns:
#    print(col)
                        
#data = pd.merge(df2, df1, on=['timestamp', 'tripID', 'stopSequence']).merge(df3, left_on=['tripID', 'stopSequence'], right_on=['trip_id', 'stop_sequence'])
#data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%I:%M:%S ")
data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%Y%d%m")

#data['Dates'] = pd.to_datetime(data.iloc[:,7]).dt.date
#data['Dates'] = pd.to_datetime(data['Dates'])
data.to_excel("data10.xlsx")    

data['Time'] = pd.to_datetime(data.iloc[:,7]).dt.time
data['Hour'] = pd.to_datetime(data.iloc[:,7]).dt.hour
data['Minute'] = pd.to_datetime(data.iloc[:,7]).dt.minute
data['weekday'] = data.iloc[:,7].dt.dayofweek
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')

data.iloc[:,1] = label_encoder.fit_transform(data.iloc[:,1]).astype('float64')
data.iloc[:,4] = label_encoder.fit_transform(data.iloc[:,4]).astype('float64')
data.iloc[:,5] = label_encoder.fit_transform(data.iloc[:,5]).astype('float64')
data.iloc[:,6] = label_encoder.fit_transform(data.iloc[:,6]).astype('float64')
#date_str = "6:29:2006"
data.iloc[:,7] = data.iloc[:,7].astype('datetime64').values.astype(float)
data.iloc[:,7] = pd.to_numeric(pd.to_datetime(data.iloc[:,7]))
data.iloc[:,9] = pd.to_numeric(pd.to_datetime(data.iloc[:,9]))
data.iloc[:,10] = label_encoder.fit_transform(data.iloc[:,10]).astype('float64')
data.iloc[:,11] = label_encoder.fit_transform(data.iloc[:,11]).astype('float64')
data.iloc[:,12] = label_encoder.fit_transform(data.iloc[:,12]).astype('float64')
data.iloc[:,13] = label_encoder.fit_transform(data.iloc[:,13]).astype('float64')
data.iloc[:,14] = label_encoder.fit_transform(data.iloc[:,14]).astype('float64')
data.iloc[:,15] = label_encoder.fit_transform(data.iloc[:,15]).astype('float64')
data.iloc[:,17] = label_encoder.fit_transform(data.iloc[:,17]).astype('float64')

data.iloc[:,18] = label_encoder.fit_transform(data.iloc[:,18]).astype('float64')
data.iloc[:,19] = label_encoder.fit_transform(data.iloc[:,19]).astype('float64')
data.iloc[:,20] = label_encoder.fit_transform(data.iloc[:,20]).astype('float64')
#data.iloc[:,21] = label_encoder.fit_transform(data.iloc[:,21]).astype('float64')
data.iloc[:,22] = label_encoder.fit_transform(data.iloc[:,22]).astype('float64')
data.iloc[:,23] = label_encoder.fit_transform(data.iloc[:,23]).astype('float64')
data.iloc[:,24] = label_encoder.fit_transform(data.iloc[:,24]).astype('float64')
data.iloc[:,25] = label_encoder.fit_transform(data.iloc[:,25]).astype('float64')
#data.iloc[:,21] = label_encoder.fit_transform(data.iloc[:,21]).astype('float64')
feature_cols1 = [1, 3,  4, 5, 6, 7, 10, 11, 12, 13, 14, 15,  17, 20,  22, 23, 24, 25, 26, 27, 28, 29]
X = data.iloc[:, feature_cols1]
y = data.iloc[:,2]
for col in X.columns:
    print(col)
y=(y-y.mean())/y.std()
train_pct_index = int(0.8 * len(data.iloc[:, 2]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]
reg = RandomForestRegressor(max_depth=100, random_state=0)
reg.fit(X, y)
y_pred1 = reg.predict(X_test)
print("y_pred1")
print(y_pred1)
print("y_test")
print(y_test)
print('R:', r2_score(y_test, y_pred1)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print('sMAPE:', smape(np.transpose(y_pred1),  y_test)) 
importance = reg.feature_importances_
print(importance)
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))  
#plt.savefig('prediction.png')
#, low_memory=False