# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 20:10:45 2021

@author: Malek
"""
from statistics import mean 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing 
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
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

print("aaaa")
#print(Weather)
for col in Weather.columns:
    print(col)
                        
#data = pd.merge(df2, df1, on=['timestamp', 'tripID', 'stopSequence']).merge(df3, left_on=['tripID', 'stopSequence'], right_on=['trip_id', 'stop_sequence'])
#data.iloc[:,7] = pd.to_datetime(data.iloc[:,7], format="%I:%M:%S ")
data.iloc[:,8] = pd.to_datetime(data.iloc[:,8], format="%Y%d%m")

#data['Dates'] = pd.to_datetime(data.iloc[:,7]).dt.date
#data['Dates'] = pd.to_datetime(data['Dates'])
data['Time'] = pd.to_datetime(data.iloc[:,7]).dt.time
data['Hour'] = pd.to_datetime(data.iloc[:,7]).dt.hour
data['Minute'] = pd.to_datetime(data.iloc[:,7]).dt.minute
data['weekday'] = data.iloc[:,8].dt.dayofweek
#data["HR"] =  data["weekday"].astype(str)  + data["Hour"].astype(str)  + data["Minute"].astype(str) 
#data["HR"]  = data["HR"].astype(int)

#print("Drop")
#print(data["HR"])

#print(data['Dates'] == datetime(1970, 1, 1))
#data = pd.merge(data, df5, left_on=['Dates'], right_on=['Date'])


#df6.index = pd.IntervalIndex.from_arrays(df6['Init (weekday+time)'],df6['End (weekday+time)'],closed='both')
#data['weekday-time'] = data['HR'].apply(lambda x : data.iloc[data.index.get_loc(x)]['HR'])
#data = data.sort_values(by='Dates',ascending=True)
#data.to_excel("data3.xlsx") 

#data['Dates2'] = pd.to_datetime(data.iloc[:,4]).dt.date
#data = data.drop(data.index[data['Dates2'] == pd.Timestamp(1970,1,1)])


#data= pd.read_excel("data.xlsx")
#def get_dataframe(data):
#    df = data

data = pd.merge(data, Weather, left_on=['Start'],  right_on=['Time'])
data.to_excel("data3.xlsx")    
feature_cols1 = [1, 3,  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33, 34]
class_cols = [2]

#data.iloc[:,9] = data.iloc[:,9].astype(str).str.replace(',', '')
data.iloc[:,1] = label_encoder.fit_transform(data.iloc[:,1]).astype('float64')
data.iloc[:,5] = label_encoder.fit_transform(data.iloc[:,5]).astype('float64')
data.iloc[:,6] = label_encoder.fit_transform(data.iloc[:,6]).astype('float64')
#date_str = "6:29:2006"
data.iloc[:,7] = data.iloc[:,7].astype('datetime64').values.astype(float)
data.iloc[:,8] = pd.to_numeric(pd.to_datetime(data.iloc[:,8]))
data.iloc[:,11] = label_encoder.fit_transform(data.iloc[:,11]).astype('float64')
data.iloc[:,12] = label_encoder.fit_transform(data.iloc[:,12]).astype('float64')
data.iloc[:,13] = label_encoder.fit_transform(data.iloc[:,13]).astype('float64')
data.iloc[:,14] = label_encoder.fit_transform(data.iloc[:,14]).astype('float64')
data.iloc[:,15] = label_encoder.fit_transform(data.iloc[:,15]).astype('float64')
data.iloc[:,18] = label_encoder.fit_transform(data.iloc[:,18]).astype('float64')
data.iloc[:,19] = label_encoder.fit_transform(data.iloc[:,19]).astype('float64')
data.iloc[:,20] = label_encoder.fit_transform(data.iloc[:,20]).astype('float64')
data.iloc[:,21] = label_encoder.fit_transform(data.iloc[:,21]).astype('float64')
#data.iloc[:,22] = label_encoder.fit_transform(data.iloc[:,22]).astype('float64')

#data.iloc[:,7]  = time.strptime(data.iloc[1,7], "%m:%d:%Y")
#data.iloc[:,7] = time.mktime(data.iloc[:,7])
#data.iloc[:,5] = data.iloc[:,5].str.replace('-', '').astype(float)

#data.iloc[:,5].fillna('', inplace=True)

print("feature_cols1")
X = data.iloc[:, feature_cols1]
for col in X.columns:
    print(col)




y = data.iloc[:,2]
y=(y-y.mean())/y.std()

train_pct_index = int(0.8 * len(data.iloc[:, 2]))
X_train, X_test = X[:train_pct_index], X[train_pct_index:]
y_train, y_test = y[:train_pct_index], y[train_pct_index:]


#print('feature_cols1')
#print(X.head())

#print(y.head())
feature_cols1 = label_encoder.fit_transform(feature_cols1).astype('float64')
y_test = y_test.apply(pd.to_numeric, errors='coerce')
model = Sequential()
model.add(Dense(12, input_dim=19, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='MAE', optimizer='adam')
model.fit(X, y, epochs=100, verbose=0)
y_pred1 = model.predict(X_test)
R = cross_val_score(model, X, y, cv=5, scoring='r2')
#R2 = 1-(1-mean(R)*(len(y)-1))/(len(y)-28-1)
MAE = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
RMSE = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
MAPE = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_percentage_error')
print('R cross val:',mean(R))
#print(R2)
print('Mean Absolute Error:', mean(MAE))
print('Root Mean Squared Error:', mean(RMSE))
print('MAPE:', mean(MAPE))
print('Adjusted R:',1 - (1-model.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('R:', r2_score(y_test, y_pred1)) 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print('sMAPE:', smape(np.transpose(y_pred1),  y_test)) 
l = list(range(len(y_pred1)))
plt.figure(figsize=(16, 8))
plt.plot(l, y_pred1, 'g-', label = 'LR')
plt.plot(l, y_test, 'b-', label = 'Real')
y_pred2 = pd.DataFrame (y_pred1)
#with pd.ExcelWriter('Predicted_values.xlsx', engine='openpyxl', mode='a') as writer:
#    y_pred2.to_excel(writer,sheet_name='RF')
plt.xlabel('Sample'); plt.ylabel('Delay (in seconds)'); plt.title('Prediction of early/late arrivals of transit in the test set')
plt.legend();