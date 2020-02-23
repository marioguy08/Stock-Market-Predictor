from alpha_vantage.timeseries import TimeSeries
from matplotlib.pyplot import figure
import numpy as np 
import pandas as pd
import pandas_datareader as web
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import math
import matplotlib.pyplot as plt
from datetime import date
from tkinter import *

stockTicker = 'TSLA'

today = date.today()

df = web.DataReader(stockTicker, data_source='yahoo', start='2012-01-01', end= today)

data = df.filter(['Close'])

dataset = data.values

trainingDataLength = math.ceil(len(dataset)*.8)

scaler = MinMaxScaler(feature_range = (0,1))

scaledData  = scaler.fit_transform(dataset)

trainingData = scaledData[0:trainingDataLength,:]

x_train = [] # features

y_train = [] # value

for i in range(60, len(trainingData)):

  x_train.append(trainingData[i-60:i, 0])

  y_train.append(trainingData[i, 0])
  
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))

model.add(LSTM(50, return_sequences= False))

model.add(Dense(25))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=1, epochs=1)

test_data = scaledData[trainingDataLength - 60: , :]

#Create the data sets x_test and y_test
x_test = []

y_test = dataset[trainingDataLength:, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

predictions = model.predict(x_test)

predictions = scaler.inverse_transform(predictions)

rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

train = data[:trainingDataLength]

valid = data[trainingDataLength:]

valid['Predictions'] = predictions

#Plot the data
plt.figure(figsize=(16,8))

plt.title(stock)

plt.xlabel('Date', fontsize=18)

plt.ylabel('Closing Price USD ($)', fontsize=18)

plt.plot(train['Close'])

plt.plot(valid[['Close', 'Predictions']])

plt.legend(['Training Data', 'Actual Values', 'Predictions'], loc='lower right')

plt.show()

dfNew = data[-60:].values

dfNewScaled = scaler.transform(dfNew)

X_test = []

X_test.append(dfNewScaled)

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

nextPrice = model.predict(X_test)

nextPrice = scaler.inverse_transform(nextPrice)

NextDay_Date = datetime.datetime.today() + datetime.timedelta(days=1)

NextDay_Date = NextDay_Date.strftime('%d-%m-%Y')

print("The stock price of" ,stockTicker, "should close at",round(float(nextPrice),3),"dollars on", NextDay_Date)