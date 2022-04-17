import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model

import streamlit as st

start = '2010-01-01'
end = '2022-04-13'

st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)
df.head()

#describing data
st.subheader('Data from 2010-2022')
st.write(df.describe())

#visualizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize = (15, 6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart With 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (15, 6))
plt.plot(ma100, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart With 100 & 200 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (15, 6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#spliting data into training and testing
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_array = scaler.fit_transform(data_train)

# dividing training data in 'X' and 'Y'
x_train = []
y_train = []

for i in range(100, data_train_array.shape[0]):
    x_train.append(data_train_array[i - 100:i])
    y_train.append(data_train_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#LOADING MY MODEL
model = load_model('keras_model.h5')

past_100_days_data = data_train.tail(100)
final_df = past_100_days_data.append(data_test, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('Actual Values Vs Predicted Values')
fig2 = plt.figure(figsize=(15, 6))
plt.plot(y_test, 'b', label = 'Actual Value')
plt.plot(y_predicted, 'r', label = 'Predicted Value')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
