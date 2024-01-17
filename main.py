import streamlit as st
import datetime as dt
import os
import math
import pandas_datareader as web
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import load_model


st.title("Cryptocurrency Prediction With LSTM")

coins = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'POL', 'DOGE', 'AVAX', 'DAI',
         'MATIC', 'SHIB', 'UNI', 'TRX', 'ETC', 'WBTC', 'LEO', 'LTC', 'NEAR', 'LINK']
currency = 'USD'

# Forms For GUI
selected_coin = st.sidebar.selectbox(
    'Select your Crypto for prediction', coins)
st.write('Selected Pair: ', selected_coin+'-'+currency)
start_date = st.sidebar.date_input("Start Date", dt.date(2016, 1, 1))
end_date = dt.date.today()
st.write('Start Date: ', start_date, '  End Date', end_date)
prediction_days = st.sidebar.number_input("Prediction Days", 7)
st.write('Prediction Days: ', prediction_days)


# Load Data
data = web.DataReader(f'{selected_coin}-{currency}',
                      'yahoo', start_date, end_date)
# st.write(data.shape)

# st.subheader("Top 5 data")
# st.write(data.head())
# st.subheader("Bottom 5 data")
# st.write(data.tail())
data = data['Close']
fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(data, label='Train', linewidth=2)
ax.set_ylabel('Price USD', fontsize=14)
ax.set_xlabel('Date', fontsize=14)
ax.set_title('Weighted Price Graph', fontsize=16)
st.plotly_chart(fig, use_container_width=True)


# Split into Train and Test Data

testing_days = 60

data_train = data[:len(data)-testing_days].values.reshape(-1, 1)
data_test = data[len(data)-testing_days:].values.reshape(-1, 1)
# st.write('Train Data: ', data_train.shape, 'Test Data: ', data_test.shape)

# fig, ax = plt.subplots(1, figsize=(4, 3))
# ax.plot(data_train, label='Train', linewidth=2)
# ax.plot(data_test, label='Test', linewidth=2)
# ax.set_ylabel('Price USD', fontsize=14)
# ax.set_title('Train Test Split', fontsize=16)
# ax.legend()
# st.plotly_chart(fig, use_container_width=True)


scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(data_train)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(data_test)


def dataset_generator_lstm(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        window_size_x = dataset[i:(i + look_back), 0]
        dataX.append(window_size_x)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


X_train, Y_train = dataset_generator_lstm(scaled_train)
X_test, Y_test = dataset_generator_lstm(scaled_test)

st.write("X_train: ", X_train.shape, "Y_train: ",
         Y_train.shape, "X_test: ", X_test.shape, "Y_test", Y_test.shape)

# st.write(X_train)


# Reshape trainX and testX into 3D

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

st.write("Shape of X_train: ", X_train.shape, "Shape of X_test: ", X_test.shape)

#st.write("trainX: ", X_train)


model = Sequential()

model.add(LSTM(units=128, activation='relu', return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')


model_path = 'models/'+selected_coin + \
    '_'+str(testing_days)+'_model.hdf5'



if os.path.exists(model_path):
    model = load_model(model_path)
    st.write('Use saved data.....')
else:
    st.write('Training data started.....')
    model.fit(X_train, Y_train, batch_size=32, epochs=25, verbose=1,
              shuffle=False, validation_data=(X_test, Y_test))
    model.save(model_path)
    st.write('Training data finished.....')

# Transformation to original form and making the predictions

predicted_price_test_data = model.predict(X_test)

predicted_price_test_data = scaler_test.inverse_transform(
    predicted_price_test_data.reshape(-1, 1))

test_actual = scaler_test.inverse_transform(Y_test.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(predicted_price_test_data, label='Predicted Test')
ax.plot(test_actual, label='Actual Test')
ax.set_ylabel('Price USD', fontsize=14)
ax.set_title('Predicted Test V/S Actual Test', fontsize=16)
ax.legend()
st.plotly_chart(fig, use_container_width=True)


predicted_price_train_data = model.predict(X_train)

predicted_price_train_data = scaler_train.inverse_transform(
    predicted_price_train_data.reshape(-1, 1))

train_actual = scaler_train.inverse_transform(Y_train.reshape(-1, 1))

fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(predicted_price_train_data, label='Predicted train')
ax.plot(train_actual, label='Actual train')
ax.set_ylabel('Price USD', fontsize=14)
ax.set_title('Predicted train V/S Actual train', fontsize=16)
ax.legend()
st.plotly_chart(fig, use_container_width=True)


# RMSE - Test Data
rmse_lstm_test = math.sqrt(mean_squared_error(
    test_actual, predicted_price_test_data))


# RMSE - Train Data
rmse_lstm_train = math.sqrt(mean_squared_error(
    train_actual, predicted_price_train_data))

st.write('Test RMSE: ', rmse_lstm_test, 'Train RMSE: ', rmse_lstm_train)


X_test_last_5_days = X_test[X_test.shape[0] - prediction_days:]

predicted_future_data = []

for i in range(prediction_days):
    predicted_future_data_x_test = model.predict(
        X_test_last_5_days[i:i+1])

    predicted_future_data_x_test = scaler_test.inverse_transform(
        predicted_future_data_x_test.reshape(-1, 1))
    # print(predicted_forecast_price_test_x)
    predicted_future_data.append(
        predicted_future_data_x_test)

predicted_future_data = np.array(predicted_future_data)
predicted_future_data = predicted_future_data.flatten()
predicted_price_test_data = predicted_price_test_data.flatten()

st.subheader('Next '+str(prediction_days)+' days Predicted Data')
st.write(predicted_future_data)


predicted_test_concatenated = np.concatenate(
    (predicted_price_test_data, predicted_future_data))


fig, ax = plt.subplots(1, figsize=(4, 3))
ax.plot(predicted_test_concatenated, label='Predicted Test')
ax.plot(test_actual, label='Actual Test')
ax.set_ylabel('Price USD', fontsize=14)
ax.set_title('Predicted Test V/S Actual Test', fontsize=16)
ax.legend()
st.plotly_chart(fig, use_container_width=True)
