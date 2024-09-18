import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

#download stock data from specific ticker (example: Apple)
stock_data = yf.download('AAPL', start='2015-01-01', end='2024-01-01')

#check that import works
print(stock_data.head())

#Process Data
stock_data.fillna(method='ffill', inplace=True)

#normalize the "close" prices using MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
stock_data_scaled = scaler.fit_transform(stock_data[['Close']])

#plot the closing prices
#stock_data['Close'].plot(figsize=(10,6))
#plt.title('Stock Closing Price')
#plt.ylabel('Price')
#plt.show()

#split data into training sets (generally 80:20 ratio of training:testing)
train_size = int(len(stock_data_scaled) * 0.8)
train_data, test_data = stock_data_scaled[:train_size], stock_data_scaled[train_size:]

#Preparing data for model -
#Stock price forecasting requires you to create input-output pairs (X, y) 
#where X represents the past stock prices and y represents the next day’s price.
def create_dataset(data, look_back = 60):
    x, y = [],[]
    for i in range(len(data) - look_back):
        x.append(data[i:i + look_back, 0])
        y.append(data[i + look_back, 0])
    return  np.array(x), np.array(y)

#prepare the training and testing datasets
look_back =  60
x_train, y_train = create_dataset(train_data,look_back)
x_test, y_test = create_dataset(test_data, look_back)

#reshape the data for LSTM (Long Short-Term Memory) network
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
#initialize model
model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(units = 50))
model.add(Dense(1))

#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#train model
model.fit(x_train, y_train, epochs=10, batch_size=32)

predicted_stock_price = model.predict(x_test)

#inverse scale the predictions to get actual prices
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1,1))

#calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data[look_back:], predicted_stock_price))
print('Root Mean Squared Error:', rmse)

#plot actual vs predicted stock prices
# Make sure both series are the same length
if len(predicted_stock_price) != len(test_data[look_back:]):
    print("Length mismatch between predicted and actual prices.")
else:
    plt.figure(figsize=(14, 7))
    
    # Inverse transform for test_data to get actual prices
    actual_prices = scaler.inverse_transform(test_data[look_back:])
    
    # Plot actual and predicted prices
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_stock_price, label='Predicted Prices', color='red')
    
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()