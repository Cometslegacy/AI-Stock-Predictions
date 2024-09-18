Overview

This project involves predicting stock prices using Long Short-Term Memory (LSTM) networks. The model is built with TensorFlow and Keras to forecast future stock prices based on historical data. The data used in this example is from Apple Inc. (AAPL)
Features

    Data Acquisition: Fetches historical stock price data using the yfinance library.
    Data Processing:
        Handles missing values using forward fill.
        Normalizes closing prices with MinMaxScaler.
    Model Building:
        Utilizes LSTM layers to capture temporal dependencies in stock prices.
        Trains the model on historical data to predict future prices.
    Evaluation:
        Evaluates the model using Root Mean Squared Error (RMSE).
        Visualizes actual vs. predicted stock prices using matplotlib.

Code Breakdown

    Data Download and Processing:
        Fetches stock data from Yahoo Finance.
        Fills missing values and scales the data.

    Data Preparation:
        Creates input-output pairs for model training using a sliding window approach.

    Model Creation:
        Builds an LSTM-based neural network with two LSTM layers and one Dense layer.

    Model Training:
        Trains the model on the prepared dataset over a specified number of epochs.

    Prediction and Evaluation:
        Makes predictions on the test set.
        Computes RMSE to evaluate model performance.
        Plots actual vs. predicted prices for visualization.
