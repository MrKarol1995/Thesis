import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Input ticker symbol
ticker = input("Enter Stock Name: ")

# Download stock data
data = yf.download(ticker, start="2009-01-01", end="2024-12-01")

# Ensure there is data to process
if data.empty:
    print("No data found for the given stock.")
else:
    # Calculate mid prices from the highest and lowest prices
    high_prices = data['High'].values
    low_prices = data['Low'].values
    mid_prices = (high_prices + low_prices) / 2

    # Split the data into training and testing sets
    train_data = mid_prices[:3500]
    test_data = mid_prices[3500:]

    # Reshape data for scaling
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    # Initialize scaler
    scaler = MinMaxScaler()

    # Perform windowed normalization on the training data
    smoothing_window_size = 250
    for di in range(0, len(train_data) - smoothing_window_size, smoothing_window_size):
        scaler.fit(train_data[di:di + smoothing_window_size, :])
        train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

    # Normalize the remaining portion of the training data
    scaler.fit(train_data[di + smoothing_window_size:, :])
    train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

    # Reshape training data back to original shape
    train_data = train_data.reshape(-1)

    # Normalize the test data using the scaler fitted on the training data
    test_data = scaler.transform(test_data).reshape(-1)

    # EMA one-step-ahead prediction
    N = train_data.size
    run_avg_predictions = []
    mse_errors = []

    # Initialize running mean
    running_mean = 0.0
    run_avg_predictions.append(running_mean)

    # Decay factor
    decay = 0.5

    # Compute running averages and MSE
    for pred_idx in range(1, N):
        running_mean = running_mean * decay + (1.0 - decay) * train_data[pred_idx - 1]
        run_avg_predictions.append(running_mean)
        mse_errors.append((run_avg_predictions[-1] - train_data[pred_idx]) ** 2)

    # Mean squared error for EMA
    mse_run_avg = 0.5 * np.mean(mse_errors)
    print(f"MSE error for EMA averaging: {mse_run_avg:.5f}")

    # Visualization of EMA predictions
    plt.figure(figsize=(15, 7))
    plt.plot(train_data, label="Smoothed Training Data")
    plt.plot(run_avg_predictions, label="EMA One-Step-Ahead Predictions")
    plt.title(f"{ticker} Training Data and EMA Predictions")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save the smoothed training data and predictions
    np.save(f"{ticker}_train_data_smoothed.npy", train_data)
    np.save(f"{ticker}_ema_predictions.npy", run_avg_predictions)
    print(f"Smoothed training data saved as {ticker}_train_data_smoothed.npy")
    print(f"EMA predictions saved as {ticker}_ema_predictions.npy")
