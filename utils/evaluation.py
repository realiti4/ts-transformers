import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.stats import linregress

plt.rcParams["figure.figsize"] = [18, 15]

np.random.seed(1337)


def custom_metric_function(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Implement your custom metric calculation here
    # For example, you might use sklearn metrics:
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {"accuracy": accuracy, "f1_score": f1}


def simple_trend_analysis(data, window_size=5):
    # Linear regression
    x = np.arange(len(data))
    slope, _, _, _, _ = linregress(x, data)

    # neutral_zone = 0.01
    neutral_zone = 0.005

    # Moving average
    if len(data) >= window_size:
        moving_avg = np.convolve(data, np.ones(window_size), "valid") / window_size
        oldest_avg = moving_avg[0]
        recent_avg = moving_avg[-1]
        overall_avg = np.mean(data)

        print("Oldest avg: ", oldest_avg, " Recent avg: ", recent_avg, " Overall avg: ", overall_avg)

        # Determine trend based on both linear regression and moving average
        if slope > 0 and overall_avg > oldest_avg * (1 + neutral_zone):
            trend = "Up"
            action = 1
        elif slope < 0 and overall_avg < oldest_avg * (1 - neutral_zone):
            trend = "Down"
            action = -1
        else:
            trend = "Neutral"
            action = 0
    else:
        trend = "Insufficient data"

    return trend, action

def calculate_volatility_bounds(historical_data, confidence_interval=1.96):
    # Calculate daily returns
    returns = np.diff(historical_data) / historical_data[:-1]

    # Calculate volatility (standard deviation of returns)
    volatility = np.std(returns)

    # Use the last known price instead of the average
    last_price = historical_data[-1]

    # Calculate lower and upper bounds
    lower_bound = last_price - (confidence_interval * volatility * last_price)
    upper_bound = last_price + (confidence_interval * volatility * last_price)

    return volatility, lower_bound, upper_bound

def calculate_ema(data: np.ndarray, period: int = 5) -> np.ndarray:
    alpha = 2 / (period + 1)
    ema = [sum(data[:period]) / period]

    for price in data[period:]:
        ema.append((price * alpha) + (ema[-1] * (1 - alpha)))

    return np.array(ema)


def trend_analysis_with_volatility(data: np.ndarray, lower_bound: float, upper_bound: float, window_size=5):
    """
    Perform trend analysis on the given data, incorporating volatility measures.

    This function uses linear regression and moving averages to determine the trend
    of the input data. It also calculates an exponential moving average (EMA) for
    recent average calculation.

    Parameters:
    -----------
    data : np.ndarray
        The input data array for trend analysis.
    lower_bound : float
        The lower threshold for determining a downward trend.
    upper_bound : float
        The upper threshold for determining an upward trend.
    window_size : int, optional
        The size of the window for calculating moving averages (default is 5).
    """

    # Linear regression
    x = np.arange(len(data))
    slope, _, _, _, _ = linregress(x, data)

    # Moving average
    if len(data) >= window_size:
        moving_avg = np.convolve(data, np.ones(window_size), "valid") / window_size
        oldest_avg = moving_avg[0]
        recent_avg = moving_avg[-1]
        overall_avg = np.mean(data)

        ema = calculate_ema(data, period=5)
        recent_avg = ema[-1]

        print("Oldest avg: ", oldest_avg, " Recent avg: ", recent_avg, " Overall avg: ", overall_avg)

        # Determine trend based on both linear regression and moving average
        if slope > 0 and recent_avg > upper_bound:
            trend = "Up"
            action = 1
        elif slope < 0 and recent_avg < lower_bound:
            trend = "Down"
            action = -1
        else:
            trend = "Neutral"
            action = 0
    else:
        trend = "Insufficient data"

    return trend, action


def generate_timeseries(length: int = 1024, trend: float = 1.0, noise: float = 0.2) -> np.ndarray:
    """
    Generate a random time series of given length.

    Returns:
        np.ndarray: Random time series.
    """

    # Generate a small upward trend
    trend = np.linspace(0, trend, length)

    # Generate a sine wave
    frequency = 5  # Adjust frequency as needed
    sine_wave = np.sin(np.linspace(0, 2 * np.pi * frequency, length))

    # Generate random noise
    noise = np.random.normal(0, noise, length)

    # Combine the trend and the noise
    time_series = trend + sine_wave + noise

    return time_series


def test_methods(length: int = 1024) -> None:
    """
    Tests simple trend analysis function
    """
    df = pd.read_csv("datasets/merged8.csv")
    array = df["close"].to_numpy()

    i = 0

    while True:
        print(f"Step: {i}")

        idx = np.random.randint(0, len(array) - length, 1)
        idx = idx[0]

        window = array[idx : idx + length]

        trend, _ = simple_trend_analysis(window, window_size=5)
        print(trend)

        plt.plot(window)
        plt.show()


def test_methods2(context_length: int, prediction_length: int) -> None:
    """
    Tests trend analysis with volatility function
    """

    df = pd.read_csv("datasets/merged8.csv")
    array = df["close"].to_numpy()

    length = context_length + prediction_length

    i = 0

    while True:
        i = i + 1
        print(f"Step: {i}")

        idx = np.random.randint(0, len(array) - length, 1)
        idx = idx[0]

        window = array[idx : idx + length]

        forecast_index = range(context_length, length)

        context_window = window[:context_length]
        prediction_window = window[forecast_index]
        volatility, lower_bound, upper_bound = calculate_volatility_bounds(context_window)

        trend, _ = trend_analysis_with_volatility(prediction_window, lower_bound, upper_bound, window_size=5)
        print("Trend: ", trend)

        # Plotting the price
        plt.plot(context_window, label="History")
        plt.plot(forecast_index, prediction_window, label="Forecast")
        plt.axhline(y=upper_bound, color="g", linestyle="-")
        plt.axhline(y=lower_bound, color="r", linestyle="-")

        plt.show()


if __name__ == "__main__":
    context_length = 512
    prediction_length = 64

    # test_methods(prediction_length)
    test_methods2(context_length, prediction_length)

    time_series = generate_timeseries(length=prediction_length, trend=1, noise=0.2)

    plt.plot(time_series)
    plt.show()

    print("done")
