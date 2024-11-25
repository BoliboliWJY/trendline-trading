import pandas as pd
import numpy as np

# Assuming 'data' is a pandas Series with your time series data
returns = data.pct_change().dropna()
window_size = 20  # Adjust as needed
rolling_volatility = returns.rolling(window=window_size).std()
threshold = rolling_volatility.mean() + 2 * rolling_volatility.std()
significant_points = data[rolling_volatility > threshold]
