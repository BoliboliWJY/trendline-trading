import pandas as pd
import numpy as np

# =============================
# 1. Import Necessary Libraries
# =============================
# (Already done above)

# =============================
# 2. Prepare Sample Data
# =============================
# For demonstration, we'll create a sample DataFrame.
# In practice, you would replace this with your actual data loading mechanism.
np.random.seed(0)  # For reproducible results
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-10-01 09:30:00', periods=600, freq='T'),  # 10 hours of data
    'high': np.random.uniform(low=100, high=200, size=600),  # Random high prices between 100 and 200
    'low': np.random.uniform(low=90, high=190, size=600)      # Random low prices between 90 and 190
})

# ================================================
# 3. Sort and Set the Timestamp as DataFrame Index
# ================================================
# Ensure that 'timestamp' is of datetime type
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Sort the data by timestamp in ascending order
data = data.sort_values('timestamp').reset_index(drop=True)

# Set 'timestamp' as the DataFrame index
data.set_index('timestamp', inplace=True)

# ================================================
# 4. Define Aggregation Thresholds and Frequencies
# ================================================
# Define thresholds as (number of points, resample frequency)
thresholds = [
    (100, '1T'),   # 1-minute intervals for first 100 points (0-100 minutes)
    (200, '3T'),   # 3-minute intervals for next 200 points (101-300 minutes)
    (200, '5T'),   # 5-minute intervals for next 200 points (301-500 minutes)
    (None, '15T')  # 15-minute intervals for the remaining data (501+ minutes)
]

# ================================================
# 5. Iteratively Aggregate Data Based on Thresholds
# ================================================
aggregated_data = []  # List to hold aggregated DataFrames
start = 0     # Starting index

for count, freq in thresholds:
    if count is not None:
        end = start + count
        subset = data.iloc[start:end]
    else:
        subset = data.iloc[start:]
    
    if not subset.empty:
        # Perform aggregation:
        # - 'high': take the maximum value in the interval
        # - 'low': take the minimum value in the interval
        agg_subset = subset.resample(freq).agg({
            'high': 'max',
            'low': 'min'
        })

        # Drop any rows with NaN values that may result from resampling
        agg_subset.dropna(inplace=True)

        aggregated_data.append(agg_subset)
    
    if count is not None:
        start += count  # Update the starting index for the next slice

# ================================================
# 6. Combine Aggregated Data into a Single DataFrame
# ================================================
compressed_data = pd.concat(aggregated_data)

# Optionally, sort the compressed data by timestamp
compressed_data = compressed_data.sort_index()

# Reset index if you prefer 'timestamp' as a column
compressed_data = compressed_data.reset_index()

# ================================================
# 7. Display the Compressed Data
# ================================================
print(compressed_data.head(10))  # Display first 10 rows
print(compressed_data.tail(10))  # Display last 10 rows

# ================================================
# 8. (Optional) Save Compressed Data to a File
# ================================================
# Uncomment the following lines to save the compressed data to a CSV file.
# compressed_data.to_csv('compressed_market_data.csv', index=False)
