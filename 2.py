#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from time import time

def prepare_price_lines(ax, data, type_data):
    # Prepare segments for high-low bars
    x = data[:, 0]
    high = data[:, 1]
    low = data[:, 2]
    segs_high_low = [((x[i], high[i]), (x[i], low[i])) for i in range(len(data))]
    colors_high_low = ['green' if t else 'red' for t in type_data]
    lc_high_low = LineCollection(segs_high_low, colors=colors_high_low, linewidths=1)
    ax.add_collection(lc_high_low)

    # Prepare segments for open-close bars
    open_ = data[:, 3]
    close = data[:, 4]
    segs_open_close = [((x[i], open_[i]), (x[i], close[i])) for i in range(len(data))]
    colors_open_close = ['green' if t else 'red' for t in type_data]
    lc_open_close = LineCollection(segs_open_close, colors=colors_open_close, linewidths=3)
    ax.add_collection(lc_open_close)
    return lc_high_low, lc_open_close

def update_price_lines(lc_high_low, lc_open_close, data, type_data):
    x = data[:, 0]
    high = data[:, 1]
    low = data[:, 2]
    segs_high_low = [((x[i], high[i]), (x[i], low[i])) for i in range(len(data))]
    colors_high_low = ['green' if t else 'red' for t in type_data]

    open_ = data[:, 3]
    close = data[:, 4]
    segs_open_close = [((x[i], open_[i]), (x[i], close[i])) for i in range(len(data))]
    colors_open_close = ['green' if t else 'red' for t in type_data]

    lc_high_low.set_segments(segs_high_low)
    lc_high_low.set_color(colors_high_low)
    lc_open_close.set_segments(segs_open_close)
    lc_open_close.set_color(colors_open_close)

def draw_trends(ax, threshold, data, type_data, trend_high, trend_low):
    # In a real scenario, you would draw trend lines once and potentially update them,
    # but here we'll just show how you might do it statically.
    for i in range(1, len(data)-1):
        for slope, j in trend_high[i]:
            if type_data[j] != type_data[j-1] or type_data[j] != type_data[j+1]:
                if abs(slope) > threshold or type_data[j] == 0:
                    continue
                end_point = data[i, [0, 1]]
                ax.axline(end_point, slope=slope, color='red', linewidth=0.1)
        for slope, j in trend_low[i]:
            if type_data[j] != type_data[j-1] or type_data[j] != type_data[j+1]:
                if abs(slope) > threshold or type_data[j] == 1:
                    continue
                point = data[i, [0, 2]]
                ax.axline(point, slope=slope, color='green', linewidth=0.1)
from math import floor
def animate(data, type_data, trend_high, trend_low, threshold=10):
    fig, ax = plt.subplots()
    # Initialize line collections with the first data point
    lc_high_low, lc_open_close = prepare_price_lines(ax, data[:1], type_data[:1])

    plt.ion()
    plt.show()

    visual_number = 100000  # number of candles to show
    for i in range(10000, len(data)):
        backtest_data = data[:i+1]
        current_type_data = type_data[:i+1]

        # Update the existing line segments rather than replotting
        update_price_lines(lc_high_low, lc_open_close, backtest_data, current_type_data)

        # Optionally, you can (re)draw trend lines. But repeatedly doing this can be slow.
        # As a demonstration, let's just draw them once after we have enough data.
        if i == len(data)//2:  # e.g., draw trends halfway through
            draw_trends(ax, threshold, backtest_data, current_type_data, trend_high, trend_low)

        # Adjust the view window
        start_idx = max(0, i+1 - visual_number)
        segment_data = backtest_data[start_idx:i+1]
        ax.set_xlim(segment_data[-1,0] - 100000000, segment_data[-1,0] + 1000000)
        ax.set_ylim(min(segment_data[:,2])*0.9995, max(segment_data[:,1])*1.0005)

        plt.draw()
        plt.pause(0.001)

    plt.ioff()
    plt.show()

#%%
# --- Example Usage ---
if __name__ == "__main__":
    np.random.seed(42)
    length = 500000
    # Generate mock timestamps
    x = np.arange(length)*60000 + int(time())*1000  # pretend each candle is 1 minute apart

    # Generate synthetic OHLC data
    base_price = 100.0
    volatility = 1.0
    close = np.cumsum(np.random.randn(length)*volatility) + base_price
    open_ = close + np.random.randn(length)*0.1
    high = np.maximum(open_, close) + np.random.rand(length)*0.5
    low = np.minimum(open_, close) - np.random.rand(length)*0.5

    data = np.column_stack([x, high, low, open_, close])
    # type_data: True for rise (close > open), False for fall (close < open)
    type_data = close > open_

    # Generate fake trend data for demonstration.
    # Let's say trend_high[i] and trend_low[i] are lists of (slope, index) tuples.
    # We'll just put some random slopes for demonstration.
    trend_high = [[] for _ in range(length)]
    trend_low = [[] for _ in range(length)]
    for i in range(1, length-1):
        # Randomly assign some slopes to show as "trends"
        trend_high[i].append((0.0001*(np.random.randn()), i-1))
        trend_low[i].append((-0.0001*(np.random.randn()), i-1))

    # Run the animation
    animate(data, type_data, trend_high, trend_low, threshold=10)
