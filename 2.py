import time
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from sortedcontainers import SortedList  # Ensure you have sortedcontainers installed
import numpy as np  # Assuming data is a NumPy array

# Placeholder functions for your actual implementations
def initial_single_slope(backtest_data, trend, idx):
    # Implement your logic here
    pass

def update_trend_high(trend_high, ...):
    # Implement your logic here
    pass

def update_trend_low(trend_low, ...):
    # Implement your logic here
    pass

def calculate_trend(threshold, data, update_trend_high, update_trend_low, trend_high, trend_low, start_idx):
    # Implement your logic here
    pass

def backtest_calculate_trend_generator(threshold, data, initial_single_slope, update_trend_high, update_trend_low, calculate_trend):
    idx_high = 1
    idx_low = 2
    trend_high = [SortedList(key=lambda x: x[0])]
    trend_low = [SortedList(key=lambda x: x[0])]

    for i in range(1, len(data)):
        # Append new empty SortedList for current iteration
        trend_high.append(SortedList(key=lambda x: x[0]))
        trend_low.append(SortedList(key=lambda x: x[0]))
        
        backtest_data = data[:i+1]
        
        initial_single_slope(backtest_data, trend=trend_high, idx=idx_high)
        initial_single_slope(backtest_data, trend=trend_low, idx=idx_low)
        
        if i >= 2:
            calculate_trend(
                threshold=threshold,
                data=backtest_data,
                update_trend_high=update_trend_high,
                update_trend_low=update_trend_low,
                trend_high=trend_high,
                trend_low=trend_low,
                start_idx=i
            )
        
        # Yield the current trends to be used by the animation
        yield trend_high.copy(), trend_low.copy()

def prepare_price_lines(ax, data, type_data):
    time_vals = data[:, 0]
    high_price = data[:, 1]
    low_price = data[:, 2]
    open_price = data[:, 3]
    close_price = data[:, 4]
    
    segs_high_low = [((time_vals[i], high_price[i]), (time_vals[i], low_price[i])) for i in range(len(data))]
    color_price = ['green' if t else 'red' for t in type_data]
    line_high_low = LineCollection(segs_high_low, colors=color_price, linewidths=0.5)
    ax.add_collection(line_high_low)
    
    segs_open_close = [((time_vals[i], open_price[i]), (time_vals[i], close_price[i])) for i in range(len(data))]
    line_open_close = LineCollection(segs_open_close, colors=color_price, linewidths=2)
    ax.add_collection(line_open_close)
    
    return line_high_low, line_open_close

def update_price_lines(line_high_low, line_open_close, data, type_data):
    time_vals = data[:, 0]
    high_price = data[:, 1]
    low_price = data[:, 2]
    open_price = data[:, 3]
    close_price = data[:, 4]
    
    segs_high_low = [((time_vals[i], high_price[i]), (time_vals[i], low_price[i])) for i in range(len(data))]
    color_price = ['green' if t else 'red' for t in type_data]
    line_high_low.set_segments(segs_high_low)
    line_high_low.set_color(color_price)
    
    segs_open_close = [((time_vals[i], open_price[i]), (time_vals[i], close_price[i])) for i in range(len(data))]
    line_open_close.set_segments(segs_open_close)
    line_open_close.set_color(color_price)

def update_trend_lines(line_trend_high, line_trend_low, threshold, data, trend_high_current, trend_low_current, type_data):
    start_point_high = []
    end_point_high = []
    for j, (slope, idx) in enumerate(trend_high_current):
        if abs(slope) > threshold or type_data[idx] == 0:
            continue
        start_point_high.append(data[idx, [0, 1]])
        end_point_high.append(data[j, [0, 1]])
    segments_high = list(zip(start_point_high, end_point_high))
    line_trend_high.set_segments(segments_high)
    
    start_point_low = []
    end_point_low = []
    for j, (slope, idx) in enumerate(trend_low_current):
        if abs(slope) > threshold or type_data[idx] == 0:
            continue
        start_point_low.append(data[idx, [0, 2]])
        end_point_low.append(data[j, [0, 2]])
    segments_low = list(zip(start_point_low, end_point_low))
    line_trend_low.set_segments(segments_low)

def animate_backtest(threshold, data, type_data, initial_single_slope, update_trend_high, update_trend_low, calculate_trend, visual_number=100):
    fig, ax = plt.subplots()
    line_high_low, line_open_close = prepare_price_lines(ax, data[:1], type_data[:1])
    
    line_trend_high = LineCollection([], colors='green', linewidths=2)
    line_trend_low = LineCollection([], colors='red', linewidths=2)
    ax.add_collection(line_trend_high)
    ax.add_collection(line_trend_low)
    
    # Initialize axis limits
    ax.set_xlim(data[0, 0] - 1000 * visual_number, data[0, 0] + 1000 * visual_number)
    ax.set_ylim(min(data[:, 2]) * 0.9995, max(data[:, 1]) * 1.0005)
    
    # Initialize the generator
    trend_generator = backtest_calculate_trend_generator(
        threshold, data, initial_single_slope, update_trend_high, update_trend_low, calculate_trend
    )
    
    def update(frame):
        try:
            trend_high_current, trend_low_current = next(trend_generator)
        except StopIteration:
            return line_high_low, line_open_close, line_trend_high, line_trend_low
        
        current_data = data[:frame+1]
        current_type = type_data[:frame+1]
        
        update_price_lines(line_high_low, line_open_close, current_data, current_type)
        update_trend_lines(
            line_trend_high, line_trend_low, threshold,
            current_data, trend_high_current[-1], trend_low_current[-1], type_data
        )
        
        if frame > visual_number:
            ax.set_xlim(current_data[frame - visual_number, 0], current_data[frame, 0])
            ax.set_ylim(
                min(current_data[frame - visual_number : frame, 2]) * 0.9995,
                max(current_data[frame - visual_number : frame, 1]) * 1.0005
            )
        else:
            ax.set_xlim(current_data[0, 0] - 1000 * visual_number, current_data[-1, 0] + 1000 * visual_number)
            ax.set_ylim(min(current_data[:, 2]) * 0.9995, max(current_data[:, 1]) * 1.0005)
        
        return line_high_low, line_open_close, line_trend_high, line_trend_low
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(data), interval=100, blit=True, repeat=False
    )
    plt.show()

def main():
    # Example data initialization
    # Replace this with your actual data loading mechanism
    num_points = 1000
    data = np.random.rand(num_points, 5)  # [time, high, low, open, close]
    data[:, 0] = np.arange(num_points)  # Simple time sequence
    type_data = np.random.choice([0, 1], size=num_points)  # Example type_data
    
    threshold = 0.5  # Example threshold value
    
    start_time = time.perf_counter()
    
    animate_backtest(
        threshold, data, type_data,
        initial_single_slope, update_trend_high, update_trend_low,
        calculate_trend,
        visual_number=100
    )
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Backtest and animation completed in: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
