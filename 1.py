# %%
from binance.spot import Spot as Client
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%%
from math import ceil,floor
import numpy as np

def time_number(interval):
    """ 将时间间隔单位转换为秒数 """
    try:
        time_unit = interval[-1]
        time_value = int(interval[:-1])
    except (ValueError, IndexError):
        raise ValueError("输入格式不正确，应为类似于 '5m', '1h' 等格式")

    # 根据时间单位确定每步的秒数
    if time_unit == 's':  # 秒
        step_seconds = time_value
    elif time_unit == 'm':  # 分钟
        step_seconds = time_value * 60
    elif time_unit == 'h':  # 小时
        step_seconds = time_value * 3600
    elif time_unit == 'd':  # 天
        step_seconds = time_value * 86400
    elif time_unit == 'w':  # 周
        step_seconds = time_value * 604800
    else:
        raise ValueError("不支持的时间单位")
    return step_seconds
def get_data_in_batches(client,coin_type,interval,total_length,current_time,limit):
    """获取行情数据

    Args:
        client (Spot): 库内容
        coin_type (string): 币种类型
        interval (string): 间隔
        total_length (int): 请求总数据大小
        current_time (int): 当前时间，用于向前推导出起始时间
        limit (int): 步长，单次请求跨度

    Returns:
        arrary: 获取结果数据
    """
    # time_choice = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    # multiple_factor = [100, 300, 500, 1500, 3000, 6000, 12000, 24000, 36000, 48000, 72000, 720*400]#每类存活时长
    # initial_index = time_choice.index(interval)
    
    total_batches = total_length // limit
    left_num = total_length % limit
    data = np.zeros((total_length, 12))
    for i in range(total_batches):
        start_time = current_time - time_number(interval) * (limit * (i + 1) + left_num)
        end_time = start_time + time_number(interval) * limit
        # print(start_time)
        batch_data = np.array(client.klines(symbol=coin_type, interval=interval, startTime=start_time*1000, endTime=end_time * 1000, limit=limit))
        start_idx = (total_batches - 1 - i) * limit
        end_idx = start_idx + batch_data.shape[0]
        data[start_idx:end_idx, :] = batch_data
    
    if left_num != 0:
        start_time = current_time - time_number(interval) * left_num
        end_time = start_time + time_number(interval) * left_num
        batch_data = np.array(client.klines(symbol=coin_type, interval=interval,    startTime=start_time*1000, endTime=end_time * 1000,  limit=left_num))
        start_idx = total_length - left_num
        end_idx = start_idx + left_num
        data[start_idx:end_idx, :] = batch_data
    

    type_data = np.zeros((total_length, 1))
    for i in range(1, total_length):
        if data[i, 1] <= data[i, 4]:
            type_data[i] = 1
      
    # 原始数据排列顺序为：[0]开盘时间、[1]开盘价、[2]最高价、[3]最低价、[4]收盘价(当前K线未结束的即为最新价)、[5]成交量、[6]收盘时间、[7]成交额、[8]成交笔数、[9]主动买入成交量、[10]主动买入成交额、[11]请忽略该参数
    data[:,0] = (data[:,0] + data[:,6]) / 2 #取时间中值,减小误差
    data = data[:, [0, 2, 3, 1, 4]].astype(float)#Ave_ime,High,Low
    # if(interval == '1s'):
    # data = data_compression(data)
    # data = data[:,[0,1]]
    return data, type_data.reshape(-1)
#%%
#获取历史数据
start_time = time.perf_counter()
key = 'uiY3WGKVNEaCkntmyikLCALO9O63PBAYcVDwLw0Xu66AgcrEBXab0UANMbWZOsj4'
secret = 'O7zn1HEFTr0e9msT1m52Nu6utZtIkmicRsbUtpSJSdVJrTlLs2NIVLLhiwALXKez'
client = Client(key, secret)
coin_type = "BTCUSDT"

mode = 'backtest' #'realtime' or 'backtest'
threshold = 0.0005
total_length = 10000
interval = '3m'
time_number(interval)
current_time = int(time.time())
limit = total_length if total_length < 1000 else 1000
# limit = 10


filename = f"{coin_type}_{interval}_{total_length}.npy"
typename = f"{coin_type}_{interval}_{total_length}_type.npy"

import os
if os.path.exists(filename) and os.path.exists(typename):
    # Load the data if the files exist
    data = np.load(filename)
    type_data = np.load(typename)
else:
    
    [data, type_data] = get_data_in_batches(client,coin_type,interval,total_length,current_time,limit)
    np.save(filename, data)
    np.save(typename, type_data)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"获取数据耗时: {elapsed_time:.6f} 秒")

# plt.figure()
# plt.plot(data[:,0])
# plt.show()
# plt.plot(data[:,0],data[:,1])
# plt.plot(data[:,0],data[:,2])
# plt.show()
#%%
# data = np.array([
#     [-1,3,0],
#     [0,10,2],
#     [1,5,-2],
#     [2,3,1.5],
#     [3,6,0.5],
#     [4,3,2.5],
#     [5,6,4],
#     [6,7,5],
#     [7,4,3],
# ])
# threshold = 2


# plt.plot(data[:,0],data[:,2])
# plt.show()

from sortedcontainers import SortedList
#先处理最高价

def initial_slope(data, trend, idx):
    """初始化斜率数组

    Args:
        data (np.list): 数据，0为时间，1为最高价，2为最底下
        trend (sortedlist): 保存斜率数据
        idx (int): 选择第1或2列数据
    """
    for i in range(1, len(data)):
        j = i-1 
        current_time, current_value = data[i,[0,idx]]
        prev_time, prev_value = data[j,[0,idx]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        trend[i].add((slope,j))

def initial_single_slope(data, trend, idx):
    """初始化斜率数组,但每次只初始化最后一个元素的斜率

    Args:
        data (np.list): 数据，0为时间，1为最高价，2为最底下
        trend (sortedlist): 保存斜率数据
        idx (int): 选择第1或2列数据
    """
    i = len(data)-1
    j = i - 1
    current_time, current_value = data[i,[0,idx]]
    prev_time, prev_value = data[j,[0,idx]]
    slope = (current_value - prev_value) / (current_time - prev_time)
    trend[i].add((slope,j))

def remove_elements_above_threshold(trend_high, threshold):
    for current_idx in range(len(trend_high) - 1, -1, -1):
        current_list = trend_high[current_idx]
        if any(abs(slope) >= threshold for slope, _ in current_list):
            del trend_high[current_idx]

def update_trend_high(data, trend_high, current_idx, i, current_slope, threshold):
    index = trend_high[i].bisect_left((current_slope,))
    for j in range(index - 1, -1, -1):
        current_time, current_value = data[current_idx, [0, 1]]
        prev_index = trend_high[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 1]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        del trend_high[i][j]
        # remove_elements_above_threshold(trend_high, threshold)
        trend_high[current_idx].add((slope, prev_index))
        update_trend_high(data, trend_high, current_idx, prev_index, slope, threshold)

def update_trend_low(data, trend_low, current_idx, i, current_slope, threshold):
    index = trend_low[i].bisect_right((current_slope,))
    if len(trend_low[i]) == 0:
        return
    for j in range(len(trend_low[i])-1, index-1, -1):
        current_time, current_value = data[current_idx, [0, 2]]
        prev_index = trend_low[i][j][1]
        prev_time, prev_value = data[prev_index, [0, 2]]
        slope = (current_value - prev_value) / (current_time - prev_time)
        del trend_low[i][j]
        # if abs(current_slope) >= threshold:
        #     del trend_low[current_idx][0]
        trend_low[current_idx].add((slope, prev_index))
        update_trend_low(data, trend_low, current_idx, prev_index, slope, threshold)

def new_trend(data, update_trend_high, update_trend_low, initial_single_slope, trend_high, idx_high, trend_low, idx_low):
    for i in range(1,len(data)):
        trend_high.append(SortedList(key=lambda x: x[0]))
        initial_single_slope(data[:i+1], trend_high, idx_high)
        current_slope = trend_high[i][0][0]
        update_trend_high(data,trend_high = trend_high,current_idx = i, i = i - 1, current_slope=current_slope)
    
        trend_low.append(SortedList(key=lambda x: x[0]))  
        initial_single_slope(data[:i+1], trend_low, idx_low)
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low = trend_low, current_idx = i, i = i - 1, current_slope=current_slope)

def price_visualize(data, type_data):
    """plot the price with high and low data in plt

    Args:
        data (np.array): market data
        type_data (np.array(bool)): rise or fall, based on (open - close)
    """
    x = data[:, 0]
    y1 = data[:, 1]
    y2 = data[:, 2]
    # 创建交替的 x 和 y 数据
    x_combined = np.empty((x.size + x.size,), dtype=x.dtype)
    y_combined = np.empty((y1.size + y2.size,), dtype=y1.dtype)
    x_combined[0::2] = x
    x_combined[1::2] = x
    y_combined[0::2] = y1
    y_combined[1::2] = y2
    for i in range(len(data)):
        if type_data[i]:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2], color='green')
        else:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2], color='red')
            
    x = data[:, 0]
    y1 = data[:, 3]
    y2 = data[:, 4]
    # 创建交替的 x 和 y 数据
    x_combined = np.empty((x.size + x.size,), dtype=x.dtype)
    y_combined = np.empty((y1.size + y2.size,), dtype=y1.dtype)
    x_combined[0::2] = x
    x_combined[1::2] = x
    y_combined[0::2] = y1
    y_combined[1::2] = y2
    for i in range(len(data)):
        if type_data[i]:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2],linewidth=3, color='green')
        else:
            plt.plot(x_combined[2*i:2*i+2], y_combined[2*i:2*i+2],linewidth=3, color='red')

def trend_visualize(threshold, data, type_data, trend_high, trend_low):
    """visualize the trend data

    Args:
        threshold (int): the maxnium number of the slope
        data (array): market data
        type_data (array): rise or fall type
        trend_high (array): slope data for rise
        trend_low (array): slope data for fall
    """
    for i in range(1, len(data)-1):
        if type_data[i] != type_data[i - 1] or type_data[i] != type_data[i + 1]:
            for slope, j in trend_high[i]:
                if type_data[j] != type_data[j-1] or type_data[j] != type_data[j+1]:
                    if abs(slope) > threshold or type_data[j] == 0:
                        continue
                    start_point = data[j, [0, 1]]
                    end_point = data[i, [0, 1]]
                    m = slope
                    plt.axline(end_point, slope=m, color='red', linewidth=0.1, label=f'Slope = {m}  ')
            for slope, j in trend_low[i]:
                if  type_data[j] != type_data[j-1] or type_data[j] != type_data[j+1]:
                    if abs(slope) > threshold or type_data[j] == 1:
                        continue
                    point = data[i, [0, 2]]
                    m = slope
                    plt.axline(point, slope=m, color='green', linewidth=0.1, label=f'Slope = {m}')

# loop_times = []#计算耗时
#%%
# mean = sum(data[:, 1] - data[:, 2]) / len(data)
def calculate_trend(threshold, data, update_trend_high, update_trend_low, trend_high, trend_low, start_idx):
    """calculate rise/fall trend, it shall calculate the length of the data

    Args:
        threshold (double): maxinum number of slope
        data (array): market data
        update_trend_high (func): updating rise trend
        update_trend_low (func): updating fall trend
        trend_high (array): rise trend data
        trend_low (array): fall trend data
        start_idx (int): the beginning index for calculating, for realtime, it should be 2
    """
    for i in range(start_idx,len(data)):
        # start_time = time.perf_counter()
    #process high_time
        current_slope = trend_high[i][0][0]
        update_trend_high(data, trend_high=trend_high, current_idx=i, i=i - 1, current_slope=current_slope, threshold=threshold)
    #process low_time
        current_slope = trend_low[i][0][0]
        update_trend_low(data, trend_low=trend_low, current_idx=i, i=i - 1, current_slope=current_slope, threshold=threshold)
    
    # end_time = time.perf_counter()  # 记录结束时间
    # elapsed_time = end_time - start_time  # 计算耗时
    # loop_times.append(elapsed_time)  # 将耗时添加到列表中
    # 可选：打印每次循环的耗时
    # print(f"循环索引 {i} 耗时: {elapsed_time:.6f} 秒")
    # print("calculating completed")
   
"""cumulative_times = np.cumsum(loop_times)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    # 子图1：每次循环的耗时
    ax1.plot(loop_times, color='blue')
    ax1.grid(True)
    # 子图2：累计耗时
    ax2.plot(cumulative_times, color='green')
    ax2.grid(True)
    # 自动调整子图参数，防止标签重叠
    plt.tight_layout()
    # 显示图形
    plt.show()"""

#%%
import copy
import cProfile
import pstats
import io
mode = 'realtime'
mode = 'backtest'
def profile_method(func):
            def wrapper(*args, **kwargs):
                profiler = cProfile.Profile()
                profiler.enable()

                result = func(*args, **kwargs)

                profiler.disable()
                s = io.StringIO()
                sortby = 'tottime'
                ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
                ps.print_stats(10)
                print(s.getvalue())

                return result
            return wrapper
# profiler = cProfile.Profile()#分析运行时间
# profiler.enable()
if mode == 'realtime':
    # plt.ion()
    plt.figure()
    # plt.plot(data[:,0],data[:,1],marker = '.',markersize = 3)
    # plt.plot(data[:,0],data[:,2],marker = '.',markersize = 3)
    price_visualize(data, type_data)
    # plt.plot(x_combined, y_combined, marker='.', markersize=3)
    
    #initialize trend_high
    trend_high = [SortedList(key=lambda x: x[0]) for _ in range(len(data))]
    idx_high = 1
    initial_slope(data, trend_high, idx_high)
    #initialize trend_low
    trend_low = [SortedList(key=lambda x: x[0]) for _ in range(len(data))]
    idx_low = 2
    initial_slope(data, trend_low,idx_low)
    
    start_time = time.perf_counter()
    calculate_trend(threshold, data, update_trend_high, update_trend_low, trend_high, trend_low, 2)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"实测耗时: {elapsed_time:.6f} 秒")
    print("calculating completed")
    
    trend_visualize(threshold, data, type_data, trend_high, trend_low)
        # plt.draw()
        # plt.pause(1)  # 暂停以便观察更新，时间可以根据需要调整

    # plt.ioff()
    visual_number = total_length if total_length < 500 else 500 #单图内可视化数量
    plt.xlim(current_time * 1000 - visual_number * time_number(interval) * 1000, current_time * 1000 + time_number(interval) * 1000)
    plt.ylim(min(data[-visual_number:,2])* 0.9995, max(data[-visual_number:,1])* 1.0005)
    plt.show()
       
elif mode == 'backtest':
    
    
    
    def filter_trend(trend_high, trend_low, data, threshold=0.00003):
        # filtered_trend_high = [[] for _ in range(len(trend_high))]
        # for i in range(1, len(trend_high)):
        #     for slope, j in trend_high[i]:
        #         if np.abs(slope) <= threshold:
        #             filtered_trend_high[i].append([slope, j])
        # filtered_trend_high = np.array(filtered_trend_high, dtype=object)
        
        # filtered_trend_low = [[] for _ in range(len(trend_low))]
        # for i in range(1, len(trend_low)):
        #     for slope, j in trend_low[i]:
        #         if np.abs(slope) <= threshold:
        #             filtered_trend_low[i].append([slope, j])
        # filtered_trend_low = np.array(filtered_trend_low, dtype=object)
        def _filter_single_trend(trend):
            """
            Filters a single trend based on the slope threshold.
            
            Parameters:
            - trend: List[Tuple[float, int]] - A list of (slope, j) tuples.
            
            Returns:
            - filtered_trend: List[List[Tuple[float, int]]] - Filtered trend.
            """
            # Using list comprehension for faster filtering
            return [
                [(slope, j) for slope, j in sub_trend if abs(slope) <= threshold]
                for sub_trend in trend
            ]
        
        # Filter trend_high and trend_low
        filtered_trend_high = _filter_single_trend(trend_high)
        filtered_trend_low = _filter_single_trend(trend_low)
        
        # Convert to NumPy arrays with dtype=object
        filtered_trend_high = np.array(filtered_trend_high, dtype=object)
        filtered_trend_low = np.array(filtered_trend_low, dtype=object)
        
        #暂时先不过滤
        # filtered_trend_high = trend_high
        # filtered_trend_low = trend_low
        return filtered_trend_high, filtered_trend_low

    def backtest_calculate_trend_generator(threshold, data, initial_single_slope, update_trend_high, update_trend_low, calculate_trend):
        idx_high = 1
        idx_low = 2
        trend_high = [SortedList(key=lambda x: x[0])]
        trend_low = [SortedList(key=lambda x: x[0])]
        for i in range(1,len(data)):
            trend_high.append(SortedList(key=lambda x: x[0]))
            trend_low.append(SortedList(key=lambda x: x[0]))
            
            backtest_data = data[:i+1]
            
            initial_single_slope(backtest_data, trend=trend_high, idx=idx_high)
            initial_single_slope(backtest_data, trend=trend_low, idx=idx_low)
            if (i >= 2):
                calculate_trend(threshold=threshold, data=backtest_data,    update_trend_high=update_trend_high, update_trend_low=update_trend_low,    trend_high=trend_high, trend_low=trend_low,start_idx=i)
            #for deepcopy:
            # trend_high.append(new_trend_high)
            # trend_low.append(new_trend_low)
            # #for shallow copy:
            # filtered_trend_high = trend_high.copy()
            # filtered_trend_low = trend_low.copy()
            # filtered_trend_high, filtered_trend_low = filter_trend(filtered_trend_high, filtered_trend_low, data)
                
            yield trend_high, trend_low
            
    import sys
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets
    import numpy as np
    from collections import OrderedDict
        
    class PlotWindow(QtWidgets.QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.plotter = None  # Will be set by Plotter
            self.setWindowTitle("Data Plotter with FPS")
            self.resize(1200, 800)
            self.setStyleSheet("""
                QWidget {
                    background-color: black;
                }
                QPushButton {
                    background-color: #333333;
                    color: white;
                    border: 2px solid #555555;
                    border-radius: 5px;
                    padding: 5px 10px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QPushButton:pressed {
                    background-color: #777777;
                }
            """)
            # Create a vertical layout
            layout = QtWidgets.QVBoxLayout()
            self.setLayout(layout)

            # Create the pyqtgraph PlotWidget
            self.plot_widget = pg.PlotWidget()
            layout.addWidget(self.plot_widget)
            
            #hoirzontal layout for buttons
            button_layout = QtWidgets.QHBoxLayout()
            layout.addLayout(button_layout)
            
            
            self.prev_button = QtWidgets.QPushButton("Previous")
            button_layout.addWidget(self.prev_button)
            self.prev_button.clicked.connect(self.on_prev_button)
            
            self.next_button = QtWidgets.QPushButton("Next")
            button_layout.addWidget(self.next_button)
            self.next_button.clicked.connect(self.on_next_button)

            # Add a Pause/Resume button
            self.pause_button = QtWidgets.QPushButton("Pause")
            layout.addWidget(self.pause_button)
            # Connect the button to the toggle_pause method
            self.pause_button.clicked.connect(self.on_pause_button)

        def on_prev_button(self):
            if self.plotter:
                self.plotter.show_previous_frame()             
        def on_next_button(self):
            if self.plotter:
                self.plotter.show_next_frame()         
        def on_pause_button(self):
            if self.plotter:
                self.plotter.toggle_pause()
        def keyPressEvent(self, event):
            if event.key() == QtCore.Qt.Key_Space:
                if self.plotter:
                    self.plotter.toggle_pause()
            elif event.key() == QtCore.Qt.Key_Left:
                if self.plotter:
                    self.plotter.show_previous_frame()
            elif event.key() == QtCore.Qt.Key_Right:
                if self.plotter:
                    self.plotter.show_next_frame()
            else:
                super().keyPressEvent(event)
    
    # @profile_method
    class Plotter:
        def __init__(self, data, type_data, trend_generator, filter_trend, base_trend_number = 1000, visual_number = 100, update_interval = 200, cache_size = 100):
            self.data = data
            self.type_data = type_data
            self.trend_generator = trend_generator
            self.filter_trend = filter_trend
            self.base_trend_number = base_trend_number
            self.visual_number = visual_number
            self.update_interval = update_interval
            self.frame_count = 0
            self.is_paused = True  # State to track pause/resume
            self.plot_cache = OrderedDict()
            self.cache_size = cache_size
            
            #FPS
            self.frame_count_fps = 0
            self.last_time_fps = time.time()
            self.fps = 0
            #application and window
            self.app = QtWidgets.QApplication(sys.argv)
            self.win = PlotWindow()
            self.win.plotter = self  # Reference for key events
            
            self.plot = self.win.plot_widget.plotItem
            self.set_plot_ranges(self.data[self.base_trend_number: self.base_trend_number + self.visual_number])
            self.plot.getViewBox().setBackgroundColor('k')
            
            self.plot_lines = {}
            self.initialize_plot_lines()
            
            self.current_data = self.data[: self.base_trend_number + self.visual_number]
            self.current_type = self.type_data[: self.base_trend_number + self.visual_number]
            self.update_plot_initial()
            
            #timer
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_plot)
            self.timer.start(self.update_interval)
            
            #fps
            self.fps_timer = QtCore.QTimer()
            self.fps_timer.timeout.connect(self.update_fps)
            self.fps_timer.start(1000)

            self.win.closeEvent = self.on_close
         
        def initialize_plot_lines(self, ):
            """
            Initializes all plot lines based on the plot configurations.
            """
            linewidth = 300 / self.visual_number
            self.plot_configs = [
                {
                    'name':'high_low_green',
                    'color':'green',
                    'width':linewidth,
                    'columns':[0,2,0,1],  # [x, y_high, x, y_high]
                    'connect':'pairs',
                    'condition':lambda type_val:type_val == 0
                },
                {
                    'name': 'high_low_red',
                    'color': 'red',
                    'width': linewidth,
                    'columns': [0,2,0,1],  # [x, y_high, x, y_high]
                    'connect': 'pairs',
                    'condition': lambda type_val:type_val == 1
                },
                {
                     'name': 'open_close_green',
                     'color': 'green',
                     'width': 2 * linewidth,
                     'columns': [0, 3, 0, 4],  # [x, y_low, x, y_low]
                     'connect': 'pairs',
                     'condition': lambda type_val: type_val == 0
                },
                {
                    'name': 'open_close_red',
                    'color': 'red',
                    'width': 2 * linewidth,
                    'columns': [0, 3, 0, 4],  # [x, y_low, x, y_low]
                    'connect': 'pairs',
                    'condition': lambda type_val: type_val == 1
                }
            ]
            
            for config in self.plot_configs:
                pen = pg.mkPen(color=config['color'], width=config['width'])
                self.plot_lines[config['name']] = self.plot.plot([], [], pen=pen, connect=config['connect'])
                
            # Initialize Trend Lines
            trend_pen_high = pg.mkPen(color='green', width=0.5, style=QtCore.Qt.SolidLine)
            trend_pen_low = pg.mkPen(color='red', width=0.5, style=QtCore.Qt.SolidLine)
            self.plot_lines['trend_high'] = self.plot.plot([], [], pen=trend_pen_high, name='Trend High')
            self.plot_lines['trend_low'] = self.plot.plot([], [], pen=trend_pen_low, name='Trend Low')
                
        def set_plot_ranges(self, data_slice):
            """
            Sets the X and Y ranges of the plot based on the provided data slice.
            """
            x_interval = data_slice[1, 0] - data_slice[0, 0]
            x_min = data_slice[0, 0]
            x_max = data_slice[-1, 0]
            y_min = np.min(data_slice[:, 1:3])  # Considering y_high and y_low
            y_max = np.max(data_slice[:, 1:3])
            self.plot.setXRange(x_min, x_max + self.visual_number * 0.05 * x_interval)
            self.plot.setYRange(y_min, y_max)
        
        
        def update_plot_line(self, plot_name, x_data, y_data):
            """
            Updates a specific plot line with new X and Y data.
            """
            self.plot_lines[plot_name].setData(x_data, y_data)
            
        def extract_cache_data(self):
            """
            Extracts and returns the current data and processed pairs for caching.
            Returns a dictionary keyed by plot line names.
            """
            cache_data = {'current_data': self.current_data,
                          'current_type': self.current_type,}
            for config in self.plot_configs:
                if config['name'] in ['trend_high', 'trend_low']:
                    continue
                condition = config['condition']
                indices = condition(self.current_type)
                filtered_data = self.current_data[indices]
                pairs = filtered_data[:, config['columns']].reshape(-1,4)
                cache_data[config['name']] = pairs
            return cache_data
        
        def trend_to_line(self, data, trend_high, trend_low):
            # x_high, y_high, x_low, y_low = [], [], [], []

            # for i in range(1, len(trend_high)):
            #     delta = data[max(i, len(trend_high)),0] - data[max(0, i - len(trend_high)),0]
            #     for slope, j in trend_high[i]:
            #         if 0 <= j < len(data) and 0 <= i < len(data):
            #             x_high.extend([data[j, 0] - delta, data[i, 0] + delta, np.nan])
            #             y_high.extend([data[j, 1] - delta * slope, data[i, 1] + delta * slope, np.nan])
                    
            # for i in range(1, len(trend_low)):
            #     delta = data[max(i, len(trend_low)),0] - data[max(0, i - len(trend_low)),0]
            #     for slope, j in trend_low[i]:
            #         if 0 <= j < len(data) and 0 <= i < len(data):
            #             x_low.extend([data[j, 0] - delta, data[i, 0] + delta, np.nan])
            #             y_low.extend([data[j, 2] - delta * slope, data[i, 2] + delta * slope, np.nan])
            # return np.array(x_high), np.array(y_high), np.array(x_low), np.array(y_low)
            # Precompute delta_high and delta_low
            N_high = len(trend_high)
            if N_high > 0 and N_high < len(data):
                delta_high = data[N_high, 0] - data[0, 0]
            else:
                delta_high = 0  # Handle edge cases appropriately

            N_low = len(trend_low)
            if N_low > 0 and N_low < len(data):
                delta_low = data[N_low, 0] - data[0, 0]
            else:
                delta_low = 0  # Handle edge cases appropriately

            # Process trend_high
            slopes_high = []
            js_high = []
            is_high = []

            for i in range(1, N_high):
                for slope, j in trend_high[i]:
                    if 0 <= j < len(data):
                        slopes_high.append(slope)
                        js_high.append(j)
                        is_high.append(i)

            slopes_high = np.array(slopes_high)
            js_high = np.array(js_high)
            is_high = np.array(is_high)

            # Compute x_high and y_high using vectorized operations
            num_high = len(slopes_high)
            x_high = np.empty(num_high * 3)
            y_high = np.empty(num_high * 3)

            x_high[0::3] = data[js_high, 0] - delta_high
            x_high[1::3] = data[is_high, 0] + delta_high
            x_high[2::3] = np.nan  # Separator

            y_high[0::3] = data[js_high, 1] - delta_high * slopes_high
            y_high[1::3] = data[is_high, 1] + delta_high * slopes_high
            y_high[2::3] = np.nan  # Separator

            # Process trend_low
            slopes_low = []
            js_low = []
            is_low = []

            for i in range(1, N_low):
                for slope, j in trend_low[i]:
                    if 0 <= j < len(data):
                        slopes_low.append(slope)
                        js_low.append(j)
                        is_low.append(i)

            slopes_low = np.array(slopes_low)
            js_low = np.array(js_low)
            is_low = np.array(is_low)

            # Compute x_low and y_low using vectorized operations
            num_low = len(slopes_low)
            x_low = np.empty(num_low * 3)
            y_low = np.empty(num_low * 3)

            x_low[0::3] = data[js_low, 0] - delta_low
            x_low[1::3] = data[is_low, 0] + delta_low
            x_low[2::3] = np.nan  # Separator

            y_low[0::3] = data[js_low, 2] - delta_low * slopes_low
            y_low[1::3] = data[is_low, 2] + delta_low * slopes_low
            y_low[2::3] = np.nan  # Separator

            return x_high, y_high, x_low, y_low
                    
        
        def update_plot_initial(self):
            """
            Initializes the plot with the first set of data and caches it.
            """
            try:
                for _ in range(len(self.current_data) - 1):
                    trend_high, trend_low = next(self.trend_generator)
            except StopIteration:
                trend_high, trend_low = [], []
                
            trend_high, trend_low =self.filter_trend(trend_high, trend_low, data)#filter the trend in need    
                
            x_high, y_high, x_low, y_low = self.trend_to_line(self.data, trend_high, trend_low)
            self.update_plot_line('trend_high', x_high.flatten(), y_high.flatten())
            self.update_plot_line('trend_low', x_low.flatten(), y_low.flatten())
                
            for config in self.plot_configs:
                condition = config['condition']
                indices = condition(self.current_type)
                filtered_data = self.current_data[indices]
                
                pairs = filtered_data[:, config['columns']].reshape(-1,4)
                x_data = pairs[:, [0, 2]].flatten()
                y_data = pairs[:, [1, 3]].flatten()
                self.update_plot_line(config['name'], x_data, y_data)
            
            cache_data = self.extract_cache_data()
            cache_data['trend_high'] = [x_high, y_high]
            cache_data['trend_low'] = [x_low, y_low]
            self.plot_cache[0] = cache_data
            
        def organize_data(self):
            """a merged update line data for manual and auto operation
            """
            self.current_data = self.data[self.base_trend_number + self.frame_count: self.base_trend_number + self.visual_number + self.frame_count]
            self.current_type = self.type_data[self.base_trend_number + self.frame_count: self.base_trend_number + self.visual_number + self.frame_count]
            try:
                trend_high, trend_low = next(self.trend_generator)
            except StopIteration:
                trend_high, trend_low = [], []
                
            trend_high, trend_low =self.filter_trend(trend_high, trend_low, data)#filter the trend in need  
            
            x_high, y_high, x_low, y_low = self.trend_to_line(self.data, trend_high, trend_low)
            regular_cache = self.extract_cache_data()
            regular_cache['trend_high'] = [x_high, y_high]
            regular_cache['trend_low'] = [x_low, y_low]
            self.plot_cache[self.frame_count] = regular_cache
            return regular_cache
        
        # @profile_method
        def update_plot(self):
            """update the later trend data
            """
            if self.is_paused:
                return

            
            self.frame_count += 1
            self.frame_count_fps += 1
            # print(f"Frame: {self.frame_count}") 
            end_index = self.frame_count + self.visual_number
            if end_index > len(self.data):
                print("Reached end of data. Stopping the plot.")
                self.timer.stop()
                self.fps_timer.stop()
                return
            
            if self.frame_count in self.plot_cache:
                cached_data = self.plot_cache[self.frame_count]
                self.current_data = cached_data['current_data']
                self.current_type = cached_data['current_type']
            else:
                cached_data = self.organize_data()
            self.update_trend_lines(cached_data)
                
            if len(self.plot_cache) > self.cache_size:
                removed_key, _ = self.plot_cache.popitem(last=False)
                 
            with pg.BusyCursor():
                for config in self.plot_configs:
                    plot_name = config['name']
                    if plot_name in ['trend_high', 'trend_low']:
                        continue
                    pairs = cached_data.get(plot_name, [])
                    if isinstance(pairs, np.ndarray) and pairs.size > 0:
                        x_data = pairs[:, [0, 2]].flatten()
                        y_data = pairs[:, [1, 3]].flatten()
                        self.update_plot_line(plot_name, x_data, y_data)
                    else:
                        self.update_plot_line(plot_name, [], [])
                    
            self.set_plot_ranges(self.current_data)
        
        def update_trend_lines(self, cached_data):
            """
            Updates the trend_high and trend_low plot lines.
            """
            x_high, y_high = cached_data.get('trend_high', ([], []))
            x_low, y_low = cached_data.get('trend_low', ([], []))

            if len(x_high) > 0 and len(y_high) > 0:
                self.update_plot_line('trend_high', x_high.flatten(), y_high.flatten())
            else:
                self.update_plot_line('trend_high', [], [])

            if len(x_low) > 0 and len(y_low) > 0:
                self.update_plot_line('trend_low', x_low.flatten(), y_low.flatten())
            else:
                self.update_plot_line('trend_low', [], [])
            
        def on_close(self, event):
            self.timer.stop()
            self.fps_timer.stop()
            self.app.quit()
            event.accept()
            
        def update_fps(self):
            current_time = time.time()
            elapsed = current_time - self.last_time_fps
            if elapsed > 0:
                self.fps = self.frame_count_fps / elapsed
            else:
                self.fps = 0
            self.win.setWindowTitle(f"FPS: {self.fps:.2f}")
            self.frame_count_fps = 0
            self.last_time_fps = current_time
        
        def load_frame_from_cache(self):
            if self.frame_count in self.plot_cache:
                cached_data = self.plot_cache[self.frame_count]
            else:
                if self.frame_count + self.base_trend_number + self.visual_number <= len(self.data):
                    cached_data = self.organize_data()
                    
                if len(self.plot_cache) > self.cache_size:
                    removed_key, _ = self.plot_cache.popitem(last=False)
                    print(f"Removed frame {removed_key} from cache.")
                        
            self.current_data = cached_data['current_data']
            self.current_type = cached_data['current_type']
            self.update_trend_lines(cached_data)
                
            for config in self.plot_configs:
                plot_name = config['name']
                pairs = cached_data.get(plot_name, [])
                if isinstance(pairs, np.ndarray) and pairs.size > 0:
                    x_data = pairs[:, [0, 2]].flatten()
                    y_data = pairs[:, [1, 3]].flatten()
                    self.update_plot_line(plot_name, x_data, y_data)
                else:
                    self.update_plot_line(plot_name, [], [])
            # self.set_plot_ranges(self.current_data) #手动查看时不修改xy范围
                
        def get_min_cached_frame(self):
            if self.plot_cache:
                return min(self.plot_cache.keys())
            return 0
        
        def show_previous_frame(self):
            if self.frame_count > self.get_min_cached_frame():
                self.frame_count -= 1
                self.load_frame_from_cache()
            else:
                print("Reached the oldest cached frame. Cannot go back further.")
                
                
        def show_next_frame(self):
            end_index = self.frame_count + self.visual_number + 1
            if end_index <= len(self.data):
                self.frame_count += 1
                self.load_frame_from_cache()
            else:
                print("Reached end of data. Stopping the plot.")
                self.timer.stop()
                self.fps_timer.stop()
                
        def pause_plotting(self):
            if not self.is_paused:
                self.timer.stop()
                self.fps_timer.stop()
                self.win.pause_button.setText("Resume")
                self.is_paused = True
                print("Plotting Paused.")
        
        def resume_plotting(self):
            """
            Resumes the automatic plot updates.
            """
            if self.is_paused:
                self.timer.start(self.update_interval)
                self.fps_timer.start(1000)
                self.win.pause_button.setText("Pause")
                self.is_paused = False
                print("Plotting Resumed.")
        
        def toggle_pause(self):
            if self.is_paused:
                #resume the timer
                self.resume_plotting()
            else:
                #pause the timer
                self.pause_plotting()
                
        def run(self):
            self.win.show()
            
            sys.exit(self.app.exec_())
            
        

    trend_generator = backtest_calculate_trend_generator(threshold=threshold, data=data, initial_single_slope=initial_single_slope, update_trend_high=update_trend_high, update_trend_low=update_trend_low, calculate_trend=calculate_trend)
    Plotter_backtest = Plotter(data, type_data, trend_generator, filter_trend ,base_trend_number = int(total_length * 0.8), visual_number=200, update_interval=30, cache_size = 500)
    Plotter_backtest.run()

    
    
      
    
            
            

    
    
    

        
        


#%%