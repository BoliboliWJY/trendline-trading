
#from binance.spot import Spot as Client
from binance import Client
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#%%
import numpy as np
import os

import yaml

from src.time_number import time_number
from src.data_fetcher import get_data_in_batches
#%%
#获取历史数据

###############################################
# 全局变量及配置加载
###############################################
# 用于统计交易结果，格式为 [盈利, 亏损]
total_result = [0, 0]

# 记录获取历史数据的开始时间
start_time = time.perf_counter()

# 从配置文件中加载基本参数（包括API密钥、币种、目标时间、数据总长度、K线间隔等）
with open('config/basic_config.yaml', 'r') as file:
    basic_config = yaml.safe_load(file)

key = basic_config['key']
secret = basic_config['secret']
coin_type = basic_config['coin_type']
aim_time = basic_config['aim_time']
total_length = basic_config['total_length']
interval = basic_config['interval']
current_time = int(time.time())

# 设置回测数据存储目录
backtest_dir = os.path.join(os.getcwd(), 'backtest')
os.makedirs(backtest_dir, exist_ok=True)
directory = os.path.join(backtest_dir, coin_type)
filename = os.path.join(directory, f"{coin_type}_{interval}_{total_length}.npy")
typename = os.path.join(directory, f"{coin_type}_{interval}_{total_length}_type.npy")
os.makedirs(directory, exist_ok=True)

# 检查数据文件是否存在
if os.path.exists(filename) and os.path.exists(typename):
    # 如果存在，则加载数据
    data = np.load(filename)

    type_data = np.load(typename)
else:
    # 如果数据文件不存在，则获取数据
    limit = total_length if total_length < 1000 else 1000
    client = Client(key, secret)
    [data, type_data] = get_data_in_batches(client,coin_type,interval,total_length,current_time,limit)
    np.save(filename, data)
    np.save(typename, type_data)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"获取数据耗时: {elapsed_time:.6f} 秒")

#%%
from sortedcontainers import SortedList
import copy
import cProfile
import pstats
import io

from src.filter.filters import filter_trend, filter_trend_initial
from src.trading_strategy import TradingStrategy
from src.trend_process import calculate_trend, initial_single_slope
#%%
###############################################
# 辅助函数与装饰器定义
###############################################
def profile_method(func):
    #how to use: @profile_method before the function you want to profile
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

###############################################
# 趋势和交易策略的配置参数
###############################################
trend_config = {
    'delay': 4,#生成延迟
    'interval': time_number(interval) * 1000,#步长大小
     
    'filter_slope': False,#斜率大小限制
    'slope_threshold': 0.0001,#最大斜率阈值

    'filter_line_age': True,#趋势年龄限制
    'min_line_age': 40,#最小趋势年龄阈值

    'filter_distance': True,#距离限制，不能太近
    'distance_threshold': 100,#最小距离阈值
    
    'filter_trending_line': True,#处于趋势之中的线不考虑
    'filter_trending_line_number': 5,#连接当前点趋势的线数量阈值
}

trading_config = {
    'fee': 0.002,#手续费
    'stop_loss': 0.0005,#止损
    'take_profit': 0.001,#止盈
    
    'high_threshold': 0.0002,
    'low_threshold': 0.0002,
    'number': 1, #密集趋势线数量阈值
    
    'reverse_percent': 0.02, #潜在利润百分比
}

###############################################
# 回测用趋势数据生成器函数
###############################################
def backtest_calculate_trend_generator(data, initial_single_slope, calculate_trend):
    idx_high = 1
    idx_low = 2
    trend_high = [SortedList(key=lambda x: x[0])]
    trend_low = [SortedList(key=lambda x: x[0])]
    for i in range(1,len(data)):
        trend_high.append(SortedList(key=lambda x: x[0]))
        trend_low.append(SortedList(key=lambda x: x[0]))
        
        # 当前回测数据：前 i+1 个数据
        backtest_data = data[:i+1]
        
        # 计算初始单点斜率
        initial_single_slope(backtest_data, trend=trend_high, idx=idx_high)
        initial_single_slope(backtest_data, trend=trend_low, idx=idx_low)
        
        deleted_high = set()
        deleted_low = set()
        
        if (i >= 2):
            calculate_trend(data=backtest_data, trend_high=trend_high, trend_low=trend_low, start_idx=i, deleted_high=deleted_high, deleted_low=deleted_low)
        # print(deleted_high)
        # print(deleted_low)
        yield trend_high, trend_low, deleted_high, deleted_low
        
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import datetime
import numpy as np
from collections import OrderedDict

###############################################
# PyQtGraph 可视化窗口及绘图轴定义
###############################################
class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def tickStrings(self, values, scale, spacing):
        # 将时间戳转换为日期时间字符串
        return [
            datetime.datetime.fromtimestamp(value / 1000).strftime("%Y-%m-%d %H:%M:%S")
            for value in values
        ]

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

            time_axis = TimeAxisItem(orientation='bottom')
            
            # Create the pyqtgraph PlotWidget
            self.plot_widget = pg.PlotWidget(axisItems={'bottom': time_axis})
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
#%%
###############################################
# Plotter 类：数据可视化、趋势更新、缓存管理与交易策略执行
###############################################
# @profile_method
class Plotter:
    def __init__(self, data, type_data, trend_generator, filter_trend, trend_config, base_trend_number = 1000, visual_number = 100, update_interval = 200, cache_size = 100):
        # 保存基本数据及配置
        self.data = data
        self.type_data = type_data
        self.trend_generator = trend_generator
        self.filter_trend = filter_trend
        self.trend_config = trend_config

        # 基本参数
        self.base_trend_number = base_trend_number
        self.visual_number = visual_number
        self.update_interval = update_interval
        self.frame_count = 0
        
        # 缓存管理
        self.plot_cache = OrderedDict()
        self.cache_size = cache_size
        
        # 趋势管理
        self.deleted_trends = {}
        self.last_filtered_high = []
        self.last_filtered_low = []

        # 交易记录
        self.record_trade = []
        self.result_trade = []
        # self.total_result = [0, 0]
        self.total_result = total_result
        
        # 可视化参数
        self.visulize_mode = True #是否可视化
        self.is_paused = False  # State to track pause/resume
        
        # FPS 计算  
        self.frame_count_fps = 0
        self.last_time_fps = time.time()
        self.fps = 0
        
        # 应用和窗口
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
        
        #FPS 计时器，每秒更新一次
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        
        # 窗口关闭事件
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
            
            horizontal_pen_high = pg.mkPen(color='yellow', width=0.5, style=QtCore.Qt.SolidLine)
            self.plot_lines['horizontal_high'] = self.plot.plot([], [], pen=horizontal_pen_high, name='Horizontal High')
            horizontal_pen_low = pg.mkPen(color='white', width=0.5, style=QtCore.Qt.SolidLine)
            self.plot_lines['horizontal_low'] = self.plot.plot([], [], pen=horizontal_pen_low, name='Horizontal Low')
            
            self.plot_lines['long'] = self.plot.plot(
                    [], [], 
                    pen=None, 
                    symbol='o', 
                    symbolBrush='green', 
                    symbolSize=20, 
                    name='Long'
                )
                
            self.plot_lines['short'] = self.plot.plot(
                    [], [], 
                    pen=None, 
                    symbol='o', 
                    symbolBrush='red', 
                    symbolSize=20, 
                    name='Short'
                )
            
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
    
    def trend_to_line(self, trend_high, trend_low):
            N_high = len(trend_high)
            N_low = len(trend_low)
            data = self.data
            delta = (data[1,0] - data[0,0]) * 120
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

            if len(js_high) > 0:
                slopes_high = np.array(slopes_high)
                js_high = np.array(js_high)
                is_high = np.array(is_high)


                num_high = len(slopes_high)
                x_high = np.empty(num_high * 3)
                y_high = np.empty(num_high * 3)

                x_high[0::3] = data[js_high, 0]#start
                x_high[1::3] = data[min(N_high, len(data)-1), 0] + delta #end
                x_high[2::3] = np.nan  # Separator

                y_high[0::3] = data[js_high, 1]#start
                y_high[1::3] = data[js_high, 1] + (data[min(N_high, len(data)-1), 0] - data[js_high,    0] + delta) * slopes_high#end
                y_high[2::3] = np.nan  # Separator
            else:
                x_high = []
                y_high = []

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
            
            if len(js_low) > 0:
                slopes_low = np.array(slopes_low)
                js_low = np.array(js_low)
                is_low = np.array(is_low)

                num_low = len(slopes_low)
                x_low = np.empty(num_low * 3)
                y_low = np.empty(num_low * 3)

                x_low[0::3] = data[js_low, 0]
                x_low[1::3] = data[min(N_low, len(data)-1), 0] + delta
                x_low[2::3] = np.nan  # Separator

                y_low[0::3] = data[js_low, 2]
                y_low[1::3] = data[js_low, 2] + (data[min(N_low, len(data)-1), 0] - data[js_low, 0] +   delta) * slopes_low
                y_low[2::3] = np.nan  # Separator
            else:
                x_low = []
                y_low = []

            return x_high, y_high, x_low, y_low
    
    def horizontal_line(self, trend_high, trend_low):
            N_high = len(trend_high)
            N_low = len(trend_low)
            delta = (self.data[1,0] - self.data[0,0]) * 20
            slopes_high = []
            js_high = []
            is_high = []
            
            for i in range(1, N_high):
                for slope, j in trend_high[i]:
                    if 0 <= j < len(data) and slope <= 0:
                        slopes_high.append(slope)
                        js_high.append(j)
                        is_high.append(i)
                        
            if len(js_high) > 0:
                slopes_high = np.array(slopes_high)
                js_high = np.array(js_high)
                is_high = np.array(is_high)
                
                num_high = len(slopes_high)
                x_high = np.empty(num_high * 3)
                y_high = np.empty(num_high * 3)
                
                x_high[0::3] = data[js_high, 0]
                x_high[1::3] = data[min(N_high, len(data)-1), 0] + delta
                x_high[2::3] = np.nan
                
                y_high[0::3] = data[js_high, 1]
                y_high[1::3] = data[js_high, 1]
                y_high[2::3] = np.nan
            else:
                x_high = []
                y_high = []
                
            # Process trend_low
            slopes_low = []
            js_low = []
            is_low = []
            
            for i in range(1, N_low):
                for slope, j in trend_low[i]:
                    if 0 <= j < len(data) and slope >= 0:
                        slopes_low.append(slope)
                        js_low.append(j)
                        is_low.append(i)

            if len(js_low) > 0:
                slopes_low = np.array(slopes_low)
                js_low = np.array(js_low)
                is_low = np.array(is_low)
                
                num_low = len(slopes_low)
                x_low = np.empty(num_low * 3)
                y_low = np.empty(num_low * 3)
                
                x_low[0::3] = data[js_low, 0]
                x_low[1::3] = data[min(N_low, len(data)-1), 0] + delta
                x_low[2::3] = np.nan
                
                y_low[0::3] = data[js_low, 2]
                y_low[1::3] = data[js_low, 2]
                y_low[2::3] = np.nan
            else:
                x_low = []
                y_low = []
                
            return x_high, y_high, x_low, y_low
    # @profile_method
    def update_plot_initial(self):
            """
            Initializes the plot with the first set of data and caches it.
            """
            try:
                for _ in range(len(self.current_data) - 1):
                    trend_high, trend_low, deleted_high, deleted_low = next(self.trend_generator)
            except StopIteration:
                trend_high, trend_low, deleted_high, deleted_low = [], [], [], []
            
            trend_high, trend_low = filter_trend_initial(trend_high, trend_low, self.data, self.trend_config)#filter the trend in need
            
                # Start of Selection
            self.last_filtered_high = copy.deepcopy(trend_high)
            self.last_filtered_low = copy.deepcopy(trend_low)
                
            if self.visulize_mode:
                x_high, y_high, x_low, y_low = self.trend_to_line(trend_high, trend_low)
                x_high, y_high, x_low, y_low = self.horizontal_line(trend_high, trend_low)
                
            else:
                x_high, y_high, x_low, y_low = [], [], [], []
            
            self.safe_update_plot_line('trend_high', x_high, y_high)
            self.safe_update_plot_line('trend_low', x_low, y_low)
            
            self.safe_update_plot_line('horizontal_high', x_high, y_high)
            self.safe_update_plot_line('horizontal_low', x_low, y_low)
                

            for config in self.plot_configs:
                condition = config['condition']
                indices = condition(self.current_type)
                filtered_data = self.current_data[indices]
                
                pairs = filtered_data[:, config['columns']].reshape(-1,4)
                x_data = pairs[:, [0, 2]].flatten()
                y_data = pairs[:, [1, 3]].flatten()
                self.safe_update_plot_line(config['name'], x_data, y_data)
            
            cache_data = self.extract_cache_data()
            cache_data['trend_high'] = [x_high, y_high]
            cache_data['trend_low'] = [x_low, y_low]
            self.plot_cache[0] = cache_data
    
    def safe_update_plot_line(self, plot_name, x_data, y_data):
        if len(x_data) > 0 and len(y_data) > 0:
            self.update_plot_line(plot_name, x_data, y_data)
        else:
            self.update_plot_line(plot_name, [], [])
    
    # @profile_method
    def organize_data(self):
        """a merged update line data for manual and auto operation
        """
        self.current_data = self.data[self.base_trend_number + self.frame_count: self.base_trend_number + self.visual_number + self.frame_count]
        self.current_type = self.type_data[self.base_trend_number + self.frame_count: self.base_trend_number + self.visual_number + self.frame_count]
        try:
            trend_high, trend_low, deleted_high, deleted_low = next(self.trend_generator)
        except StopIteration:
            trend_high, trend_low, deleted_high, deleted_low = [], [], [], []
            
        for i in range(len(self.last_filtered_high)):
            self.last_filtered_high[i] = [item for item in self.last_filtered_high[i] if tuple(item) not in deleted_high]
        for i in range(len(self.last_filtered_low)):
            self.last_filtered_low[i] = [item for item in self.last_filtered_low[i] if tuple(item) not in deleted_low]
        
        self.last_filtered_high.append(trend_high[-self.trend_config.get('delay', 10):][0])
        self.last_filtered_low.append(trend_low[-self.trend_config.get('delay', 10):][0])
        
        self.last_filtered_high, self.last_filtered_low =self.filter_trend(trend_high, trend_low, self.last_filtered_high,self.last_filtered_low, self.data,self.trend_config)#filter the trend in need  
        
        x_high = []
        y_high = []
        x_low = []
        y_low = []
        self.record_trade = TradingStrategy(data = self.data, trend_high = self.last_filtered_high, trend_low = self.last_filtered_low, basic_config = basic_config, trend_config = trend_config, trading_config = trading_config, record_trade = self.record_trade, result_trade = self.result_trade, total_result = self.total_result).return_data()

        if self.visulize_mode:
            x_high, y_high, x_low, y_low = self.trend_to_line(self.last_filtered_high, self.last_filtered_low)
            
        regular_cache = self.extract_cache_data()
        regular_cache['trend_high'] = [x_high, y_high]
        regular_cache['trend_low'] = [x_low, y_low]
        
        if self.visulize_mode:
            x_high, y_high, x_low, y_low = self.horizontal_line(self.last_filtered_high, self.last_filtered_low)
        regular_cache['horizontal_high'] = [x_high, y_high]
        regular_cache['horizontal_low'] = [x_low, y_low]
        
        
        long_x = []
        long_y = []
        short_x = []
        short_y = []
        for i in range(len(self.record_trade)):
            if self.record_trade[i][-1] == 'long':
                long_x.append(self.record_trade[i][0][0])
                long_y.append(self.record_trade[i][0][1])
            elif self.record_trade[i][-1] == 'short':
                short_x.append(self.record_trade[i][0][0])
                short_y.append(self.record_trade[i][0][2])
        
        self.plot_lines['long'].setData(long_x, long_y)
        self.plot_lines['short'].setData(short_x, short_y)

        
        self.plot_cache[self.frame_count] = regular_cache
        return regular_cache
    
    # @profile_method
    def update_plot(self):
        """update the later trend data
        """
        if self.is_paused:
            return
        
        end_index = self.base_trend_number + self.frame_count + self.visual_number
        if end_index >= len(self.data):
            print("Reached end of data. Stopping the plot.")
            self.timer.stop()
            self.fps_timer.stop()
            self.pause_plotting()
            
            self.win.close()
            self.app.quit()
            
            return
        
        self.frame_count += 1
        self.frame_count_fps += 1
        # print(f"Frame: {self.frame_count}") 
        
        if self.frame_count in self.plot_cache:
            cached_data = self.plot_cache[self.frame_count]
            self.current_data = cached_data['current_data']
            self.current_type = cached_data['current_type']
        else:
            cached_data = self.organize_data()
        self.update_trend_lines(cached_data)
        self.update_horizontal_lines(cached_data)
            
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
                    self.safe_update_plot_line(plot_name, x_data, y_data)
                else:
                    self.safe_update_plot_line(plot_name, [], [])
                
        self.set_plot_ranges(self.current_data)
    
    def update_trend_lines(self, cached_data):
        """
        Updates the trend_high and trend_low plot lines.
        """
        x_high, y_high = cached_data.get('trend_high', ([], []))
        x_low, y_low = cached_data.get('trend_low', ([], []))
        self.safe_update_plot_line('trend_high', x_high, y_high)
        self.safe_update_plot_line('trend_low', x_low, y_low)
    
    def update_horizontal_lines(self, cached_data):
        x_high, y_high = cached_data.get('horizontal_high', ([], []))
        x_low, y_low = cached_data.get('horizontal_low', ([], []))
        
        self.safe_update_plot_line('horizontal_high', x_high, y_high)    
        self.safe_update_plot_line('horizontal_low', x_low, y_low)
        
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
            end_index = self.base_trend_number + self.frame_count + self.visual_number
            if end_index <= len(self.data):
                cached_data = self.organize_data()
            else:
                print("Reached end of data. Stopping the plot.")
                self.pause_plotting()
                return
                
            if len(self.plot_cache) > self.cache_size:
                removed_key, _ = self.plot_cache.popitem(last=False)
                print(f"Removed frame {removed_key} from cache.")
                    
        self.current_data = cached_data['current_data']
        self.current_type = cached_data['current_type']
        self.update_trend_lines(cached_data)
        self.update_horizontal_lines(cached_data)
            
        for config in self.plot_configs:
            plot_name = config['name']
            pairs = cached_data.get(plot_name, [])
            if isinstance(pairs, np.ndarray) and pairs.size > 0:
                x_data = pairs[:, [0, 2]].flatten()
                y_data = pairs[:, [1, 3]].flatten()
                self.safe_update_plot_line(plot_name, x_data, y_data)
            else:
                self.safe_update_plot_line(plot_name, [], [])
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
        end_index = self.base_trend_number + self.frame_count + self.visual_number + 1
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
        
        self.app.exec_()
        
###############################################
# 主程序入口：目标时间索引计算、初始化及运行 Plotter
###############################################
# 将目标时间字符串转换为毫秒时间戳        
aim_time = datetime.datetime.strptime(aim_time, '%Y-%m-%d %H:%M:%S').timestamp() * 1000
# 查找第一条时间大于目标时间的数据索引
index = np.searchsorted(data[:,0], aim_time, side='left')
if index < len(data):
    print(f"First index with data[:,0] > aim_time is {index}, with timestamp {data[index, 0]}")
else:
    print("No data point found with data[:,0] > aim_time")

visual_number = basic_config['visual_number']
base_trend_number = index - visual_number
# 或者使用固定值，比如： base_trend_number = 10

# 初始化趋势数据生成器
trend_generator = backtest_calculate_trend_generator(data=data, initial_single_slope=initial_single_slope, calculate_trend=calculate_trend)

# 初始化 Plotter 类
Plotter_backtest = Plotter(data, type_data, trend_generator, filter_trend, trend_config, base_trend_number = base_trend_number, visual_number=visual_number, update_interval=20, cache_size = visual_number*2)

# 运行 Plotter
Plotter_backtest.run()


print("Plotter has ended. Continuing with post-Plotter operations...")
    
    
    
      
    
            
            

    
    
    

        
        


#%%