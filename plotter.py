#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
【plotter.py】

本模块封装了 Plotter 类，负责：
  - 数据帧的组织与更新
  - 趋势线、水平线和交易信号的绘制
  - 帧数据的缓存管理（支持手动切帧）
  - 自动更新与 FPS 统计
  
详细注释均已保留，便于后续维护和修改。
"""

import sys
import time
import copy
import numpy as np
import pyqtgraph as pg
from collections import OrderedDict
from pyqtgraph.Qt import QtCore, QtWidgets

# 导入自定义的交易策略和过滤函数（请确保这些模块存在）
from src.filter.filters import filter_trend_initial
from src.trading_strategy import TradingStrategy

class Plotter:
    def __init__(self, data: np.ndarray, type_data: np.ndarray, trend_generator, filter_trend, trend_config: dict,
                 base_trend_number: int = 1000, visual_number: int = 100, update_interval: int = 200, cache_size: int = 100):
        """
        初始化 Plotter 类
        
        参数:
          data: 完整的历史K线数据
          type_data: 数据类型信息（例如交易类型的标记等）
          trend_generator: 趋势数据生成器（需支持迭代器接口）
          filter_trend: 趋势过滤函数（用于过滤不需要的趋势）
          trend_config: 趋势相关配置参数
          base_trend_number: 数据显示起始索引
          visual_number: 显示窗口中数据点的数量
          update_interval: 自动更新图表的时间间隔（毫秒）
          cache_size: 帧数据缓存大小（最多缓存的帧数）
        """
        # 保存基础数据及参数
        self.data = data
        self.type_data = type_data
        self.trend_generator = trend_generator
        self.filter_trend = filter_trend
        self.trend_config = trend_config

        self.base_trend_number = base_trend_number
        self.visual_number = visual_number
        self.update_interval = update_interval
        self.frame_count = 0

        # 帧数据缓存管理
        self.plot_cache = OrderedDict()
        self.cache_size = cache_size

        # 趋势管理数据
        self.deleted_trends = {}
        self.last_filtered_high = []
        self.last_filtered_low = []

        # 交易记录及结果
        self.record_trade = []
        self.result_trade = []
        self.total_result = [0, 0]

        # 可视化与控制参数
        self.visulize_mode = True  # 是否开启可视化显示
        self.is_paused = False

        # FPS 参数
        self.frame_count_fps = 0
        self.last_time_fps = time.time()
        self.fps = 0

        # 初始化 PyQt 应用和图形窗口
        self.app = QtWidgets.QApplication(sys.argv)
        from src.plot_figure.plot_ui import PlotWindow  # 延迟导入，避免循环依赖
        self.win = PlotWindow()
        self.win.plotter = self
        self.plot = self.win.plot_widget.plotItem
        self.plot.getViewBox().setBackgroundColor('k')

        self.set_plot_ranges(self.data[self.base_trend_number: self.base_trend_number + self.visual_number])

        # 初始化绘图线配置（例如各类趋势、水平线及交易信号的绘制）
        self.initialize_plot_lines()
        self.current_data = self.data[: self.base_trend_number + self.visual_number]
        self.current_type = self.type_data[: self.base_trend_number + self.visual_number]
        self.update_plot_initial()

        # 定时器：用于自动更新图表数据
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)

        # FPS 定时器：每秒更新一次 FPS 显示
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)

        # 窗口关闭事件
        self.win.closeEvent = self.on_close

    def initialize_plot_lines(self):
        """
        根据配置初始化所有需要绘制的图形线条，如趋势线、水平线、及交易信号显示
        """
        linewidth = 300 / self.visual_number
        self.plot_configs = [
            {
                'name': 'high_low_green',
                'color': 'green',
                'width': linewidth,
                'columns': [0, 2, 0, 1],
                'connect': 'pairs',
                'condition': lambda type_val: type_val == 0
            },
            {
                'name': 'high_low_red',
                'color': 'red',
                'width': linewidth,
                'columns': [0, 2, 0, 1],
                'connect': 'pairs',
                'condition': lambda type_val: type_val == 1
            },
            {
                'name': 'open_close_green',
                'color': 'green',
                'width': 2 * linewidth,
                'columns': [0, 3, 0, 4],
                'connect': 'pairs',
                'condition': lambda type_val: type_val == 0
            },
            {
                'name': 'open_close_red',
                'color': 'red',
                'width': 2 * linewidth,
                'columns': [0, 3, 0, 4],
                'connect': 'pairs',
                'condition': lambda type_val: type_val == 1
            }
        ]
        self.plot_lines = {}
        for config in self.plot_configs:
            pen = pg.mkPen(color=config['color'], width=config['width'])
            self.plot_lines[config['name']] = self.plot.plot([], [], pen=pen, connect=config['connect'])
        
        # 趋势线及水平线
        trend_pen_high = pg.mkPen(color='green', width=0.5)
        trend_pen_low = pg.mkPen(color='red', width=0.5)
        self.plot_lines['trend_high'] = self.plot.plot([], [], pen=trend_pen_high, name='Trend High')
        self.plot_lines['trend_low'] = self.plot.plot([], [], pen=trend_pen_low, name='Trend Low')

        horizontal_pen_high = pg.mkPen(color='yellow', width=0.5)
        horizontal_pen_low = pg.mkPen(color='white', width=0.5)
        self.plot_lines['horizontal_high'] = self.plot.plot([], [], pen=horizontal_pen_high, name='Horizontal High')
        self.plot_lines['horizontal_low'] = self.plot.plot([], [], pen=horizontal_pen_low, name='Horizontal Low')
        
        # 交易信号：多头（long）和空头（short）
        self.plot_lines['long'] = self.plot.plot([], [], pen=None, symbol='o', symbolBrush='green', symbolSize=20, name='Long')
        self.plot_lines['short'] = self.plot.plot([], [], pen=None, symbol='o', symbolBrush='red', symbolSize=20, name='Short')
    
    def set_plot_ranges(self, data_slice):
        """
        设置图表的 X 与 Y 轴范围
        
        参数:
          data_slice: 当前显示数据的切片（用于计算坐标范围）
        """
        x_interval = data_slice[1, 0] - data_slice[0, 0]
        x_min = data_slice[0, 0]
        x_max = data_slice[-1, 0]
        y_min = np.min(data_slice[:, 1:3])
        y_max = np.max(data_slice[:, 1:3])
        self.plot.setXRange(x_min, x_max + self.visual_number * 0.05 * x_interval)
        self.plot.setYRange(y_min, y_max)
    
    def update_plot_line(self, plot_name, x_data, y_data):
        """
        更新指定绘图线的 X 与 Y 数据
        
        参数:
          plot_name: 绘图线名称
          x_data: X 轴数据
          y_data: Y 轴数据
        """
        self.plot_lines[plot_name].setData(x_data, y_data)
    
    def extract_cache_data(self):
        """
        提取当前帧数据与处理后的绘图数据，存入字典用于缓存
        """
        cache_data = {
            'current_data': self.current_data,
            'current_type': self.current_type,
        }
        for config in self.plot_configs:
            if config['name'] in ['trend_high', 'trend_low']:
                continue
            condition = config['condition']
            indices = condition(self.current_type)
            filtered_data = self.current_data[indices]
            pairs = filtered_data[:, config['columns']].reshape(-1, 4)
            cache_data[config['name']] = pairs
        return cache_data

    def trend_to_line(self, trend_high, trend_low):
        """
        根据趋势数据生成可绘制的趋势线数据
        
        返回:
          x_high, y_high, x_low, y_low 分别为高价和低价趋势线的 X、Y 坐标列表
        """
        # 此处代码为示例，具体实现请参照原始逻辑
        N_high = len(trend_high)
        delta = (self.data[1, 0] - self.data[0, 0]) * 120
        slopes_high = []
        js_high = []
        for i in range(1, N_high):
            for slope, j in trend_high[i]:
                if 0 <= j < len(self.data):
                    slopes_high.append(slope)
                    js_high.append(j)
        if slopes_high:
            slopes_high = np.array(slopes_high)
            js_high = np.array(js_high)
            num_high = len(slopes_high)
            x_high = np.empty(num_high * 3)
            y_high = np.empty(num_high * 3)
            x_high[0::3] = self.data[js_high, 0]
            x_high[1::3] = self.data[min(N_high, len(self.data)-1), 0] + delta
            x_high[2::3] = np.nan
            y_high[0::3] = self.data[js_high, 1]
            y_high[1::3] = self.data[js_high, 1] + (self.data[min(N_high, len(self.data)-1), 0] - self.data[js_high, 0] + delta) * slopes_high
            y_high[2::3] = np.nan
        else:
            x_high, y_high = [], []
        
        # 同理计算低价趋势线
        N_low = len(trend_low)
        slopes_low = []
        js_low = []
        for i in range(1, N_low):
            for slope, j in trend_low[i]:
                if 0 <= j < len(self.data):
                    slopes_low.append(slope)
                    js_low.append(j)
        if slopes_low:
            slopes_low = np.array(slopes_low)
            js_low = np.array(js_low)
            num_low = len(slopes_low)
            x_low = np.empty(num_low * 3)
            y_low = np.empty(num_low * 3)
            x_low[0::3] = self.data[js_low, 0]
            x_low[1::3] = self.data[min(N_low, len(self.data)-1), 0] + delta
            x_low[2::3] = np.nan
            y_low[0::3] = self.data[js_low, 2]
            y_low[1::3] = self.data[js_low, 2] + (self.data[min(N_low, len(self.data)-1), 0] - self.data[js_low, 0] + delta) * slopes_low
            y_low[2::3] = np.nan
        else:
            x_low, y_low = [], []
        
        return x_high, y_high, x_low, y_low

    def horizontal_line(self, trend_high, trend_low):
        """
        生成水平线数据，通常用于展示趋势线的辅助参考线
        
        返回:
          x_high, y_high, x_low, y_low 分别代表高、低价格水平线的数据
        """
        N_high = len(trend_high)
        delta = (self.data[1, 0] - self.data[0, 0]) * 20
        slopes_high = []
        js_high = []
        for i in range(1, N_high):
            for slope, j in trend_high[i]:
                if 0 <= j < len(self.data) and slope <= 0:
                    slopes_high.append(slope)
                    js_high.append(j)
        if slopes_high:
            slopes_high = np.array(slopes_high)
            js_high = np.array(js_high)
            num_high = len(slopes_high)
            x_high = np.empty(num_high * 3)
            y_high = np.empty(num_high * 3)
            x_high[0::3] = self.data[js_high, 0]
            x_high[1::3] = self.data[min(N_high, len(self.data)-1), 0] + delta
            x_high[2::3] = np.nan
            y_high[0::3] = self.data[js_high, 1]
            y_high[1::3] = self.data[js_high, 1]
            y_high[2::3] = np.nan
        else:
            x_high, y_high = [], []
        
        N_low = len(trend_low)
        slopes_low = []
        js_low = []
        for i in range(1, N_low):
            for slope, j in trend_low[i]:
                if 0 <= j < len(self.data) and slope >= 0:
                    slopes_low.append(slope)
                    js_low.append(j)
        if slopes_low:
            slopes_low = np.array(slopes_low)
            js_low = np.array(js_low)
            num_low = len(slopes_low)
            x_low = np.empty(num_low * 3)
            y_low = np.empty(num_low * 3)
            x_low[0::3] = self.data[js_low, 0]
            x_low[1::3] = self.data[min(N_low, len(self.data)-1), 0] + delta
            x_low[2::3] = np.nan
            y_low[0::3] = self.data[js_low, 2]
            y_low[1::3] = self.data[js_low, 2]
            y_low[2::3] = np.nan
        else:
            x_low, y_low = [], []
        
        return x_high, y_high, x_low, y_low

    def update_plot_initial(self):
        """
        初始化图表：加载初始数据、计算趋势、过滤并绘制初始图形，
        同时将初始数据缓存
        """
        # 尝试生成所有初始趋势数据
        try:
            for _ in range(len(self.current_data) - 1):
                trend_high, trend_low, deleted_high, deleted_low = next(self.trend_generator)
        except StopIteration:
            trend_high, trend_low, deleted_high, deleted_low = [], [], [], []
        
        # 初始过滤趋势数据
        trend_high, trend_low = filter_trend_initial(trend_high, trend_low, self.data, self.trend_config)
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

    def safe_update_plot_line(self, plot_name: str, x_data, y_data):
        """
        安全更新绘图线：判断数据是否有效，避免空数据异常
        
        参数:
          plot_name: 绘图线名称
          x_data, y_data: 绘图所需数据
        """
        if len(x_data) > 0 and len(y_data) > 0:
            self.update_plot_line(plot_name, x_data, y_data)
        else:
            self.update_plot_line(plot_name, [], [])
    
    def organize_data(self) -> dict:
        """
        整合当前帧数据：
          - 截取应显示的当前数据帧
          - 调用趋势生成器获得最新趋势信息，并过滤不需要的数据
          - 调用交易策略更新交易信号
          - 返回包含所有绘图数据的字典，并将其缓存

        返回:
          regular_cache: 包含当前帧所有可视化数据的字典
        """
        self.current_data = self.data[self.base_trend_number + self.frame_count:
                                      self.base_trend_number + self.visual_number + self.frame_count]
        self.current_type = self.type_data[self.base_trend_number + self.frame_count:
                                             self.base_trend_number + self.visual_number + self.frame_count]
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
        
        self.last_filtered_high, self.last_filtered_low = self.filter_trend(trend_high, trend_low, self.last_filtered_high, self.last_filtered_low, self.data, self.trend_config)
        
        # 调用交易策略，更新交易信号（触发绘制多空标记）
        self.record_trade = TradingStrategy(
            data=self.data,
            trend_high=self.last_filtered_high,
            trend_low=self.last_filtered_low,
            basic_config={},  # 此处基本配置可以在 main 中统一设置或传入
            trend_config=self.trend_config,
            trading_config={},  # 同上
            record_trade=self.record_trade,
            result_trade=self.result_trade,
            total_result=self.total_result
        ).return_data()
        
        if self.visulize_mode:
            x_high, y_high, x_low, y_low = self.trend_to_line(self.last_filtered_high, self.last_filtered_low)
        else:
            x_high, y_high, x_low, y_low = [], [], [], []
        
        regular_cache = self.extract_cache_data()
        regular_cache['trend_high'] = [x_high, y_high]
        regular_cache['trend_low'] = [x_low, y_low]
        
        if self.visulize_mode:
            x_high, y_high, x_low, y_low = self.horizontal_line(self.last_filtered_high, self.last_filtered_low)
        regular_cache['horizontal_high'] = [x_high, y_high]
        regular_cache['horizontal_low'] = [x_low, y_low]
        
        # 整理交易标记，多头与空头
        long_x, long_y, short_x, short_y = [], [], [], []
        for trade in self.record_trade:
            if trade[-1] == 'long':
                long_x.append(trade[0][0])
                long_y.append(trade[0][1])
            elif trade[-1] == 'short':
                short_x.append(trade[0][0])
                short_y.append(trade[0][2])
        self.plot_lines['long'].setData(long_x, long_y)
        self.plot_lines['short'].setData(short_x, short_y)
        
        self.plot_cache[self.frame_count] = regular_cache
        return regular_cache

    def update_plot(self):
        """
        定时更新函数：
          - 每当定时器触发时，更新当前帧数据
          - 使用缓存机制快速加载已处理数据
          - 更新所有趋势线、水平线及交易信号
          - 检查是否到达数据末尾，若是则停止更新并退出程序
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
        
        if self.frame_count in self.plot_cache:
            cached_data = self.plot_cache[self.frame_count]
            self.current_data = cached_data['current_data']
            self.current_type = cached_data['current_type']
        else:
            cached_data = self.organize_data()
        
        self.update_trend_lines(cached_data)
        self.update_horizontal_lines(cached_data)
        
        if len(self.plot_cache) > self.cache_size:
            self.plot_cache.popitem(last=False)
        
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
    
    def update_trend_lines(self, cached_data: dict):
        """
        更新趋势线数据（高价与低价趋势线）
        """
        x_high, y_high = cached_data.get('trend_high', ([], []))
        x_low, y_low = cached_data.get('trend_low', ([], []))
        self.safe_update_plot_line('trend_high', x_high, y_high)
        self.safe_update_plot_line('trend_low', x_low, y_low)
    
    def update_horizontal_lines(self, cached_data: dict):
        """
        更新水平辅助线数据
        """
        x_high, y_high = cached_data.get('horizontal_high', ([], []))
        x_low, y_low = cached_data.get('horizontal_low', ([], []))
        self.safe_update_plot_line('horizontal_high', x_high, y_high)
        self.safe_update_plot_line('horizontal_low', x_low, y_low)
    
    def on_close(self, event):
        """
        处理窗口关闭事件：停止所有定时器并退出应用
        """
        self.timer.stop()
        self.fps_timer.stop()
        self.app.quit()
        event.accept()
    
    def update_fps(self):
        """
        每秒更新 FPS 显示，修改窗口标题显示当前帧率
        """
        current_time_val = time.time()
        elapsed = current_time_val - self.last_time_fps
        if elapsed > 0:
            self.fps = self.frame_count_fps / elapsed
        else:
            self.fps = 0
        self.win.setWindowTitle(f"FPS: {self.fps:.2f}")
        self.frame_count_fps = 0
        self.last_time_fps = current_time_val
    
    def load_frame_from_cache(self):
        """
        从缓存中加载当前帧数据，并刷新所有绘图线（用于手动切换帧）
        """
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
                self.plot_cache.popitem(last=False)
                print("Removed oldest frame from cache.")
                    
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
    
    def get_min_cached_frame(self) -> int:
        """
        返回缓存中最小的帧编号（用于判断是否可以回退）
        """
        return min(self.plot_cache.keys()) if self.plot_cache else 0
    
    def show_previous_frame(self):
        """
        显示上一帧数据，支持手动回溯
        """
        if self.frame_count > self.get_min_cached_frame():
            self.frame_count -= 1
            self.load_frame_from_cache()
        else:
            print("Reached the oldest cached frame. Cannot go back further.")
    
    def show_next_frame(self):
        """
        显示下一帧数据，支持手动切换
        """
        end_index = self.base_trend_number + self.frame_count + self.visual_number + 1
        if end_index <= len(self.data):
            self.frame_count += 1
            self.load_frame_from_cache()
        else:
            print("Reached end of data. Stopping the plot.")
            self.timer.stop()
            self.fps_timer.stop()
    
    def pause_plotting(self):
        """
        暂停自动更新绘图：停止定时器并更新界面按钮状态
        """
        if not self.is_paused:
            self.timer.stop()
            self.fps_timer.stop()
            self.win.pause_button.setText("Resume")
            self.is_paused = True
            print("Plotting Paused.")
    
    def resume_plotting(self):
        """
        恢复自动更新绘图：重启定时器并更新界面状态
        """
        if self.is_paused:
            self.timer.start(self.update_interval)
            self.fps_timer.start(1000)
            self.win.pause_button.setText("Pause")
            self.is_paused = False
            print("Plotting Resumed.")
    
    def toggle_pause(self):
        """
        切换暂停和恢复状态
        """
        if self.is_paused:
            self.resume_plotting()
        else:
            self.pause_plotting()
    
    def run(self):
        """
        启动图形界面并进入事件循环
        """
        self.win.show()
        self.app.exec_() 