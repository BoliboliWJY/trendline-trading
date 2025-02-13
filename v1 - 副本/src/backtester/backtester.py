import time
import datetime
import numpy as np
import copy
import sys
from collections import OrderedDict
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from src.plot_figure.plot_ui import PlotWindow
from src.utils import profile_method

# Import other dependencies from src as needed:
from src.filter.filters import filter_trend, filter_trend_initial
from src.trading_strategy import TradingStrategy
from src.trend_process import calculate_trend, initial_single_slope
from src.trend_calculator.compute_initial_trends import compute_initial_trends

class Backtester:
    def __init__(self, data, type_data, trend_generator, filter_trend, trend_config, base_trend_number = 1000):
        self.data = data
        self.type_data = type_data
        self.trend_generator = trend_generator 
        self.filter_trend = filter_trend
        self.trend_config = trend_config
        self.base_trend_number = base_trend_number #前置趋势数量

        # 趋势管理
        self.deleted_trends = {}
        self.last_filtered_high = []
        self.last_filtered_low = []

        # 可视化参数

        # 当前数据(随回测进行而增长)
        self.current_data = self.data[: self.base_trend_number]
        self.current_type = self.type_data[: self.base_trend_number]

        # 趋势数据
        self.trend_high = []
        self.trend_low = []

        self.backtest_count = 1 # 回测计数

        self.initial_trend() # 初始化趋势
        # self.update_trend() # 更新趋势

    def initial_trend(self):
        """
        初始化趋势，并记录被删除的趋势
        """
        # 计算
        trend_high, trend_low, raw_filtered_high, raw_filtered_low = (
            compute_initial_trends(
                self.current_data,
                self.trend_generator,
                self.data,
                self.trend_config,
                self.last_filtered_high,
                self.last_filtered_low,
            )
        )
        # 保存计算结果
        self.trend_high = trend_high
        self.trend_low = trend_low

        # 过滤趋势
        self.last_filtered_high, self.last_filtered_low = self.filter_trend(
            self.trend_high,
            self.trend_low,
            raw_filtered_high,
            raw_filtered_low,
            self.data,
            self.trend_config,
        )

        self.trend_high, self.trend_low, self.last_filtered_high, self.last_filtered_low = compute_initial_trends(self.current_data, self.trend_generator, self.data, self.trend_config, self.last_filtered_high, self.last_filtered_low)

    def update_trend(self):
        """更新一次趋势数据
        """
        end_index = self.base_trend_number + self.backtest_count
        if end_index >= len(self.data):
            print("Reached end of data. Stopping the backtest.")
            return False # 如果数据结束，返回False

        # 更新当前数据
        self.backtest_count += 1
        self.current_data = self.data[self.base_trend_number: self.base_trend_number + self.backtest_count]
        self.current_type = self.type_data[self.base_trend_number: self.base_trend_number + self.backtest_count]

        # 更新趋势数据
        try:
            self.trend_high, self.trend_low, deleted_high, deleted_low = next(self.trend_generator)
        except StopIteration:
            self.trend_high, self.trend_low, deleted_high, deleted_low = [], [], [], []

        # 更新被删除的趋势
        for i in range(len(self.last_filtered_high)):
            self.last_filtered_high[i] = [item for item in self.last_filtered_high[i] if tuple(item) not in deleted_high]
        for i in range(len(self.last_filtered_low)):
            self.last_filtered_low[i] = [item for item in self.last_filtered_low[i] if tuple(item) not in deleted_low]

        # 添加新的趋势
        self.last_filtered_high.append(self.trend_high[-self.trend_config.get('delay'):][0])
        self.last_filtered_low.append(self.trend_low[-self.trend_config.get('delay'):][0])

        # 过滤趋势
        self.last_filtered_high, self.last_filtered_low = self.filter_trend(
            self.trend_high, self.trend_low, self.last_filtered_high, self.last_filtered_low, self.data, self.trend_config
        )

        return True # 如果数据未结束，返回True

    def run_backtest(self, delay = 0):
        """运行回测
        """
        while self.update_trend():
            if delay > 0:
                time.sleep(delay)
