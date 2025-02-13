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

class Plotter:
    """
    Plotter 类：数据可视化、趋势更新、缓存管理与交易策略执行
    """
    def __init__(self, data, type_data, trend_generator, filter_trend, 
                 trend_config, trading_config, basic_config,
                 base_trend_number=1000, visual_number=100, update_interval=200, cache_size=100):
        # 保存基本数据及配置
        self.data = data
        self.type_data = type_data
        self.trend_generator = trend_generator
        self.filter_trend = filter_trend
        self.trend_config = trend_config
        self.trading_config = trading_config
        self.basic_config = basic_config

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
        self.total_result = [0, 0]

        # 可视化参数
        self.visulize_mode = True  # 是否可视化
        self.is_paused = False  # State to track pause/resume

        # FPS 计数
        self.frame_count_fps = 0
        self.last_time_fps = time.time()
        self.fps = 0

        # 应用和窗口初始化
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

        # Timer for updating plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(self.update_interval)

        # FPS timer
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)

        # 窗口关闭事件处理
        self.win.closeEvent = self.on_close

    def initialize_plot_lines(self):
        """
        Initializes all plot lines based on the plot configurations.
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

        self.plot_lines['long'] = self.plot.plot([], [], pen=None, symbol='o', symbolBrush='green', symbolSize=20, name='Long')
        self.plot_lines['short'] = self.plot.plot([], [], pen=None, symbol='o', symbolBrush='red', symbolSize=20, name='Short')

    def set_plot_ranges(self, data_slice):
        """
        Sets the X and Y ranges of the plot based on the provided data slice.
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
            pairs = filtered_data[:, config['columns']].reshape(-1, 4)
            cache_data[config['name']] = pairs
        return cache_data

    def trend_to_line(self, trend_high, trend_low):
        N_high = len(trend_high)
        N_low = len(trend_low)
        data = self.data
        delta = (data[1, 0] - data[0, 0]) * 120

        # Process trend_high
        slopes_high, js_high, is_high = [], [], []
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
            x_high[0::3] = data[js_high, 0]
            x_high[1::3] = data[min(N_high, len(data)-1), 0] + delta
            x_high[2::3] = np.nan
            y_high[0::3] = data[js_high, 1]
            y_high[1::3] = data[js_high, 1] + (data[min(N_high, len(data)-1), 0] - data[js_high, 0] + delta) * slopes_high
            y_high[2::3] = np.nan
        else:
            x_high, y_high = [], []

        # Process trend_low
        slopes_low, js_low, is_low = [], [], []
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
            x_low[2::3] = np.nan
            y_low[0::3] = data[js_low, 2]
            y_low[1::3] = data[js_low, 2] + (data[min(N_low, len(data)-1), 0] - data[js_low, 0] + delta) * slopes_low
            y_low[2::3] = np.nan
        else:
            x_low, y_low = [], []

        return x_high, y_high, x_low, y_low

    def horizontal_line(self, trend_high, trend_low):
        N_high = len(trend_high)
        N_low = len(trend_low)
        delta = (self.data[1, 0] - self.data[0, 0]) * 20
        slopes_high, js_high, is_high = [], [], []
        for i in range(1, N_high):
            for slope, j in trend_high[i]:
                if 0 <= j < len(self.data) and slope <= 0:
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
            x_high[0::3] = self.data[js_high, 0]
            x_high[1::3] = self.data[min(N_high, len(self.data)-1), 0] + delta
            x_high[2::3] = np.nan
            y_high[0::3] = self.data[js_high, 1]
            y_high[1::3] = self.data[js_high, 1]
            y_high[2::3] = np.nan
        else:
            x_high, y_high = [], []

        slopes_low, js_low, is_low = [], [], []
        for i in range(1, N_low):
            for slope, j in trend_low[i]:
                if 0 <= j < len(self.data) and slope >= 0:
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
            x_low[0::3] = self.data[js_low, 0]
            x_low[1::3] = self.data[min(N_low, len(self.data)-1), 0] + delta
            x_low[2::3] = np.nan
            y_low[0::3] = self.data[js_low, 2]
            y_low[1::3] = self.data[js_low, 2]
            y_low[2::3] = np.nan
        else:
            x_low, y_low = [], []

        return x_high, y_high, x_low, y_low

    def draw_initial_plot(self, trend_high, trend_low):
        """
        Draws the initial plot based on the computed trends.
        """
        if self.visulize_mode:
            # 分别计算趋势线和水平线的绘图坐标，
            # 这样可以灵活调整不同展示逻辑（避免原先直接覆盖的问题）
            x_trend_high, y_trend_high, x_trend_low, y_trend_low = self.trend_to_line(trend_high, trend_low)
            x_horiz_high, y_horiz_high, x_horiz_low, y_horiz_low = self.horizontal_line(trend_high, trend_low)
        else:
            x_trend_high, y_trend_high, x_trend_low, y_trend_low = [], [], [], []
            x_horiz_high, y_horiz_high, x_horiz_low, y_horiz_low = [], [], [], []

        # 更新趋势线和水平线
        self.safe_update_plot_line('trend_high', x_trend_high, y_trend_high)
        self.safe_update_plot_line('trend_low', x_trend_low, y_trend_low)
        self.safe_update_plot_line('horizontal_high', x_horiz_high, y_horiz_high)
        self.safe_update_plot_line('horizontal_low', x_horiz_low, y_horiz_low)

        # 更新其它图形（如K线、柱状图等，根据plot_configs配置）
        for config in self.plot_configs:
            condition = config['condition']
            indices = condition(self.current_type)
            filtered_data = self.current_data[indices]
            pairs = filtered_data[:, config['columns']].reshape(-1, 4)
            x_data = pairs[:, [0, 2]].flatten()
            y_data = pairs[:, [1, 3]].flatten()
            self.safe_update_plot_line(config['name'], x_data, y_data)

        # 更新缓存数据
        cache_data = self.extract_cache_data()
        cache_data['trend_high'] = [x_trend_high, y_trend_high]
        cache_data['trend_low'] = [x_trend_low, y_trend_low]
        cache_data['horizontal_high'] = [x_horiz_high, y_horiz_high]
        cache_data['horizontal_low'] = [x_horiz_low, y_horiz_low]
        self.plot_cache[0] = cache_data

    def update_plot_initial(self):
        """
        Initializes the plot with the first set of data and caches it.
        原先在更新初始图像时同时计算初始趋势和绘制图像，
        现在将趋势计算过程抽取出来方便在实盘初始化时单独使用。
        """
        # 计算初始趋势，实盘初始化也可以直接调用该方法
        trend_high, trend_low, self.last_filtered_high, self.last_filtered_low = (
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

        self.last_filtered_high.append(trend_high[-self.trend_config.get("delay") :][0])
        self.last_filtered_low.append(trend_low[-self.trend_config.get("delay") :][0])
        # 根据计算结果绘制图像
        self.draw_initial_plot(trend_high, trend_low)

    def safe_update_plot_line(self, plot_name, x_data, y_data):
        if len(x_data) > 0 and len(y_data) > 0:
            self.update_plot_line(plot_name, x_data, y_data)
        else:
            self.update_plot_line(plot_name, [], [])

    def organize_data(self):
        """Merge update data for manual and auto operation."""
        self.current_data = self.data[self.base_trend_number + self.frame_count: self.base_trend_number + self.visual_number + self.frame_count]
        self.current_type = self.type_data[self.base_trend_number + self.frame_count: self.base_trend_number + self.visual_number + self.frame_count]
        try:
            trend_high, trend_low, deleted_high, deleted_low = next(self.trend_generator)
        except StopIteration:
            trend_high, trend_low, deleted_high, deleted_low = [], [], [], []

        for idx, item_to_delete in deleted_high:
            if idx < len(self.last_filtered_high):
                try:
                    self.last_filtered_high[idx].remove(item_to_delete)
                except ValueError:
                    # 如果在对应的子列表中没有找到此项，则忽略。
                    pass

        for idx, item_to_delete in deleted_low:
            if idx < len(self.last_filtered_low):
                try:
                    self.last_filtered_low[idx].remove(item_to_delete)
                except ValueError:
                    # 如果在对应的子列表中没有找到此项，则忽略。
                    pass

        self.last_filtered_high.append(trend_high[-self.trend_config.get('delay', 10):][0])
        self.last_filtered_low.append(trend_low[-self.trend_config.get('delay', 10):][0])

        self.last_filtered_high, self.last_filtered_low = self.filter_trend(
            trend_high, trend_low, self.last_filtered_high, self.last_filtered_low, self.data, self.trend_config
        )

        x_high = []
        y_high = []
        x_low = []
        y_low = []
        self.record_trade = TradingStrategy(
            data=self.data, 
            trend_high=self.last_filtered_high, 
            trend_low=self.last_filtered_low, 
            basic_config=self.basic_config, 
            trend_config=self.trend_config, 
            trading_config=self.trading_config, 
            record_trade=self.record_trade, 
            result_trade=self.result_trade, 
            total_result=self.total_result
        ).return_data()

        if self.visulize_mode:
            x_high, y_high, x_low, y_low = self.trend_to_line(self.last_filtered_high, self.last_filtered_low)

        regular_cache = self.extract_cache_data()
        regular_cache['trend_high'] = [x_high, y_high]
        regular_cache['trend_low'] = [x_low, y_low]

        if self.visulize_mode:
            x_high, y_high, x_low, y_low = self.horizontal_line(self.last_filtered_high, self.last_filtered_low)
        regular_cache['horizontal_high'] = [x_high, y_high]
        regular_cache['horizontal_low'] = [x_low, y_low]

        # Trade markers update (long / short)
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
        """Update the later trend data."""
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
                self.plot_cache.popitem(last=False)

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
            self.resume_plotting()
        else:
            self.pause_plotting()

    def run(self):
        self.win.show()
        self.app.exec_() 
