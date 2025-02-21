import numpy as np
import sys
import time
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
from src.plotter.plot_ui import PlotWindow
from src.plotter.plot_setup import setup_plot_lines

from collections import deque


class Plotter:
    def __init__(
        self,
        data: np.ndarray,
        type_data: np.ndarray,
        initial_trend_data: dict,
        visual_number: int,
        delay: int,
        cache_len: int,
    ):
        self.data = data
        self.type_data = type_data
        self.trend_high = initial_trend_data["trend_high"]
        self.trend_low = initial_trend_data["trend_low"]
        self.visual_number = visual_number
        self.delay = delay + 1

        self.frame_count = 0  # 帧计数
        self.start_index = len(self.trend_high) - self.visual_number  # 绘图起始索引
        self.tick_index = self.start_index
        self.current_data = self.data[
            self.start_index : self.start_index + self.visual_number + self.delay
        ]  # 当前绘图数据
        self.current_type = self.type_data[
            self.start_index : self.start_index + self.visual_number + self.delay
        ]

        self.app = QtWidgets.QApplication(sys.argv)
        self.win = PlotWindow()
        self.win.plotter = self
        self.plot = self.win.plot_widget.plotItem

        # 使用独立的函数初始化绘图内容
        self.plot_lines, self.plot_configs = setup_plot_lines(
            self.plot, self.visual_number
        )

        self.set_plot_ranges(self.current_data)
        self.plot.getViewBox().setBackgroundColor("k")

        # 控制器
        self.paused = False  # 是否暂停
        from src.plotter.plot_controller import PlotController

        self.controller = PlotController(self)

        self.plot_cache = deque(maxlen=cache_len)
        self.current_snapshot_index = -1  # 记录当前缓存索引

        # 计算FPS
        self.fps_last_time = time.time()
        self.fps_count = 0

        self.initial_plot()

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

    def initial_plot(self):
        self.signals = {}
        self.close_signals = {}
        self.plot_in_one()

    def update_plot(self, current_trend, signals, close_signals, tick_index):
        self.trend_high = current_trend["trend_high"]
        self.trend_low = current_trend["trend_low"]
        self.signals = signals
        self.close_signals = close_signals
        self.tick_index = tick_index
        self.current_data = self.data[
            self.start_index
            + self.frame_count : self.start_index
            + self.frame_count
            + self.visual_number
            + self.delay
        ]
        # 检查
        if self.current_data.shape[0] < self.visual_number + self.delay:
            print("Reached the end of the data. Auto_apdate is paused.")
            return
        self.current_type = self.type_data[
            self.start_index
            + self.frame_count : self.start_index
            + self.frame_count
            + self.visual_number
            + self.delay
        ]
        self.frame_count += 1
        self.set_plot_ranges(self.current_data)
        self.plot_in_one()

        # 更新FPS
        self.update_fps()

    def update_fps(self):
        self.fps_count += 1
        current_time = time.time()
        elapsed = current_time - self.fps_last_time
        if elapsed >= 1:
            fps = self.fps_count / elapsed
            self.win.setWindowTitle(f"Plotter - FPS: {fps:.2f}")
            self.fps_last_time = current_time
            self.fps_count = 0

    def plot_in_one(self):
        # 缓存当前绘图数据
        snapshot = {}

        # 更新趋势线和水平线
        x_trend_high, y_trend_high, x_trend_low, y_trend_low = self.trend_to_line()
        self.safe_update_plot_line("trend_high", x_trend_high, y_trend_high)
        self.safe_update_plot_line("trend_low", x_trend_low, y_trend_low)
        snapshot["trend_high"] = (x_trend_high, y_trend_high)
        snapshot["trend_low"] = (x_trend_low, y_trend_low)

        # 更新其它图形（如K线、柱状图等，根据plot_configs配置）
        for config in self.plot_configs:
            condition = config["condition"]
            indices = condition(self.current_type)
            filtered_data = self.current_data[indices]
            pairs = filtered_data[:, config["columns"]].reshape(-1, 4)
            x_data = pairs[:, [0, 2]].flatten()
            y_data = pairs[:, [1, 3]].flatten()
            self.safe_update_plot_line(config["name"], x_data, y_data)
            snapshot[config["name"]] = (x_data, y_data)

        # 处理 tick 相关数据，并确保每一帧都更新 bounce 线
        self.update_point(self.signals, "high_bounce", "high_tick_price", snapshot)
        self.update_point(self.signals, "low_bounce", "low_tick_price", snapshot)
        
        self.update_point(self.close_signals, "high_close", "high_tick_price", snapshot)
        self.update_point(self.close_signals, "low_close", "low_tick_price", snapshot)

        # 同时缓存当前的坐标轴范围
        snapshot["axis_range"] = {
            "x": self.plot.viewRange()[0],
            "y": self.plot.viewRange()[1],
        }

        # 更新缓存
        self.plot_cache.append(snapshot)
        if not self.paused:
            self.current_snapshot_index = len(self.plot_cache) - 1

    def update_point(self,signals, point_key, tick_price_key, snapshot):
        """
        统一处理 bounce 线的更新逻辑。若 bounce 信号存在则暂停画面更新，
        并根据 tick_price_key 获取对应数据，更新指定 bounce_key 的数据。
        """
        if signals.get(point_key, None) is not None:
            pass
            # self.paused = True  # 暂停
            
        
        tick_price = signals.get(tick_price_key, None)
        current_x = signals.get(tick_price_key, None)
        current_x = self.data[self.tick_index, 0]
        if tick_price is not None:
            tick_price = np.array(tick_price)
            x_data = np.full(tick_price.shape, current_x)
            y_data = tick_price
        else:
            x_data, y_data = [], []
        self.safe_update_plot_line(point_key, x_data, y_data)
        snapshot[point_key] = (x_data, y_data)

    def refresh_frame(self):
        if self.plot_cache and 0 <= self.current_snapshot_index < len(self.plot_cache):
            snapshot = self.plot_cache[self.current_snapshot_index]
            self.win.plot_widget.setUpdatesEnabled(False)  # 禁用自动更新
            for key, (x_data, y_data) in snapshot.items():
                if key != "axis_range":  # 跳过坐标轴范围
                    self.safe_update_plot_line(key, x_data, y_data)
            # 恢复坐标轴范围
            # if "axis_range" in snapshot:
            #     axis = snapshot["axis_range"]
            #     self.plot.setXRange(*axis["x"])
            #     self.plot.setYRange(*axis["y"])
            self.win.plot_widget.setUpdatesEnabled(True)  # 恢复自动更新
        else:
            print("No snapshot available.")

    def safe_update_plot_line(self, plot_name, x_data, y_data):
        if len(x_data) > 0 and len(y_data) > 0:
            self.plot_lines[plot_name].setData(x_data, y_data)
        else:
            self.plot_lines[plot_name].setData([], [])

    def trend_to_line(self):
        N_high = len(self.trend_high)
        N_low = len(self.trend_low)
        data = self.data
        delta = (data[1, 0] - data[0, 0]) * 120

        # Process trend_high
        slopes_high, js_high, is_high = [], [], []
        for i in range(1, N_high):
            for slope, j in self.trend_high[i]:
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
            x_high[1::3] = data[min(N_high, len(data) - 1), 0] + delta
            x_high[2::3] = np.nan
            y_high[0::3] = data[js_high, 1]
            y_high[1::3] = (
                data[js_high, 1]
                + (data[min(N_high, len(data) - 1), 0] - data[js_high, 0] + delta)
                * slopes_high
            )
            y_high[2::3] = np.nan
        else:
            x_high, y_high = [], []

        # Process trend_low
        slopes_low, js_low, is_low = [], [], []
        for i in range(1, N_low):
            for slope, j in self.trend_low[i]:
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
            x_low[1::3] = data[min(N_low, len(data) - 1), 0] + delta
            x_low[2::3] = np.nan
            y_low[0::3] = data[js_low, 2]
            y_low[1::3] = (
                data[js_low, 2]
                + (data[min(N_low, len(data) - 1), 0] - data[js_low, 0] + delta)
                * slopes_low
            )
            y_low[2::3] = np.nan
        else:
            x_low, y_low = [], []

        return x_high, y_high, x_low, y_low

    def run(self):
        """
        处理 Qt 的事件循环，让界面可以及时刷新和响应用户交互。
        注意：不要在这里调用阻塞式的 app.exec_()，而是依赖 processEvents()。
        """
        # 如果窗口还未显示，首次调用时显示窗口
        if not self.win.isVisible():
            self.win.show()
        # 处理所有pending的事件
        self.app.processEvents()
