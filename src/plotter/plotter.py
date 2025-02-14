import numpy as np
import sys
import time
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg
from src.plotter.plot_ui import PlotWindow


class Plotter:
    def __init__(
        self,
        data: np.ndarray,
        type_data: np.ndarray,
        initial_trend_data: dict,
        visual_number: int,
        delay: int,
    ):
        self.data = data
        self.type_data = type_data
        self.trend_high = initial_trend_data["trend_high"]
        self.trend_low = initial_trend_data["trend_low"]
        self.visual_number = visual_number
        self.delay = delay + 1

        self.frame_count = 0  # 帧计数
        self.start_index = len(self.trend_high) - self.visual_number  # 绘图起始索引
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

        self.plot_lines = {}
        self.plot_config()

        self.set_plot_ranges(self.current_data)
        self.plot.getViewBox().setBackgroundColor("k")

        self.initial_plot()

        #
        self.fps_last_time = time.time()
        self.fps_count = 0

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

    def plot_config(self):
        linewidth = 300 / self.visual_number
        self.plot_configs = [
            {
                "name": "high_low_green",
                "color": "green",
                "width": linewidth,
                "columns": [0, 2, 0, 1],
                "connect": "pairs",
                "condition": lambda type_val: type_val == 0,
            },
            {
                "name": "high_low_red",
                "color": "red",
                "width": linewidth,
                "columns": [0, 2, 0, 1],
                "connect": "pairs",
                "condition": lambda type_val: type_val == 1,
            },
            {
                "name": "open_close_green",
                "color": "green",
                "width": 2 * linewidth,
                "columns": [0, 3, 0, 4],
                "connect": "pairs",
                "condition": lambda type_val: type_val == 0,
            },
            {
                "name": "open_close_red",
                "color": "red",
                "width": 2 * linewidth,
                "columns": [0, 3, 0, 4],
                "connect": "pairs",
                "condition": lambda type_val: type_val == 1,
            },
        ]

        for config in self.plot_configs:
            pen = pg.mkPen(color=config["color"], width=config["width"])
            self.plot_lines[config["name"]] = self.plot.plot(
                [], [], pen=pen, connect=config["connect"]
            )

        # Initialize Trend Lines
        trend_pen_high = pg.mkPen(color="green", width=0.5, style=QtCore.Qt.SolidLine)
        trend_pen_low = pg.mkPen(color="red", width=0.5, style=QtCore.Qt.SolidLine)
        self.plot_lines["trend_high"] = self.plot.plot(
            [], [], pen=trend_pen_high, name="Trend High"
        )
        self.plot_lines["trend_low"] = self.plot.plot(
            [], [], pen=trend_pen_low, name="Trend Low"
        )

        horizontal_pen_high = pg.mkPen(
            color="yellow", width=0.5, style=QtCore.Qt.SolidLine
        )
        self.plot_lines["horizontal_high"] = self.plot.plot(
            [], [], pen=horizontal_pen_high, name="Horizontal High"
        )
        horizontal_pen_low = pg.mkPen(
            color="white", width=0.5, style=QtCore.Qt.SolidLine
        )
        self.plot_lines["horizontal_low"] = self.plot.plot(
            [], [], pen=horizontal_pen_low, name="Horizontal Low"
        )

        self.plot_lines["long"] = self.plot.plot(
            [],
            [],
            pen=None,
            symbol="o",
            symbolBrush="green",
            symbolSize=20,
            name="Long",
        )
        self.plot_lines["short"] = self.plot.plot(
            [], [], pen=None, symbol="o", symbolBrush="red", symbolSize=20, name="Short"
        )

    def initial_plot(self):
        self.plot_in_one()

    def update_plot(self, current_trend):
        self.trend_high = current_trend["trend_high"]
        self.trend_low = current_trend["trend_low"]
        self.current_data = self.data[
            self.start_index
            + self.frame_count : self.start_index
            + self.frame_count
            + self.visual_number
            + self.delay
        ]
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
        # 更新趋势线和水平线
        x_trend_high, y_trend_high, x_trend_low, y_trend_low = self.trend_to_line()
        self.safe_update_plot_line("trend_high", x_trend_high, y_trend_high)
        self.safe_update_plot_line("trend_low", x_trend_low, y_trend_low)

        # 更新其它图形（如K线、柱状图等，根据plot_configs配置）
        for config in self.plot_configs:
            condition = config["condition"]
            indices = condition(self.current_type)
            filtered_data = self.current_data[indices]
            pairs = filtered_data[:, config["columns"]].reshape(-1, 4)
            x_data = pairs[:, [0, 2]].flatten()
            y_data = pairs[:, [1, 3]].flatten()
            self.safe_update_plot_line(config["name"], x_data, y_data)

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
