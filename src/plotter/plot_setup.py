from pyqtgraph.Qt import QtCore
import pyqtgraph as pg


def setup_plot_lines(plot, visual_number):
    """
    根据 visual_number 配置所有绘图内容，返回一个包含绘图对象（plot_lines）和配置（plot_configs）的字典。
    """
    linewidth = 300 / visual_number # 线宽
    # K线设定
    plot_configs = [
        {
            "name": "high_low_green",
            "color": "green",
            "width": linewidth,
            "columns": [2, 3, 0, 1],
            "connect": "pairs",
            "condition": lambda type_val: type_val == 0,
        },
        {
            "name": "high_low_red",
            "color": "red",
            "width": linewidth,
            "columns": [2, 3, 0, 1],
            "connect": "pairs",
            "condition": lambda type_val: type_val == 1,
        },
        # {
        #     "name": "open_close_green",
        #     "color": "green",
        #     "width": 2 * linewidth,
        #     "columns": [-1, 4, -1, 5],
        #     "connect": "pairs",
        #     "condition": lambda type_val: type_val == 0,
        # },
        # {
        #     "name": "open_close_red",
        #     "color": "red",
        #     "width": 2 * linewidth,
        #     "columns": [-1, 4, -1, 5],
        #     "connect": "pairs",
        #     "condition": lambda type_val: type_val == 1,
        # },
    ]
    # 初始化绘图对象
    plot_lines = {}
    for config in plot_configs:
        pen = pg.mkPen(color=config["color"], width=config["width"])
        plot_lines[config["name"]] = plot.plot(
            [], [], pen=pen, connect=config["connect"]
        )

    # 初始化趋势线
    trend_pen_high = pg.mkPen(color="green", width=0.5, style=QtCore.Qt.SolidLine)
    trend_pen_low = pg.mkPen(color="red", width=0.5, style=QtCore.Qt.SolidLine)
    plot_lines["trend_high"] = plot.plot([], [], pen=trend_pen_high, name="Trend High")
    plot_lines["trend_low"] = plot.plot([], [], pen=trend_pen_low, name="Trend Low")

    horizontal_pen_high = pg.mkPen(color="yellow", width=0.5, style=QtCore.Qt.SolidLine)
    plot_lines["horizontal_high"] = plot.plot(
        [], [], pen=horizontal_pen_high, name="Horizontal High"
    )
    horizontal_pen_low = pg.mkPen(color="white", width=0.5, style=QtCore.Qt.SolidLine)
    plot_lines["horizontal_low"] = plot.plot(
        [], [], pen=horizontal_pen_low, name="Horizontal Low"
    )

    plot_lines["price_time"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush=(255, 255, 255, 100), symbolSize=10, name="Price Time"
    )
    
    # 初始化多空标记点
    plot_lines["long"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush="green", symbolSize=20, name="Long"
    )
    plot_lines["short"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush="red", symbolSize=20, name="Short"
    )

    # 开平仓信号点
    plot_lines["high_open_enter"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush=(102,178,155), symbolSize=15, name="High Open Enter"
    )
    plot_lines["low_open_enter"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush=(255,102,255), symbolSize=15, name="Low Open Enter"
    )
    plot_lines["high_open"] = plot.plot(
        [], [], pen=None, symbol="x", symbolBrush="blue", symbolSize=15, name="High Open"
    )
    plot_lines["low_open"] = plot.plot(
        [], [], pen=None, symbol="x", symbolBrush=(255,0,255), symbolSize=15, name="Low Open"
    )
    plot_lines["high_close"] = plot.plot(
        [], [], pen=None, symbol="x", symbolBrush="purple", symbolSize=15, name="High Close"
    )
    plot_lines["low_close"] = plot.plot(
        [], [], pen=None, symbol="x", symbolBrush="orange", symbolSize=15, name="Low Close"
    )
    
    
    
    plot_lines["trend_price_high"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush=(0, 255, 0, 100), symbolSize=10, name="Trend Price High"
    )
    plot_lines["trend_price_low"] = plot.plot(
        [], [], pen=None, symbol="o", symbolBrush=(255, 0, 0, 100), symbolSize=10, name="Trend Price Low"
    )
    
    # 提前绘制一些k线
    pre_pen_green = pg.mkPen(color=(0, 255, 0, 100), width=linewidth, style=QtCore.Qt.SolidLine)
    pre_pen_red   = pg.mkPen(color=(255, 0, 0, 100), width=linewidth, style=QtCore.Qt.SolidLine)
    plot_lines["pre_high_low_green"] = plot.plot([], [], pen=pre_pen_green, connect="pairs", name="Pre High Low Green")
    plot_lines["pre_high_low_red"]   = plot.plot([], [], pen=pre_pen_red, connect="pairs", name="Pre High Low Red")

    return plot_lines, plot_configs
