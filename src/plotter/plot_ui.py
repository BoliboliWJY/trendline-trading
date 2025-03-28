import datetime
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        # 将时间戳转换为日期时间字符串
        return [
            datetime.datetime.fromtimestamp(value / 1000, tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")
            for value in values
        ]


class PlotWindow(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.plotter = None  # Will be set by Plotter
        self.setWindowTitle("Data Plotter with FPS")
        self.resize(1920, 1080)
        self.setStyleSheet(
            """
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
        """
        )
        # Create a vertical layout
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

        time_axis = TimeAxisItem(orientation="bottom")
        # Create the pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget(axisItems={"bottom": time_axis})
        layout.addWidget(self.plot_widget)

        # Horizontal layout for buttons
        button_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(button_layout)

        self.prev_button = QtWidgets.QPushButton("Previous")
        button_layout.addWidget(self.prev_button)
        self.prev_button.clicked.connect(self.on_prev_button)

        self.next_button = QtWidgets.QPushButton("Next")
        button_layout.addWidget(self.next_button)
        self.next_button.clicked.connect(self.on_next_button)

        # Add a Print Frame Count button
        self.print_frame_count_button = QtWidgets.QPushButton("Print Frame Count")
        button_layout.addWidget(self.print_frame_count_button)
        self.print_frame_count_button.clicked.connect(self.on_print_frame_count_button)

        # Add a Pause/Resume button
        self.pause_button = QtWidgets.QPushButton("Pause")
        layout.addWidget(self.pause_button)
        self.pause_button.clicked.connect(self.on_pause_button)

    def on_prev_button(self):
        if self.plotter:
            self.plotter.controller.show_previous_frame()

    def on_next_button(self):
        if self.plotter:
            self.plotter.controller.show_next_frame()

    def on_pause_button(self):
        if self.plotter:
            self.plotter.controller.toggle_pause()

    def on_print_frame_count_button(self):
        if self.plotter:
            self.plotter.print_frame_count()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            if self.plotter:
                self.plotter.controller.toggle_pause()
        elif event.key() == QtCore.Qt.Key_Left:
            if self.plotter:
                self.plotter.controller.show_previous_frame()
        elif event.key() == QtCore.Qt.Key_Right:
            if self.plotter:
                self.plotter.controller.show_next_frame()
        else:
            super().keyPressEvent(event)
