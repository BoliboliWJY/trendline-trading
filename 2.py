#%%
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np
import random

# ---------------------------
# 2.1. Generator Function
# ---------------------------
#%%
def data_generator(initial_num_lines, max_x=1.0, max_y=1.0):
    """
    Generator that yields new lines defined by two points.
    Each line is a sorted list of two points based on the x-coordinate.
    """
    while True:
        # Generate two random points
        point1 = [random.uniform(0, max_x), random.uniform(0, max_y)]
        point2 = [random.uniform(0, max_x), random.uniform(0, max_y)]
        # Sort points based on x-coordinate
        sorted_line = sorted([point1, point2], key=lambda p: p[0])
        yield sorted_line

# ---------------------------
# 2.2. PyQtGraph Setup
# ---------------------------

def main():
    # Parameters
    INITIAL_NUM_LINES = 10000  # Starting number of lines
    POINTS_PER_LINE = 2
    UPDATE_INTERVAL = 50  # milliseconds between updates
    MAX_LINES = 20000  # Maximum number of lines to retain for performance
    NEW_LINES_PER_UPDATE = 100  # Number of new lines to add each update

    # Initialize the application
    app = QtWidgets.QApplication(sys.argv)

    # Create a window
    win = pg.GraphicsLayoutWidget(show=True, title="PyQtGraph Real-Time Dynamic Lines")
    win.resize(1200, 800)
    win.setWindowTitle('PyQtGraph Real-Time Dynamic Lines Visualization')

    # Add a plot
    plot = win.addPlot()
    plot.setXRange(0, 1)
    plot.setYRange(0, 1)
    plot.getViewBox().setBackgroundColor('k')  # Black background

    # Initialize data: NumPy array of shape (num_lines * 2, 2)
    # Each pair of points defines a separate line
    data = np.random.rand(INITIAL_NUM_LINES, POINTS_PER_LINE, 2)
    data = np.sort(data, axis=1)  # Sort each line based on x-coordinate
    lines_array = data.reshape(-1, 2)  # Flatten to shape (num_lines*2, 2)

    # Create a PlotDataItem with multiple lines
    plot_item = plot.plot(
        lines_array[:, 0],
        lines_array[:, 1],
        pen=pg.mkPen(color=(0, 255, 0, 50), width=1),  # Green lines with transparency
        connect='pairs'  # Connect every pair of points as separate lines
    )

    # Initialize the generator
    gen = data_generator(INITIAL_NUM_LINES)

    # ---------------------------
    # 2.3. Update Function
    # ---------------------------

    def update_plot():
        nonlocal data, lines_array, plot_item, gen

        # -----------------------
        # 2.3.1. Modify Existing Lines
        # -----------------------

        # Apply a small random shift to y-coordinates of existing points
        # Shifts range between -0.001 and 0.001
        shifts = np.random.uniform(-0.001, 0.001, size=data.shape)
        data[:, :, 1] += shifts[:, :, 1]  # Only shift y-coordinates
        # Ensure y-coordinates stay within [0, 1]
        np.clip(data[:, :, 1], 0, 1, out=data[:, :, 1])

        # -----------------------
        # 2.3.2. Add New Lines from Generator
        # -----------------------

        try:
            new_lines = [next(gen) for _ in range(NEW_LINES_PER_UPDATE)]
        except StopIteration:
            # In practice, this should not occur as the generator is infinite
            new_lines = []

        if new_lines:
            # Convert new lines to NumPy array
            new_data = np.array(new_lines)  # Shape: (NEW_LINES_PER_UPDATE, 2, 2)
            # Sort each new line based on x-coordinate
            new_data = np.sort(new_data, axis=1)
            # Append to existing data
            data = np.vstack([data, new_data])
            # Update lines_array
            lines_array = data.reshape(-1, 2)

        # -----------------------
        # 2.3.3. Remove Old Lines if Exceeding MAX_LINES
        # -----------------------

        if data.shape[0] > MAX_LINES:
            excess = data.shape[0] - MAX_LINES
            data = data[excess:]  # Remove the oldest lines
            # Update lines_array
            lines_array = data.reshape(-1, 2)

        # -----------------------
        # 2.3.4. Update the Plot
        # -----------------------

        plot_item.setData(
            lines_array[:, 0],
            lines_array[:, 1]
        )

    # ---------------------------
    # 2.4. Timer Setup for Real-Time Updates
    # ---------------------------

    # Set up a timer to call the update function periodically
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot)
    timer.start(UPDATE_INTERVAL)  # Update every 50 ms

    # ---------------------------
    # 2.5. Graceful Shutdown Handling
    # ---------------------------

    def on_close():
        timer.stop()
        QtWidgets.QApplication.quit()

    win.closeEvent = lambda event: on_close()

    # ---------------------------
    # 2.6. Start the Event Loop
    # ---------------------------

    try:
        sys.exit(app.exec_())
    except AttributeError:
        # For PyQt6 compatibility
        sys.exit(app.exec())

if __name__ == "__main__":
    main()
