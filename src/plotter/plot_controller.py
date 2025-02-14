class PlotController:
    def __init__(self, plotter):
        self.plotter = plotter

    def toggle_pause(self):
        self.plotter.paused = not self.plotter.paused
        if self.plotter.paused:
            print("Paused")
            self.plotter.win.pause_button.setText("Resume")
        else:
            print("Resumed")
            self.plotter.win.pause_button.setText("Pause")

    def show_previous_frame(self):
        if self.plotter.paused:
            if self.plotter.current_snapshot_index > 0:
                self.plotter.current_snapshot_index -= 1
                self.plotter.refresh_frame()
            else:
                print("No previous frame available.")
        else:
            print("Please pause the plot to navigate frames.")

    def show_next_frame(self):
        if self.plotter.paused:
            if self.plotter.current_snapshot_index < len(self.plotter.plot_cache) - 1:
                self.plotter.current_snapshot_index += 1
                self.plotter.refresh_frame()
            else:
                print("No next frame available.")
        else:
            print("Please pause the plot to navigate frames.")
