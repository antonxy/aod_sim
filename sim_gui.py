from PySide2 import QtCore, QtWidgets, QtGui
import nidaq_pattern
import hex_grid
import numpy as np
import time

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

try:
    import pco
except ImportError:
    print("PCO lib not found")

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Hex SIM GUI')
        layout = QtWidgets.QFormLayout()

        self.distance_txt = QtWidgets.QLineEdit("0.038")
        layout.addRow("Distance [deg]", self.distance_txt)

        self.steps_x_txt = QtWidgets.QLineEdit("20")
        layout.addRow("Steps X", self.steps_x_txt)

        self.steps_y_txt = QtWidgets.QLineEdit("10")
        layout.addRow("Steps Y", self.steps_y_txt)

        self.orientation_deg_txt = QtWidgets.QLineEdit("0")
        layout.addRow("Orientation [deg]", self.orientation_deg_txt)

        self.pattern_hz_txt = QtWidgets.QLineEdit("40000")
        layout.addRow("Projection rate [Hz]", self.pattern_hz_txt)

        self.exposure_lbl = QtWidgets.QLabel()
        layout.addRow("Exposure time", self.exposure_lbl)

        toolbar = QtWidgets.QToolBar("Toolbar")
        self.addToolBar(toolbar)

        connect_camera_action = QtWidgets.QAction("Connect Camera", self)
        #connect_camera_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        connect_camera_action.triggered.connect(self.connect_camera)
        toolbar.addAction(connect_camera_action)

        take_images_action = QtWidgets.QAction("Take Images", self)
        take_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        take_images_action.triggered.connect(self.take_images)
        toolbar.addAction(take_images_action)


        self.pattern_plot = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.pattern_plot)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.camera_exposure = -1;

    def connect_camera(self):
        self.camera = pco.Camera(debuglevel='error', interface="Camera Link Silicon Software")
        self.camera.default_configuration()
        self.camera.configuration = {
            'roi': (1, 1, 1008, 1008),
            #'trigger': 'external exposure start & software trigger',
            'trigger': 'auto sequence',
            'acquire': 'auto',
        }
        self.configure_camera(1e-3)

    def configure_camera(self, exposure_time_sec):
        if exposure_time_sec != self.camera_exposure:
            self.camera.configuration = {
                'exposure time': exposure_time_sec,
            }
            self.camera_exposure = exposure_time_sec


    def take_images(self):
        dist_deg = float(self.distance_txt.text());
        num_x = int(self.steps_x_txt.text());
        num_y = int(self.steps_y_txt.text());
        orientation_deg = float(self.orientation_deg_txt.text());
        pattern_rate_Hz = float(self.pattern_hz_txt.text());

        pattern_deg = hex_grid.projection_hex_pattern_deg(dist_deg, num_x, num_y, orientation_rad = np.deg2rad(orientation_deg))
        exposure_time_sec = pattern_deg.shape[1] / pattern_rate_Hz

        self.exposure_lbl.setText(f"{exposure_time_sec * 1e3:.1f} ms")

        self.pattern_plot.axes.clear()
        self.pattern_plot.axes.set_title("Projected pattern")
        for i in range(7):
            self.pattern_plot.axes.scatter(pattern_deg[i, :, 0], pattern_deg[i, :, 1])
        self.pattern_plot.draw()

        self.configure_camera(exposure_time_sec)

        self.camera.record(number_of_images=7, mode='sequence non blocking')

        #nidaq_pattern.project_patterns(pattern_deg, pattern_rate_Hz)
        while True:
            running = self.camera.rec.get_status()['is running']
            if not running:
                break
            time.sleep(0.001)

        images, metadatas = self.camera.images()
        images = np.stack(images)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()
