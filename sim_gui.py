from PySide2 import QtCore, QtWidgets, QtGui
import hex_grid
import numpy as np
import time
from hexSimProcessor import HexSimProcessor
import scipy
import threading

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams["figure.autolayout"] = True  # tight layout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

simulate = True
if simulate:
    from sim_simulated_system import SIMSimulatedSystem as SIMSystem
else:
    from sim_hardware_system import SIMHardwareSystem as SIMSystem


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def clear(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)


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

        self.aspect_txt = QtWidgets.QLineEdit("1.0")
        layout.addRow("Aspect ratio", self.aspect_txt)

        self.pattern_hz_txt = QtWidgets.QLineEdit("40000")
        layout.addRow("Projection rate [Hz]", self.pattern_hz_txt)

        self.exposure_lbl = QtWidgets.QLabel()
        layout.addRow("Exposure time", self.exposure_lbl)

        self.reconstruction_size_txt = QtWidgets.QLineEdit("128")
        layout.addRow("Reconstruction size N", self.reconstruction_size_txt)

        self.reconstruction_offset_x = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset x", self.reconstruction_offset_x)

        self.reconstruction_offset_y = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset y", self.reconstruction_offset_y)

        toolbar = QtWidgets.QToolBar("Toolbar")
        self.addToolBar(toolbar)

        connect_camera_action = QtWidgets.QAction("Connect Camera", self)
        connect_camera_action.triggered.connect(self.connect_camera)
        toolbar.addAction(connect_camera_action)

        disconnect_camera_action = QtWidgets.QAction("Disconnect Camera", self)
        disconnect_camera_action.triggered.connect(self.disconnect_camera)
        toolbar.addAction(disconnect_camera_action)

        take_images_action = QtWidgets.QAction("Take Images", self)
        take_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        take_images_action.triggered.connect(self.take_images)
        toolbar.addAction(take_images_action)

        reconstruct_images_action = QtWidgets.QAction("Reconstruct Images", self)
        reconstruct_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_R))
        reconstruct_images_action.triggered.connect(self.reconstruct_image)
        toolbar.addAction(reconstruct_images_action)

        reconstruct_images_nocal_action = QtWidgets.QAction("Reconstruct (No cal)", self)
        reconstruct_images_nocal_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_F))
        reconstruct_images_nocal_action.triggered.connect(self.reconstruct_image_nocal)
        toolbar.addAction(reconstruct_images_nocal_action)

        project_pattern_loop_action = QtWidgets.QAction("Project Pattern", self)
        project_pattern_loop_action.triggered.connect(self.project_pattern_loop)
        toolbar.addAction(project_pattern_loop_action)

        close_action = QtWidgets.QAction("Quit", self)
        close_action.setShortcut(QtGui.QKeySequence.Quit)
        close_action.triggered.connect(self.close)
        toolbar.addAction(close_action)

        self.tab_widget = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tab_widget)

        lay = QtWidgets.QVBoxLayout()
        self.pattern_plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.pattern_plot, self)
        lay.addWidget(toolbar)
        lay.addWidget(self.pattern_plot)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.tab_widget.addTab(w, "Pattern")

        lay = QtWidgets.QVBoxLayout()
        self.image_plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.image_plot, self)
        lay.addWidget(toolbar)
        lay.addWidget(self.image_plot)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.tab_widget.addTab(w, "Recorded image")

        lay = QtWidgets.QVBoxLayout()
        self.carrier_plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.carrier_plot, self)
        lay.addWidget(toolbar)
        lay.addWidget(self.carrier_plot)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.tab_widget.addTab(w, "Carrier")

        lay = QtWidgets.QVBoxLayout()
        self.recon_plot = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.recon_plot, self)
        lay.addWidget(toolbar)
        lay.addWidget(self.recon_plot)
        w = QtWidgets.QWidget()
        w.setLayout(lay)
        self.tab_widget.addTab(w, "Reconstructed image")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.sim_system = SIMSystem()

        self.p = HexSimProcessor()
        self.p.debug = False
        self.p.use_filter = False

    def connect_camera(self):
        self.sim_system.connect()

    def disconnect_camera(self):
        self.sim_system.disconnect()

    def create_patterns(self):
        dist_deg = float(self.distance_txt.text());
        num_x = int(self.steps_x_txt.text());
        num_y = int(self.steps_y_txt.text());
        orientation_deg = float(self.orientation_deg_txt.text());
        aspect = float(self.aspect_txt.text());
        self.pattern_rate_Hz = float(self.pattern_hz_txt.text());

        pattern_deg = hex_grid.projection_hex_pattern_deg(dist_deg, num_x, num_y, orientation_rad = np.deg2rad(orientation_deg), aspect_ratio=aspect)

        self.exposure_time_sec = pattern_deg.shape[1] / self.pattern_rate_Hz
        self.exposure_lbl.setText(f"{self.exposure_time_sec * 1e3:.1f} ms")

        self.pattern_plot.fig.clear()
        ax1, ax2 = self.pattern_plot.fig.subplots(1, 2, sharex=True, sharey=True)
        for i in range(7):
            ax1.scatter(pattern_deg[i, :, 0], pattern_deg[i, :, 1])
        ax2.scatter(pattern_deg[0, :, 0], pattern_deg[0, :, 1])
        self.pattern_plot.draw()

        self.pattern_deg = pattern_deg


    def take_images(self):
        self.create_patterns()

        self.frames = self.sim_system.project_patterns_and_take_images(self.pattern_deg, self.pattern_rate_Hz)

        #self.image_plot.clear()
        #im = self.image_plot.axes.imshow(self.frames[0])
        #self.image_plot.fig.colorbar(im)
        #self.image_plot.draw()

        self.image_plot.fig.clear()
        axs = self.image_plot.fig.subplots(3, 3, sharex=True, sharey=True)
        for i in range(3):
            axs[0, i].imshow(self.frames[i])
        for i in range(3):
            axs[1, i].imshow(self.frames[i + 3])
        axs[2, i].imshow(self.frames[6])
        self.image_plot.draw()

    def reconstruct_image(self):
        #import tifffile
        #self.frames = tifffile.imread("/home/anton/studium/eth/master_thesis/sim/hex_sim/5_with_python_softw/dist_60.tiff")

        N = int(self.reconstruction_size_txt.text())
        offset_x = int(self.reconstruction_offset_x.text())
        offset_y = int(self.reconstruction_offset_y.text())

        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        self.p.N = N
        self.p.calibrate(frames)

        self.carrier_plot.fig.clear()
        axs = self.carrier_plot.fig.subplots(2, 3)
        for i in range(3):
            axs[0, i].imshow(self.p.carrier_debug_img[i])
        for i in range(3):
            axs[1, i].imshow(self.p.carrier_debug_zoom_img[i])
        self.carrier_plot.draw()

        self.reconstruct_image_nocal()

    def reconstruct_image_nocal(self):
        # TODO this breaks if params changed inbetween
        N = int(self.reconstruction_size_txt.text())
        offset_x = int(self.reconstruction_offset_x.text())
        offset_y = int(self.reconstruction_offset_y.text())

        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        reconstruct = self.p.reconstruct_fftw(frames)

        self.recon_plot.fig.clear()
        ax1, ax2 = self.recon_plot.fig.subplots(1, 2, sharex=True, sharey=True)
        ax1.imshow(reconstruct)
        ax2.imshow(scipy.ndimage.zoom(np.sum(frames, axis=0), (2, 2), order=1))
        self.recon_plot.draw()

    def project_pattern_loop(self):
        self.create_patterns()
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target = self.sim_system.project_patterns_looping, args = (self.pattern_deg, self.pattern_rate_Hz, run_event))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Projecting pattern. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()