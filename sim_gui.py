from PySide2 import QtCore, QtWidgets, QtGui
import hex_grid
import numpy as np
import time
from hexSimProcessor import HexSimProcessor
import scipy
import threading
import tifffile
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')

#matplotlib.rcParams["figure.autolayout"] = True  # tight layout, leads to plots moving around sometimes
matplotlib.rcParams["figure.subplot.bottom"] = 0.04
matplotlib.rcParams["figure.subplot.top"] = 1.0
matplotlib.rcParams["figure.subplot.left"] = 0.04
matplotlib.rcParams["figure.subplot.right"] = 1.0
matplotlib.rcParams["figure.subplot.wspace"] = 0.1
matplotlib.rcParams["figure.subplot.hspace"] = 0.1

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

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent)
        lay = QtWidgets.QVBoxLayout()
        self.plot = MplCanvas(self, width=5, height=100, dpi=100)

        layh = QtWidgets.QHBoxLayout()
        toolbar = NavigationToolbar(self.plot, self)
        layh.addWidget(toolbar)

        self.clim_min_txt = QtWidgets.QLineEdit("None")
        self.clim_min_txt.returnPressed.connect(self.clim_changed)
        self.clim_min_txt.setMaximumWidth(100)
        self.clim_max_txt = QtWidgets.QLineEdit("None")
        self.clim_max_txt.returnPressed.connect(self.clim_changed)
        self.clim_max_txt.setMaximumWidth(100)

        layh.addWidget(self.clim_min_txt)
        layh.addWidget(self.clim_max_txt)

        lay.addLayout(layh)
        lay.addWidget(self.plot)
        self.setLayout(lay)

        self.clim_axes = None

    def clim_changed(self):
        cmin = None
        if self.clim_min_txt.text() != "None":
            cmin = float(self.clim_min_txt.text())
        cmax = None
        if self.clim_max_txt.text() != "None":
            cmax = float(self.clim_max_txt.text())

        if self.clim_axes is not None:
            for ax in self.clim_axes:
                ax.set(clim = (cmin, cmax))
            self.plot.draw()

    def connect_clim(self, axes):
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        self.clim_axes = axes
        self.clim_changed()



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Hex SIM GUI')
        layout = QtWidgets.QFormLayout()

        self.distance_txt = QtWidgets.QLineEdit("0.0345")
        layout.addRow("Distance [deg]", self.distance_txt)

        self.steps_x_txt = QtWidgets.QLineEdit("18")
        layout.addRow("Steps X", self.steps_x_txt)

        self.steps_y_txt = QtWidgets.QLineEdit("10")
        layout.addRow("Steps Y", self.steps_y_txt)

        self.orientation_deg_txt = QtWidgets.QLineEdit("0")
        layout.addRow("Orientation [deg]", self.orientation_deg_txt)

        self.aspect_txt = QtWidgets.QLineEdit("1.045")
        layout.addRow("Aspect ratio", self.aspect_txt)

        self.pattern_hz_txt = QtWidgets.QLineEdit("40000")
        layout.addRow("Projection rate [Hz]", self.pattern_hz_txt)

        self.exposure_lbl = QtWidgets.QLabel()
        layout.addRow("Exposure time", self.exposure_lbl)

        self.reconstruction_size_txt = QtWidgets.QLineEdit("256")
        layout.addRow("Reconstruction size N", self.reconstruction_size_txt)

        self.reconstruction_offset_x = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset x", self.reconstruction_offset_x)

        self.reconstruction_offset_y = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset y", self.reconstruction_offset_y)

        self.reconstruction_eta_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Reconstruction eta", self.reconstruction_eta_txt)

        cameraMenu = self.menuBar().addMenu("&Camera")
        connect_camera_action = QtWidgets.QAction("&Connect Camera", self)
        connect_camera_action.triggered.connect(self.connect_camera)
        cameraMenu.addAction(connect_camera_action)

        disconnect_camera_action = QtWidgets.QAction("&Disconnect Camera", self)
        disconnect_camera_action.triggered.connect(self.disconnect_camera)
        cameraMenu.addAction(disconnect_camera_action)

        take_images_action = QtWidgets.QAction("Take &Images", self)
        take_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        take_images_action.triggered.connect(self.take_images)
        cameraMenu.addAction(take_images_action)

        load_images_action = QtWidgets.QAction("&Open Images", self)
        load_images_action.setShortcut(QtGui.QKeySequence.Open)
        load_images_action.triggered.connect(self.load_images)
        cameraMenu.addAction(load_images_action)

        save_images_action = QtWidgets.QAction("&Save Images", self)
        save_images_action.setShortcut(QtGui.QKeySequence.Save)
        save_images_action.triggered.connect(self.save_images)
        cameraMenu.addAction(save_images_action)

        reconstructMenu = self.menuBar().addMenu("&Reconstruct")
        reconstruct_images_action = QtWidgets.QAction("&Reconstruct Images", self)
        reconstruct_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_R))
        reconstruct_images_action.triggered.connect(self.reconstruct_image)
        reconstructMenu.addAction(reconstruct_images_action)

        reconstruct_images_nocal_action = QtWidgets.QAction("Reconstruct (&No cal)", self)
        reconstruct_images_nocal_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_F))
        reconstruct_images_nocal_action.triggered.connect(self.reconstruct_image_nocal)
        reconstructMenu.addAction(reconstruct_images_nocal_action)

        project_pattern_loop_action = QtWidgets.QAction("Project &Pattern", self)
        project_pattern_loop_action.triggered.connect(self.project_pattern_loop)
        cameraMenu.addAction(project_pattern_loop_action)

        close_action = QtWidgets.QAction("&Quit", self)
        close_action.setShortcut(QtGui.QKeySequence.Quit)
        close_action.triggered.connect(self.close)
        cameraMenu.addAction(close_action)

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

        self.image_plot = PlotWidget(self)
        self.tab_widget.addTab(self.image_plot, "Recorded Images")

        self.fft_plot = PlotWidget(self)
        self.tab_widget.addTab(self.fft_plot, "FFT")

        self.carrier_plot = PlotWidget(self)
        self.tab_widget.addTab(self.carrier_plot, "Carrier")

        self.recon_plot = PlotWidget(self)
        self.tab_widget.addTab(self.recon_plot, "Reconstructed Image")

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
        self.plot_images()

    def plot_images(self):
        self.image_plot.plot.fig.clear()
        axs = self.image_plot.plot.fig.subplots(3, 3, sharex=True, sharey=True)
        ims = []
        for i in range(3):
            ims.append(axs[0, i].imshow(self.frames[i]))
        for i in range(3):
            ims.append(axs[1, i].imshow(self.frames[i + 3]))
        ims.append(axs[2, i].imshow(self.frames[6]))
        self.image_plot.connect_clim(ims)
        self.image_plot.plot.draw()

        self.fft_plot.plot.clear()
        im = self.fft_plot.plot.axes.imshow(np.abs(np.fft.fftshift(np.fft.fft2(self.frames[0]))))
        self.fft_plot.connect_clim(im)
        self.fft_plot.plot.draw()

    def load_images(self):
        file_dialog = QtWidgets.QFileDialog()
        filename = file_dialog.getOpenFileName(self, "Load tiff file", "", "TIFF (*.tiff);; TIF (*.tif)")[0]
        if filename != "":
            self.frames = tifffile.imread(filename)
            self.plot_images()

    def save_images(self):
        file_dialog = QtWidgets.QFileDialog()
        filename = file_dialog.getSaveFileName(self, "Save tiff file", "", "TIFF (*.tiff);; TIF (*.tif)")[0]
        if filename != "":
            path = Path(filename)
            if path.suffix == "":
                filename += ".tiff"
            tifffile.imwrite(filename, self.frames)

    def reconstruct_image(self):
        N = int(self.reconstruction_size_txt.text())
        offset_x = int(self.reconstruction_offset_x.text())
        offset_y = int(self.reconstruction_offset_y.text())
        eta = float(self.reconstruction_eta_txt.text())

        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        self.p.N = N
        self.p.eta = eta
        self.p.calibrate(frames)

        self.carrier_plot.plot.fig.clear()
        axs = self.carrier_plot.plot.fig.subplots(2, 3)
        ims = []
        for i in range(3):
            ims.append(axs[0, i].imshow(self.p.carrier_debug_img[i]))
        for i in range(3):
            ims.append(axs[1, i].imshow(self.p.carrier_debug_zoom_img[i]))
        self.carrier_plot.connect_clim(ims)
        self.carrier_plot.plot.draw()

        self.reconstruct_image_nocal()

    def reconstruct_image_nocal(self):
        # TODO this breaks if params changed inbetween
        N = int(self.reconstruction_size_txt.text())
        offset_x = int(self.reconstruction_offset_x.text())
        offset_y = int(self.reconstruction_offset_y.text())

        assert self.frames.shape[0] == 7
        frames = self.frames[:, offset_y:offset_y + N, offset_x:offset_x + N]

        reconstruct = self.p.reconstruct_fftw(frames)

        self.recon_plot.plot.fig.clear()
        ax1, ax2 = self.recon_plot.plot.fig.subplots(1, 2, sharex=True, sharey=True)
        im1 = ax1.imshow(reconstruct)
        im2 = ax2.imshow(scipy.ndimage.zoom(np.sum(frames, axis=0), (2, 2), order=1))
        self.recon_plot.connect_clim([im1, im2])
        self.recon_plot.plot.draw()

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
