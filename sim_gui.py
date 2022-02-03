from PySide2 import QtCore, QtWidgets, QtGui
import hex_grid
import numpy as np
import time
from hexSimProcessor import HexSimProcessor
import scipy
import threading
import tifffile
from pathlib import Path
from detect_orientation_dialog import DetectOrientationDialog
import subprocess
import json

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

import sys

simulate = len(sys.argv) > 1 and sys.argv[1] == 'sim'
if simulate:
    from sim_simulated_system import SIMSimulatedSystem as SIMSystem
else:
    from sim_hardware_system import SIMHardwareSystem as SIMSystem


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'describe', '--always', '--dirty', '--tags']).decode('ascii').strip()


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def clear(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)

class ClimWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ClimWidget, self).__init__(parent)
        self.parent = parent
        layh = QtWidgets.QHBoxLayout()

        self.clim_min_txt = QtWidgets.QLineEdit("None")
        self.clim_min_txt.returnPressed.connect(self.clim_changed)
        self.clim_min_txt.setMaximumWidth(100)
        self.clim_max_txt = QtWidgets.QLineEdit("None")
        self.clim_max_txt.returnPressed.connect(self.clim_changed)
        self.clim_max_txt.setMaximumWidth(100)

        layh.addWidget(self.clim_min_txt)
        layh.addWidget(self.clim_max_txt)

        self.setLayout(layh)

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
            self.parent.plot.draw()

    def connect_clim(self, axes):
        if not hasattr(axes, "__iter__"):
            axes = [axes]
        self.clim_axes = axes
        self.clim_changed()

class PlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, num_clim = 1):
        super(PlotWidget, self).__init__(parent)
        lay = QtWidgets.QVBoxLayout()
        self.plot = MplCanvas(self, width=5, height=100, dpi=100)

        layh = QtWidgets.QHBoxLayout()
        toolbar = NavigationToolbar(self.plot, self)
        layh.addWidget(toolbar)
        layh.addStretch(1)

        self.clim_widgets = []
        for i in range(num_clim):
            climw = ClimWidget(self)
            self.clim_widgets.append(climw)
            layh.addSpacing(10)
            layh.addWidget(climw)

        lay.addLayout(layh)
        lay.addWidget(self.plot)
        self.setLayout(lay)

    def connect_clim(self, axes, clim_num = 0):
        self.clim_widgets[clim_num].connect_clim(axes)



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Hex SIM GUI')

        recording_group = QtWidgets.QGroupBox("Recording")
        layout = QtWidgets.QFormLayout()
        self.orientation_deg_txt = QtWidgets.QLineEdit("0")
        layout.addRow("Orientation [deg]", self.orientation_deg_txt)

        self.desired_distance_txt = QtWidgets.QLineEdit("0.0345")
        layout.addRow("Desired dot distance [deg]", self.desired_distance_txt)

        self.grating_distance_x_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance x [deg]", self.grating_distance_x_txt)

        self.grating_distance_y_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance y [deg]", self.grating_distance_y_txt)

        self.steps_x_lbl = QtWidgets.QLabel()
        layout.addRow("Steps X", self.steps_x_lbl)

        self.steps_y_lbl = QtWidgets.QLabel()
        layout.addRow("Steps Y", self.steps_y_lbl)

        self.dot_distance_x_lbl = QtWidgets.QLabel()
        layout.addRow("Dot distance x", self.dot_distance_x_lbl)

        self.aspect_lbl = QtWidgets.QLabel()
        layout.addRow("Aspect ratio", self.aspect_lbl)

        self.pattern_hz_txt = QtWidgets.QLineEdit("40000")
        layout.addRow("Projection rate [Hz]", self.pattern_hz_txt)

        self.exposure_lbl = QtWidgets.QLabel()
        layout.addRow("Exposure time", self.exposure_lbl)

        self.image_notes_txt = QtWidgets.QLineEdit("")
        layout.addRow("Recording notes", self.image_notes_txt)

        recording_group.setLayout(layout)

        reconstruction_group = QtWidgets.QGroupBox("Reconstruction")
        layout = QtWidgets.QFormLayout()

        self.reconstruction_size_txt = QtWidgets.QLineEdit("256")
        layout.addRow("Reconstruction size N", self.reconstruction_size_txt)

        self.reconstruction_offset_x = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset x", self.reconstruction_offset_x)

        self.reconstruction_offset_y = QtWidgets.QLineEdit("0")
        layout.addRow("Reconstruction offset y", self.reconstruction_offset_y)

        self.reconstruction_eta_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Reconstruction eta", self.reconstruction_eta_txt)

        self.use_filter_chb = QtWidgets.QCheckBox("Use frequency space filtering")
        layout.addRow("Filter", self.use_filter_chb)

        reconstruction_group.setLayout(layout)

        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(recording_group)
        hlayout.addWidget(reconstruction_group)
        layout.addLayout(hlayout)


        cameraMenu = self.menuBar().addMenu("&Camera")
        connect_camera_action = QtWidgets.QAction("&Connect Camera", self)
        connect_camera_action.triggered.connect(self.connect_camera)
        cameraMenu.addAction(connect_camera_action)

        disconnect_camera_action = QtWidgets.QAction("&Disconnect Camera", self)
        disconnect_camera_action.triggered.connect(self.disconnect_camera)
        cameraMenu.addAction(disconnect_camera_action)

        patternMenu = self.menuBar().addMenu("&Pattern")
        update_pattern_action = QtWidgets.QAction("&Update Pattern", self)
        update_pattern_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_P))
        update_pattern_action.triggered.connect(self.create_patterns)
        patternMenu.addAction(update_pattern_action)

        project_pattern_loop_action = QtWidgets.QAction("Project &Pattern", self)
        project_pattern_loop_action.triggered.connect(self.project_pattern_loop)
        patternMenu.addAction(project_pattern_loop_action)

        measure_orientation_action = QtWidgets.QAction("&Measure Orientation", self)
        measure_orientation_action.triggered.connect(self.measure_orientation)
        patternMenu.addAction(measure_orientation_action)

        imageMenu = self.menuBar().addMenu("&Image")
        take_images_action = QtWidgets.QAction("Take &Images", self)
        take_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        take_images_action.triggered.connect(self.take_images)
        imageMenu.addAction(take_images_action)

        load_images_action = QtWidgets.QAction("&Open Images", self)
        load_images_action.setShortcut(QtGui.QKeySequence.Open)
        load_images_action.triggered.connect(self.load_images)
        imageMenu.addAction(load_images_action)

        save_images_action = QtWidgets.QAction("&Save Images", self)
        save_images_action.setShortcut(QtGui.QKeySequence.Save)
        save_images_action.triggered.connect(self.save_images)
        imageMenu.addAction(save_images_action)

        reconstructMenu = self.menuBar().addMenu("&Reconstruct")
        reconstruct_images_action = QtWidgets.QAction("&Reconstruct Images", self)
        reconstruct_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_R))
        reconstruct_images_action.triggered.connect(self.reconstruct_image)
        reconstructMenu.addAction(reconstruct_images_action)

        reconstruct_images_nocal_action = QtWidgets.QAction("Reconstruct (&No cal)", self)
        reconstruct_images_nocal_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_F))
        reconstruct_images_nocal_action.triggered.connect(self.reconstruct_image_nocal)
        reconstructMenu.addAction(reconstruct_images_nocal_action)

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

        self.recon_plot = PlotWidget(self, num_clim = 2)
        self.tab_widget.addTab(self.recon_plot, "Reconstructed Image")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.sim_system = SIMSystem()

        self.p = HexSimProcessor()
        self.p.debug = False

        self.frames = None
        self.metadata = None

    def connect_camera(self):
        self.sim_system.connect()

    def disconnect_camera(self):
        self.sim_system.disconnect()

    def create_patterns(self):
        desired_distance = float(self.desired_distance_txt.text())
        grating_distance_x = float(self.grating_distance_x_txt.text())
        steps_x = round(grating_distance_x / desired_distance)
        distance_x = grating_distance_x / steps_x

        grating_distance_y = float(self.grating_distance_y_txt.text())
        desired_distance_y = np.sin(np.deg2rad(60)) * distance_x * 2
        steps_y = round(grating_distance_y / desired_distance_y)
        distance_y = grating_distance_y / steps_y

        aspect = grating_distance_y / (desired_distance_y * steps_y)
        self.aspect_lbl.setText(str(aspect))

        self.steps_x_lbl.setText(str(steps_x))
        self.steps_y_lbl.setText(str(steps_y))
        self.dot_distance_x_lbl.setText(str(distance_x))

        orientation_deg = float(self.orientation_deg_txt.text())
        self.pattern_rate_Hz = float(self.pattern_hz_txt.text())

        pattern_deg = hex_grid.projection_hex_pattern_deg(distance_x, steps_x, steps_y, orientation_rad = np.deg2rad(orientation_deg), aspect_ratio=aspect)

        self.exposure_time_sec = pattern_deg.shape[1] / self.pattern_rate_Hz
        self.exposure_lbl.setText(f"{self.exposure_time_sec * 1e3:.1f} ms * 7 = {self.exposure_time_sec * 1e3 * 7:.1f} ms")

        self.pattern_plot.fig.clear()
        ax1, ax2 = self.pattern_plot.fig.subplots(1, 2, sharex=True, sharey=True)
        for i in range(7):
            ax1.scatter(pattern_deg[i, :, 0], pattern_deg[i, :, 1])
        ax2.scatter(pattern_deg[0, :, 0], pattern_deg[0, :, 1])
        ax1.set_aspect(1)
        ax2.set_aspect(1)
        self.pattern_plot.draw()

        self.pattern_deg = pattern_deg


    def take_images(self):
        self.create_patterns()
        self.frames = self.sim_system.project_patterns_and_take_images(self.pattern_deg, self.pattern_rate_Hz)

        self.metadata = {
            "orientation_deg": float(self.orientation_deg_txt.text()),
            "pattern_rate_Hz": float(self.pattern_hz_txt.text()),
            "desired_distance": float(self.desired_distance_txt.text()),
            "grating_distance_x": float(self.grating_distance_x_txt.text()),
            "grating_distance_y": float(self.grating_distance_y_txt.text()),
            "software_version": get_git_revision_short_hash(),
        }
        self.plot_images()

    def measure_orientation(self):
        pattern_deg = np.zeros((7, 500, 2))
        self.frames = self.sim_system.project_patterns_and_take_images(pattern_deg, 1000)
        self.plot_images()

        orientation = DetectOrientationDialog.run_calibration_dialog(self.frames[0], self)
        if orientation is not None:
            self.orientation_deg_txt.setText(str(orientation))

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

            self.metadata["recording_notes"] = self.image_notes_txt.text()
            with open(f"{Path.joinpath(path.parent, path.stem)}_metadata.json", "w") as f:
                json.dump(self.metadata, f)

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

        self.p.use_filter = self.use_filter_chb.isChecked()
        reconstruct = self.p.reconstruct_fftw(frames)

        self.recon_plot.plot.fig.clear()
        ax1, ax2 = self.recon_plot.plot.fig.subplots(1, 2, sharex=True, sharey=True)
        im1 = ax1.imshow(reconstruct / 4)
        im2 = ax2.imshow(scipy.ndimage.zoom(np.sum(frames, axis=0), (2, 2), order=1))
        self.recon_plot.connect_clim(im1, 0)
        self.recon_plot.connect_clim(im2, 1)
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
