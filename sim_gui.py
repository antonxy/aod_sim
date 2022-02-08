from PySide2 import QtCore, QtWidgets, QtGui
import hex_grid
import numpy as np
import time
import scipy
import threading
import tifffile
from pathlib import Path
from detect_orientation_dialog import DetectOrientationDialog
import subprocess
import json

from widgets import PlotWidget

import sys
import imaging_method

simulate = len(sys.argv) > 1 and sys.argv[1] == 'sim'
if simulate:
    from sim_simulated_system import SIMSimulatedSystem as SIMSystem
else:
    from sim_hardware_system import SIMHardwareSystem as SIMSystem


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'describe', '--always', '--dirty', '--tags']).decode('ascii').strip()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Hex SIM GUI')

        self.sim_imaging = imaging_method.SIMImaging()
        self.lmi_imaging = imaging_method.LMIImaging()

        recording_group = QtWidgets.QGroupBox("Recording")
        layout = QtWidgets.QFormLayout()
        self.orientation_deg_txt = QtWidgets.QLineEdit("0")
        layout.addRow("Orientation [deg]", self.orientation_deg_txt)

        self.grating_distance_x_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance x [deg]", self.grating_distance_x_txt)

        self.grating_distance_y_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance y [deg]", self.grating_distance_y_txt)

        self.pattern_delay_txt = QtWidgets.QLineEdit("0.0")
        layout.addRow("Pattern delay [sec]", self.pattern_delay_txt)

        self.image_notes_txt = QtWidgets.QLineEdit("")
        layout.addRow("Recording notes", self.image_notes_txt)

        recording_group.setLayout(layout)


        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(recording_group)
        hlayout.addWidget(self.sim_imaging.parameters_widget)
        hlayout.addWidget(self.lmi_imaging.parameters_widget)
        hlayout.addWidget(self.sim_imaging.reconstruction_parameters_widget)
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

        project_zero_pattern_loop_action = QtWidgets.QAction("Project &Zero Pattern", self)
        project_zero_pattern_loop_action.triggered.connect(self.project_zero_pattern_loop)
        patternMenu.addAction(project_zero_pattern_loop_action)

        project_single_pattern_loop_action = QtWidgets.QAction("Project &Single Pattern", self)
        project_single_pattern_loop_action.triggered.connect(self.project_single_pattern_loop)
        patternMenu.addAction(project_single_pattern_loop_action)

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

        self.tab_widget.addTab(self.sim_imaging.patterns_widget, "SIM Pattern")
        self.tab_widget.addTab(self.lmi_imaging.patterns_widget, "LMI Pattern")
        for name, widget in self.sim_imaging.debug_tabs:
            self.tab_widget.addTab(widget, name)
        for name, widget in self.lmi_imaging.debug_tabs:
            self.tab_widget.addTab(widget, name)

        self.recon_plot = PlotWidget(None, num_clim = 3)
        self.tab_widget.addTab(self.recon_plot, "All Reconstructions")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.sim_system = SIMSystem()

        self.frames = None
        self.metadata = None

    def connect_camera(self):
        self.sim_system.connect()

    def disconnect_camera(self):
        self.sim_system.disconnect()

    def parse_global_params(self):
        global_params = {
            "grating_distance_x": float(self.grating_distance_x_txt.text()),
            "grating_distance_y": float(self.grating_distance_y_txt.text()),
            "orientation_deg": float(self.orientation_deg_txt.text()),
            "pattern_delay_sec": float(self.pattern_delay_txt.text()),
        }
        return global_params

    def create_patterns(self):
        self.sim_imaging.update_patterns(global_params = self.parse_global_params())
        self.lmi_imaging.update_patterns(global_params = self.parse_global_params())

    def take_images(self):
        self.create_patterns()
        self.sim_imaging.take_images(self.sim_system)
        self.lmi_imaging.take_images(self.sim_system)
        self.wf_image = self.sim_system.take_widefield_image()

    def measure_orientation(self):
        pattern_deg = np.zeros((1, 300, 2))
        frames = self.sim_system.project_patterns_and_take_images(pattern_deg, 10000, delay_sec = 0.0)

        orientation = DetectOrientationDialog.run_calibration_dialog(frames[0], self)
        if orientation is not None:
            self.orientation_deg_txt.setText(str(orientation))

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
        self.sim_imaging.calibrate()
        self.lmi_imaging.calibrate()
        self.reconstruct_image_nocal()

    def reconstruct_image_nocal(self):
        sim_image, sim_wf = self.sim_imaging.reconstruct()
        lmi_image, lmi_wf = self.lmi_imaging.reconstruct()

        self.recon_plot.plot.fig.clear()
        ax1, ax2, ax3 = self.recon_plot.plot.fig.subplots(1, 3, sharex=True, sharey=True)

        if sim_image is not None:
            im1 = ax1.imshow(sim_image)
            ax1.set_title("Hex SIM")
            self.recon_plot.connect_clim(im1, 0)
        if lmi_image is not None:
            im2 = ax2.imshow(lmi_image)
            ax2.set_title("LMI")
            self.recon_plot.connect_clim(im2, 1)
        if self.wf_image is not None:
            wf_upres = scipy.ndimage.zoom(self.wf_image, (2, 2), order=2)
            ax3.set_title("WF")
            im3 = ax3.imshow(wf_upres)
            self.recon_plot.connect_clim(im3, 2)
        elif sim_wf is not None:
            ax3.set_title("WF (sum of SIM images)")
            im3 = ax3.imshow(sim_wf)
            self.recon_plot.connect_clim(im3, 2)
        elif lmi_wf is not None:
            ax3.set_title("WF (sum of LMI images)")
            im3 = ax3.imshow(lmi_wf)
            self.recon_plot.connect_clim(im3, 2)

        self.recon_plot.plot.draw()

    def project_pattern_loop(self):
        self.create_patterns()
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target = self.sim_system.project_patterns_looping, args = (self.sim_imaging.pattern_deg, self.sim_imaging.pattern_rate_Hz, run_event))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Projecting pattern. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()

    def project_single_pattern_loop(self):
        self.create_patterns()
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target = self.sim_system.project_patterns_looping, args = (self.sim_imaging.pattern_deg[0, :, :], self.sim_imaging.pattern_rate_Hz, run_event))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Projecting single pattern. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()

    def project_zero_pattern_loop(self):
        pattern_deg = np.zeros((1, 500, 2))
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target = self.sim_system.project_patterns_looping, args = (pattern_deg, 500, run_event))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Projecting zero pattern. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()
