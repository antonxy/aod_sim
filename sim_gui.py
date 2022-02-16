from PySide2 import QtCore, QtWidgets, QtGui
import hex_grid
import numpy as np
import time
import scipy
import threading
from pathlib import Path
from detect_orientation_dialog import DetectOrientationDialog
from measure_grating import MeasureGratingDialog
import subprocess
import os
from datetime import datetime
import re
import argparse
import json

from widgets import PlotWidget

import sys
import imaging_method

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder')
parser.add_argument('-s', '--settings')
parser.add_argument('--simulate', action='store_true')
args = parser.parse_args()

if args.simulate:
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
        self.pattern_delay_txt = QtWidgets.QLineEdit("0.0")
        layout.addRow("Pattern delay [sec]", self.pattern_delay_txt)

        self.output_folder_txt = QtWidgets.QLineEdit("./recordings")
        layout.addRow("Output folder", self.output_folder_txt)

        self.recording_name_txt = QtWidgets.QLineEdit("")
        layout.addRow("Recording name", self.recording_name_txt)

        self.image_notes_txt = QtWidgets.QLineEdit("")
        layout.addRow("Recording notes", self.image_notes_txt)

        recording_group.setLayout(layout)

        optical_group = QtWidgets.QGroupBox("Optical Parameters")
        layout = QtWidgets.QFormLayout()

        self.orientation_deg_txt = QtWidgets.QLineEdit("0")
        layout.addRow("Orientation [deg]", self.orientation_deg_txt)

        self.grating_distance_x_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance x [deg]", self.grating_distance_x_txt)

        self.grating_distance_y_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance y [deg]", self.grating_distance_y_txt)

        self.aod_deg_to_um_in_sample_plane = QtWidgets.QLineEdit("1483")
        layout.addRow("Shift in sample plane per AOD angle[um / deg]", self.aod_deg_to_um_in_sample_plane)

        self.pixelsize_txt = QtWidgets.QLineEdit("11")
        layout.addRow("Pixelsize [um]", self.pixelsize_txt)

        self.magnification_txt = QtWidgets.QLineEdit("1.5")
        layout.addRow("Magnification", self.magnification_txt)

        self.ex_na_txt = QtWidgets.QLineEdit("0.02")
        layout.addRow("Excitation NA", self.ex_na_txt)

        self.em_na_txt = QtWidgets.QLineEdit("0.02")
        layout.addRow("Emission NA", self.em_na_txt)

        self.ex_wavelength_txt = QtWidgets.QLineEdit("0.660")
        layout.addRow("Excitation Wavelength [um]", self.ex_wavelength_txt)

        self.em_wavelength_txt = QtWidgets.QLineEdit("0.680")
        layout.addRow("Emission Wavelength [um]", self.em_wavelength_txt)

        self.mtf_data_txt = QtWidgets.QTextEdit("")
        self.mtf_data_txt.setToolTip("Measured MTF in camera plane. TSV with header \"lp/mm\tModulation Factor\"")
        layout.addRow("MTF data", self.mtf_data_txt)

        optical_group.setLayout(layout)


        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(recording_group)
        hlayout.addWidget(optical_group)
        hlayout.addWidget(self.sim_imaging.parameters_widget)
        hlayout.addWidget(self.lmi_imaging.parameters_widget)
        hlayout.addWidget(self.sim_imaging.reconstruction_parameters_widget)
        layout.addLayout(hlayout)

        appMenu = self.menuBar().addMenu("&Application")

        load_settings_action = QtWidgets.QAction("&Load Settings", self)
        load_settings_action.triggered.connect(self.load_settings_action)
        appMenu.addAction(load_settings_action)

        save_settings_action = QtWidgets.QAction("&Save Settings", self)
        save_settings_action.triggered.connect(self.save_settings_action)
        appMenu.addAction(save_settings_action)

        close_action = QtWidgets.QAction("&Quit", self)
        close_action.setShortcut(QtGui.QKeySequence.Quit)
        close_action.triggered.connect(self.close)
        appMenu.addAction(close_action)

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
        project_zero_pattern_loop_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_L))
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

        measure_grating_action = QtWidgets.QAction("Measure &Grating", self)
        measure_grating_action.triggered.connect(self.measure_grating)
        patternMenu.addAction(measure_grating_action)

        imageMenu = self.menuBar().addMenu("&Image")
        take_images_action = QtWidgets.QAction("Take &Images", self)
        take_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        take_images_action.triggered.connect(self.take_images)
        imageMenu.addAction(take_images_action)

        record_slide_action = QtWidgets.QAction("Take Slide Image", self)
        record_slide_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_N))
        record_slide_action.triggered.connect(self.record_slide)
        imageMenu.addAction(record_slide_action)

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

        self.recon_fft_plot = PlotWidget(None, num_clim = 3)
        self.tab_widget.addTab(self.recon_fft_plot, "Reconstruction FFTs")

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.sim_system = SIMSystem()

        self.frames = None
        self.wf_image = None
        self.metadata = None

        if args.folder:
            self.load_images(folder = args.folder)
        if args.settings:
            self.load_settings_action(file = args.settings)
        else:
            self.load_settings({})

    def connect_camera(self):
        self.sim_system.connect()

    def disconnect_camera(self):
        self.sim_system.disconnect()

    def parse_global_params(self):
        return {
            "grating_distance_x": float(self.grating_distance_x_txt.text()),
            "grating_distance_y": float(self.grating_distance_y_txt.text()),
            "orientation_deg": float(self.orientation_deg_txt.text()),
            "pattern_delay_sec": float(self.pattern_delay_txt.text()),
            "aod_deg_to_um_in_sample_plane": float(self.aod_deg_to_um_in_sample_plane.text()),
            "pixelsize": float(self.pixelsize_txt.text()),
            "magnification": float(self.magnification_txt.text()),
            "ex_NA": float(self.ex_na_txt.text()),
            "em_NA": float(self.em_na_txt.text()),
            "ex_wavelength": float(self.ex_wavelength_txt.text()),
            "em_wavelength": float(self.em_wavelength_txt.text()),
            "mtf_data": self.mtf_data_txt.toPlainText(),
            "recording_notes": self.image_notes_txt.text(),
            "software_version": get_git_revision_short_hash(),
            "date_time": datetime.now().astimezone().isoformat(),
        }

    # This is for parameters which should not be saved as part of image metadata
    def parse_settings(self):
        return {
            "output_folder": self.output_folder_txt.text(),
            "recording_name": self.recording_name_txt.text(),
        }

    def save_settings_action(self, *args, file = None):
        if file is None:
            file = QtWidgets.QFileDialog.getSaveFileName(self, "Save settings", filter="JSON (*.json)")[0]
        if file is not None and file != "":
            with open(file, 'w') as f:
                params = {
                    "global": {**self.parse_global_params(), **self.parse_settings()},
                    "sim": self.sim_imaging.parse_parameters(),
                    "lmi": self.lmi_imaging.parse_parameters()
                }
                json.dump(params, f)

    def load_settings_action(self, *args, file = None):
        if file is None:
            file = QtWidgets.QFileDialog.getOpenFileName(self, "Load settings", filter="JSON (*.json)")[0]
        if file is not None and file != "":
            with open(file, 'r') as f:
                params = json.load(f)
                self.load_settings(params)

    def load_settings(self, params):
        global_params = params.get("global", {})
        self.output_folder_txt.setText(global_params.get("output_folder", "../recordings"))
        self.recording_name_txt.setText(global_params.get("recording_name", "rec001"))

        self.grating_distance_x_txt.setText(str(global_params.get("grating_distance_x", "0.27")))
        self.grating_distance_y_txt.setText(str(global_params.get("grating_distance_y", "0.27")))
        self.orientation_deg_txt.setText(str(global_params.get("orientation_deg", "0")))
        self.pattern_delay_txt.setText(str(global_params.get("pattern_delay_sec", "0")))
        self.aod_deg_to_um_in_sample_plane.setText(str(global_params.get("aod_deg_to_um_in_sample_plane", "1483")))
        self.pixelsize_txt.setText(str(global_params.get("pixelsize", "11")))
        self.magnification_txt.setText(str(global_params.get("magnification", "1.5")))
        self.ex_na_txt.setText(str(global_params.get("ex_NA", "0.025")))
        self.em_na_txt.setText(str(global_params.get("em_NA", "0.025")))
        self.ex_wavelength_txt.setText(str(global_params.get("ex_wavelength", "0.660")))
        self.em_wavelength_txt.setText(str(global_params.get("em_wavelength", "0.680")))
        self.mtf_data_txt.setText(global_params.get("mtf_data", ""))
        self.image_notes_txt.setText(global_params.get("recording_notes", ""))

        self.sim_imaging.load_parameters(params.get("sim", {}))
        self.lmi_imaging.load_parameters(params.get("lmi", {}))

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

    def measure_grating(self):
        params = MeasureGratingDialog.run_dialog(self.sim_system, self.parse_global_params(), self)
        if params is not None:
            self.orientation_deg_txt.setText(str(params['orientation_deg']))
            self.grating_distance_x_txt.setText(str(params['grating_distance_x']))
            self.grating_distance_y_txt.setText(str(params['grating_distance_y']))

    def load_images(self, *args, folder = None):
        if folder is None:
            folder = self.output_folder_txt.text()
            file_dialog = QtWidgets.QFileDialog()
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Load recording dir", folder)
        if folder is not None and folder != "":
            self.sim_imaging.load_images(folder)
            self.lmi_imaging.load_images(folder)

    def increment_filename(self):
        res = re.sub(r'[0-9]+$',
             lambda x: f"{str(int(x.group())+1).zfill(len(x.group()))}",
             self.recording_name_txt.text())
        self.recording_name_txt.setText(res)

    def record_slide(self):
        self.project_zero_pattern_loop()
        self.take_images()
        self.reconstruct_image()

    def save_images(self):
        folder = self.output_folder_txt.text()
        rec_name = self.recording_name_txt.text()
        if folder != "" and rec_name != "":
            rec_folder = os.path.join(folder, rec_name)

            if not os.path.exists(rec_folder):
                os.makedirs(rec_folder)
                self.sim_imaging.save_images(rec_folder)
                self.lmi_imaging.save_images(rec_folder)
                self.save_settings_action(file = os.path.join(rec_folder, "settings.json"))
                QtWidgets.QMessageBox.information(self, "Success", "Saved successfully")
                self.increment_filename()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Directory already exists, not saving")

    def reconstruct_image(self):
        self.sim_imaging.calibrate()
        self.lmi_imaging.calibrate()
        self.reconstruct_image_nocal()

    def reconstruct_image_nocal(self):
        sim_image, sim_wf = self.sim_imaging.reconstruct()
        lmi_image, lmi_wf = self.lmi_imaging.reconstruct()

        self.recon_plot.plot.fig.clear()
        ax1, ax2, ax3 = self.recon_plot.plot.fig.subplots(1, 3, sharex=True, sharey=True)
        self.recon_fft_plot.plot.fig.clear()
        ax1f, ax2f, ax3f = self.recon_fft_plot.plot.fig.subplots(1, 3, sharex=True, sharey=True)

        if sim_image is not None:
            im1 = ax1.imshow(sim_image)
            ax1.set_title("Hex SIM")
            self.recon_plot.connect_clim(im1, 0)

            im1 = ax1f.imshow(np.fft.fftshift(np.abs(np.fft.fft2(sim_image))))
            ax1f.set_title("Hex SIM")
            self.recon_fft_plot.connect_clim(im1, 0)

        if lmi_image is not None:
            lmi_image = scipy.ndimage.zoom(lmi_image, (2, 2), order=2)
            im2 = ax2.imshow(lmi_image)
            ax2.set_title("LMI")
            self.recon_plot.connect_clim(im2, 1)

            im2 = ax2f.imshow(np.fft.fftshift(np.abs(np.fft.fft2(lmi_image))))
            ax2f.set_title("LMI")
            self.recon_fft_plot.connect_clim(im2, 1)

        if self.wf_image is not None:
            wf_upres = scipy.ndimage.zoom(self.wf_image, (2, 2), order=2)
            ax3.set_title("WF")
            im3 = ax3.imshow(wf_upres)
            self.recon_plot.connect_clim(im3, 2)

            im3 = ax3f.imshow(np.fft.fftshift(np.abs(np.fft.fft2(wf_upres))))
            ax3f.set_title("WF")
            self.recon_fft_plot.connect_clim(im3, 2)
        elif sim_wf is not None:
            ax3.set_title("WF (sum of SIM images)")
            im3 = ax3.imshow(sim_wf)
            self.recon_plot.connect_clim(im3, 2)

            im3 = ax3f.imshow(np.fft.fftshift(np.abs(np.fft.fft2(sim_wf))))
            ax3f.set_title("WF (sum of SIM images)")
            self.recon_fft_plot.connect_clim(im3, 2)
        elif lmi_wf is not None:
            ax3.set_title("WF (sum of LMI images)")
            lmi_wf = scipy.ndimage.zoom(lmi_wf, (2, 2), order=2)
            im3 = ax3.imshow(lmi_wf)
            self.recon_plot.connect_clim(im3, 2)

            im3 = ax3f.imshow(np.fft.fftshift(np.abs(np.fft.fft2(lmi_wf))))
            ax3f.set_title("WF (sum of LMI images)")
            self.recon_fft_plot.connect_clim(im3, 2)

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
