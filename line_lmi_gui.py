from PySide2 import QtCore, QtWidgets, QtGui
import hex_grid
import numpy as np
import time
import scipy
import threading
from pathlib import Path
import subprocess
import os
from datetime import datetime
import re
import argparse
import json
from measure_line_gratings import MeasureLineGratingsDialog

from widgets import PlotWidget

import sys
import imaging_method
import nidaq_pattern

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
        self.setWindowTitle('Line LMI GUI')

        self.line_lmi_imaging = imaging_method.LineLMIImaging()

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

        self.grating_dot_distance_1_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance 1 [deg]", self.grating_dot_distance_1_txt)

        self.grating_dot_distance_2_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance 2 [deg]", self.grating_dot_distance_2_txt)

        self.grating_dot_distance_3_txt = QtWidgets.QLineEdit("0.5")
        layout.addRow("Grating dot distance 3 [deg]", self.grating_dot_distance_3_txt)
        self.grating_dot_distance_txts = [
            self.grating_dot_distance_1_txt,
            self.grating_dot_distance_2_txt,
            self.grating_dot_distance_3_txt
        ]

        self.distance_between_gratings_txt = QtWidgets.QLineEdit("1")
        layout.addRow("Distance between gratings [deg]", self.distance_between_gratings_txt)

        self.single_aod_chb = QtWidgets.QCheckBox("Single AOD")
        layout.addRow("Single AOD", self.single_aod_chb)

        self.capture_repeats_txt = QtWidgets.QLineEdit("1000")
        layout.addRow("Capture repeats", self.capture_repeats_txt)
        
        self.stage_position_txt = QtWidgets.QLineEdit("0.0")
        layout.addRow("Stage position [mm]", self.stage_position_txt)
        
        self.stage_position_increment_txt = QtWidgets.QLineEdit("0.0")
        layout.addRow("Stage position increment [mm]", self.stage_position_increment_txt)

        optical_group.setLayout(layout)


        layout = QtWidgets.QVBoxLayout()
        hlayout = QtWidgets.QHBoxLayout()
        hlayout.addWidget(recording_group)
        hlayout.addWidget(optical_group)
        hlayout.addWidget(self.line_lmi_imaging.parameters_widget)
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

        set_camera_settings_action = QtWidgets.QAction("Set Camera &Settings", self)
        set_camera_settings_action.triggered.connect(self.set_camera_settings)
        cameraMenu.addAction(set_camera_settings_action)
        
        stageMenu = self.menuBar().addMenu("&Stage")
        connect_stage_action = QtWidgets.QAction("&Connect Stage", self)
        connect_stage_action.triggered.connect(self.connect_stage)
        stageMenu.addAction(connect_stage_action)

        disconnect_stage_action = QtWidgets.QAction("&Disconnect Stage", self)
        disconnect_stage_action.triggered.connect(self.disconnect_stage)
        stageMenu.addAction(disconnect_stage_action)
        
        home_stage_action = QtWidgets.QAction("&Home Stage", self)
        home_stage_action.triggered.connect(self.home_stage)
        stageMenu.addAction(home_stage_action)
        
        set_stage_position_action = QtWidgets.QAction("&Set Stage Position", self)
        set_stage_position_action.triggered.connect(self.set_stage_position)
        set_stage_position_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_T))
        stageMenu.addAction(set_stage_position_action)

        patternMenu = self.menuBar().addMenu("&Pattern")
        update_pattern_action = QtWidgets.QAction("&Update Pattern", self)
        update_pattern_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_P))
        update_pattern_action.triggered.connect(self.create_patterns)
        patternMenu.addAction(update_pattern_action)

        project_pattern_loop_action = QtWidgets.QAction("Project &Pattern", self)
        project_pattern_loop_action.triggered.connect(self.project_pattern_loop)
        patternMenu.addAction(project_pattern_loop_action)
        
        project_zero_pattern_loop_action = QtWidgets.QAction("Project &Zero Pattern", self)
        project_zero_pattern_loop_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_L))
        project_zero_pattern_loop_action.triggered.connect(self.project_zero_pattern_loop)
        patternMenu.addAction(project_zero_pattern_loop_action)
        
        project_pattern_loopv_action = QtWidgets.QAction("Project Pattern &Video", self)
        project_pattern_loopv_action.triggered.connect(self.project_pattern_loopv)
        patternMenu.addAction(project_pattern_loopv_action)

        measure_grating_action = QtWidgets.QAction("Measure &Grating", self)
        measure_grating_action.triggered.connect(self.measure_grating)
        patternMenu.addAction(measure_grating_action)

        save_pattern_action = QtWidgets.QAction("&Save Pattern", self)
        save_pattern_action.triggered.connect(self.save_pattern)
        patternMenu.addAction(save_pattern_action)

        imageMenu = self.menuBar().addMenu("&Image")
        take_images_action = QtWidgets.QAction("Take &Images", self)
        take_images_action.setShortcut(QtGui.QKeySequence(QtGui.Qt.CTRL | QtGui.Qt.Key_Return))
        take_images_action.triggered.connect(self.take_images)
        imageMenu.addAction(take_images_action)
        
        capture_zstack_action = QtWidgets.QAction("Capture &Z-Stack", self)
        capture_zstack_action.triggered.connect(self.capture_zstack)
        imageMenu.addAction(capture_zstack_action)

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

        self.tab_widget = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tab_widget)

        self.tab_widget.addTab(self.line_lmi_imaging.patterns_widget, "LMI Pattern")
        for name, widget in self.line_lmi_imaging.debug_tabs:
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
        
        self.stage = None

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

    def set_camera_settings(self):
        self.create_patterns()
        connect = self.sim_system.camera is None

        if connect:
            self.sim_system.connect()

        self.sim_system.project_patterns_and_take_images(self.line_lmi_imaging.pattern_deg, self.line_lmi_imaging.pattern_rate_Hz, self.line_lmi_imaging.params['pattern_delay_sec'], only_configure=True, single_aod = self.line_lmi_imaging.params['single_aod'])

        if connect:
            self.sim_system.disconnect()
    
    def connect_stage(self):
        if self.stage is None:
            import kinesis
            self.stage = kinesis.Stage(kinesis.stage_serial_number)
        self.stage.connect()

    def disconnect_stage(self):
        self.stage.disconnect()
    
    def home_stage(self):
        self.stage.home()
    
    def set_stage_position(self):
        self.stage.move_to(float(self.stage_position_txt.text()))

    def parse_global_params(self):
        return {
            "grating_dot_distances": [float(txt.text()) for txt in self.grating_dot_distance_txts],
            "orientation_deg": float(self.orientation_deg_txt.text()),
            "distance_between_gratings": float(self.distance_between_gratings_txt.text()),
            "pattern_delay_sec": float(self.pattern_delay_txt.text()),
            "recording_notes": self.image_notes_txt.text(),
            "software_version": get_git_revision_short_hash(),
            "date_time": datetime.now().astimezone().isoformat(),
            "stage_position": float(self.stage_position_txt.text()),
            "stage_position_increment": float(self.stage_position_increment_txt.text()),
            "single_aod": self.single_aod_chb.isChecked(),
        }

    # This is for parameters which should not be saved as part of image metadata
    def parse_settings(self):
        return {
            "output_folder": self.output_folder_txt.text(),
            "recording_name": self.recording_name_txt.text(),
            "capture_repeats": int(self.capture_repeats_txt.text()),
        }

    def save_settings_action(self, *args, file = None):
        if file is None:
            file = QtWidgets.QFileDialog.getSaveFileName(self, "Save settings", filter="JSON (*.json)")[0]
        if file is not None and file != "":
            with open(file, 'w') as f:
                params = {
                    "global": {**self.parse_global_params(), **self.parse_settings()},
                    "line_lmi": self.line_lmi_imaging.parse_parameters()
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

        for txt, val in zip(self.grating_dot_distance_txts, global_params.get("grating_dot_distances", [0.27] * 3)):
            txt.setText(str(val))
        self.orientation_deg_txt.setText(str(global_params.get("orientation_deg", "0")))
        self.distance_between_gratings_txt.setText(str(global_params.get("distance_between_gratings", "1")))
        self.single_aod_chb.setChecked(global_params.get("single_aod", False))
        self.capture_repeats_txt.setText(str(global_params.get("capture_repeats", "1000")))
        self.stage_position_txt.setText(str(global_params.get("stage_position", 0.0)))
        self.stage_position_increment_txt.setText(str(global_params.get("stage_position_increment", 0.0)))

        self.pattern_delay_txt.setText(str(global_params.get("pattern_delay_sec", "0")))
        self.image_notes_txt.setText(global_params.get("recording_notes", ""))

        self.line_lmi_imaging.load_parameters(params.get("line_lmi", {}))

    def save_pattern(self, *args, file = None):
        if file is None:
            file = QtWidgets.QFileDialog.getSaveFileName(self, "Save pattern", filter="CSV (*.csv)")[0]
        if file is not None and file != "":
            nidaq_pattern.save_pattern_as_csv(self.line_lmi_imaging.pattern_deg, file)

    def create_patterns(self):
        self.line_lmi_imaging.update_patterns(global_params = self.parse_global_params())

    def take_images(self):
        self.create_patterns()
        self.line_lmi_imaging.take_images(self.sim_system)
        self.wf_image = self.sim_system.take_widefield_image()

    def load_images(self, *args, folder = None):
        if folder is None:
            folder = self.output_folder_txt.text()
            file_dialog = QtWidgets.QFileDialog()
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Load recording dir", folder)
        if folder is not None and folder != "":
            self.line_lmi_imaging.load_images(folder)

    def increment_filename(self):
        res = re.sub(r'[0-9]+$',
             lambda x: f"{str(int(x.group())+1).zfill(len(x.group()))}",
             self.recording_name_txt.text())
        self.recording_name_txt.setText(res)

    def measure_grating(self):
        params = MeasureLineGratingsDialog.run_dialog(self.sim_system, self.parse_global_params(), self)
        if params is not None:
            for txt, val in zip(self.grating_dot_distance_txts, params["grating_dot_distances"]):
                txt.setText(str(val))

    def save_images(self, *args, show_success=True):
        folder = self.output_folder_txt.text()
        rec_name = self.recording_name_txt.text()
        if folder != "" and rec_name != "":
            rec_folder = os.path.join(folder, rec_name)

            if not os.path.exists(rec_folder):
                os.makedirs(rec_folder)
                self.line_lmi_imaging.save_images(rec_folder)
                self.save_settings_action(file = os.path.join(rec_folder, "settings.json"))
                if show_success:
                    QtWidgets.QMessageBox.information(self, "Success", "Saved successfully")
                self.increment_filename()
            else:
                QtWidgets.QMessageBox.critical(self, "Error", "Directory already exists, not saving")

    def reconstruct_image(self):
        pass
        
    def project_zero_pattern_loop(self):
        pattern_deg = np.zeros((1, 500, 2))
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target = self.sim_system.project_patterns_looping, args = (pattern_deg, 500, run_event, self.single_aod_chb.isChecked()))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Projecting zero pattern. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()

    def project_pattern_loop(self):
        self.create_patterns()
        run_event = threading.Event()
        run_event.set()
        thread = threading.Thread(target = self.sim_system.project_patterns_looping, args = (self.line_lmi_imaging.pattern_deg, self.line_lmi_imaging.pattern_rate_Hz, run_event, self.single_aod_chb.isChecked()))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText("Projecting pattern. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()

    def project_pattern_loopv(self):
        capture_repeats = int(self.capture_repeats_txt.text())
        self.create_patterns()
        run_event = threading.Event()
        run_event.set()
        pattern = self.line_lmi_imaging.pattern_deg.reshape(-1, 2)
        pattern = np.tile(pattern, [capture_repeats, 1])
        thread = threading.Thread(target = self.sim_system.project_patterns_video, args = (pattern, self.line_lmi_imaging.pattern_rate_Hz, run_event, self.single_aod_chb.isChecked()))
        thread.start()
        msgBox = QtWidgets.QMessageBox()
        msgBox.setText(f"Projecting pattern {capture_repeats} times. Close dialog to stop")
        msgBox.exec()
        run_event.clear()
        thread.join()
        
    def capture_zstack(self):
        run_event = threading.Event()
        run_event.set()
        
        #msgBox = QtWidgets.QMessageBox(parent=self)
        #msgBox.setText(f"Capturing z-stack. Close dialog to stop")
        #msgBox.show()
        
        stage_pos = float(self.stage_position_txt.text())
        increment = float(self.stage_position_increment_txt.text())
        while True:
            print(f"Z-Stack pos: {stage_pos}")
            self.stage.move_to(stage_pos)
            #QtCore.QCoreApplication.processEvents()
            self.take_images()
            #QtCore.QCoreApplication.processEvents()
            self.save_images(show_success=False)
            #QtCore.QCoreApplication.processEvents()
            stage_pos += increment
            #print(msgBox.result())
                
        

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec_()
