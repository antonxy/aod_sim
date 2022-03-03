from PySide2 import QtCore, QtGui, QtWidgets
import widgets
import numpy as np
import lmi_pattern


class MeasureLineGratingsDialog(QtWidgets.QDialog):
    def __init__(self, sim_system, params, parent=None):
        super().__init__(parent)


        self.setWindowFlags(QtCore.Qt.Window |
                            QtCore.Qt.WindowMaximizeButtonHint |
                            QtCore.Qt.WindowCloseButtonHint)
        self.setWindowTitle('Measure Line Gratings')

        layout = QtWidgets.QVBoxLayout()
        self.image_widget = widgets.ImageView()
        layout.addWidget(self.image_widget)

        form_layout = QtWidgets.QFormLayout()

        self.grating_dot_distance_1_txt = QtWidgets.QLineEdit(str(params['grating_dot_distances'][0]))
        form_layout.addRow("Grating dot distance 1 [deg]", self.grating_dot_distance_1_txt)

        self.grating_dot_distance_2_txt = QtWidgets.QLineEdit(str(params['grating_dot_distances'][1]))
        form_layout.addRow("Grating dot distance 2 [deg]", self.grating_dot_distance_2_txt)

        self.grating_dot_distance_3_txt = QtWidgets.QLineEdit(str(params['grating_dot_distances'][2]))
        form_layout.addRow("Grating dot distance 3 [deg]", self.grating_dot_distance_3_txt)
        self.grating_dot_distance_txts = [
            self.grating_dot_distance_1_txt,
            self.grating_dot_distance_2_txt,
            self.grating_dot_distance_3_txt
        ]

        layout.addLayout(form_layout)

        hlayout = QtWidgets.QHBoxLayout()
        ok_button = QtWidgets.QPushButton('OK')
        ok_button.pressed.connect(self.accept)
        hlayout.addWidget(ok_button)
        cancel_button = QtWidgets.QPushButton('Cancel')
        cancel_button.pressed.connect(self.reject)
        hlayout.addWidget(cancel_button)
        layout.addLayout(hlayout)

        self.setLayout(layout)

        self.global_params = params
        self.sim_system = sim_system

        self.take_image()


    def get_params(self):
        return {
            "grating_dot_distances": [float(txt.text()) for txt in self.grating_dot_distance_txts],
        }

    def grating_pattern(self):
        params = self.get_params()
        grating_dot_distances = params['grating_dot_distances']
        orientation_deg = self.global_params['orientation_deg']
        distance_between_gratings = self.global_params["distance_between_gratings"]

        pattern_deg = lmi_pattern.line_lmi_pattern_deg(2, 1, grating_dot_distances, distance_between_gratings, orientation_rad=np.deg2rad(orientation_deg))

        pattern_deg = np.reshape(pattern_deg, (1, -1, 2))  # All steps in single frame

        return pattern_deg

    def take_image(self):
        pattern_deg = self.grating_pattern()
        pattern_deg = np.repeat(pattern_deg, 100, axis=1)

        frames = self.sim_system.project_patterns_and_take_images(pattern_deg, 10000, delay_sec = 0.0)
        self.adj_frame = frames[0]

        self.update_view()

    def update_view(self):
        themax = np.max(self.adj_frame[10:-10, 10:-10])
        adj_clip = (np.clip(self.adj_frame / themax * 255, 0, 255)).astype(np.uint8)
        self.frame_u8 = adj_clip
        self.image_widget.set_numpy_array(self.frame_u8)


    @staticmethod
    def run_dialog(sim_system, params, parent=None):
        dialog = MeasureLineGratingsDialog(sim_system, params, parent)
        result = dialog.exec_()
        if result > 0:
            return dialog.get_params()
        else:
            return None
