from PySide2 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import widgets
import numpy as np


class MeasureGratingDialog(QtWidgets.QDialog):
    def __init__(self, sim_system, params, parent=None):
        super(MeasureGratingDialog, self).__init__(parent)


        self.setWindowFlags(QtCore.Qt.Window |
                            QtCore.Qt.WindowMaximizeButtonHint |
                            QtCore.Qt.WindowCloseButtonHint)
        self.setWindowTitle('Measure Grating')

        layout = QtWidgets.QVBoxLayout()
        self.image_widget = widgets.ImageView()
        layout.addWidget(self.image_widget)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        self.quality_plot = widgets.MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.quality_plot, w)
        lay.addWidget(toolbar)
        lay.addWidget(self.quality_plot)
        w.setLayout(lay)
        layout.addWidget(w)


        form_layout = QtWidgets.QFormLayout()
        self.orientation_deg_txt = QtWidgets.QLineEdit(str(params['orientation_deg']))
        self.orientation_deg_txt.returnPressed.connect(self.take_image)
        form_layout.addRow("Orientation [deg]", self.orientation_deg_txt)

        self.grating_distance_x_txt = QtWidgets.QLineEdit(str(params['grating_distance_x']))
        self.grating_distance_x_txt.returnPressed.connect(self.take_image)
        form_layout.addRow("Grating dot distance x [deg]", self.grating_distance_x_txt)

        self.grating_distance_y_txt = QtWidgets.QLineEdit(str(params['grating_distance_y']))
        self.grating_distance_y_txt.returnPressed.connect(self.take_image)
        form_layout.addRow("Grating dot distance y [deg]", self.grating_distance_y_txt)

        hlayout = QtWidgets.QHBoxLayout()
        switch_button = QtWidgets.QPushButton('Switch image')
        switch_button.pressed.connect(self.switch_image)
        hlayout.addWidget(switch_button)
        ok_button = QtWidgets.QPushButton('OK')
        ok_button.pressed.connect(self.accept)
        hlayout.addWidget(ok_button)
        cancel_button = QtWidgets.QPushButton('Cancel')
        cancel_button.pressed.connect(self.reject)
        hlayout.addWidget(cancel_button)
        layout.addLayout(hlayout)

        self.setLayout(layout)

        self.sim_system = sim_system
        self.quality_hist = []

        #self.take_ref_image()
        self.take_image()
        self.show_adj_image = False


    def get_params(self):
        return {
            "grating_distance_x": float(self.grating_distance_x_txt.text()),
            "grating_distance_y": float(self.grating_distance_y_txt.text()),
            "orientation_deg": float(self.orientation_deg_txt.text()),
        }

    def take_ref_image(self):
        pattern_deg = np.zeros((1, 1000, 2))
        frames = self.sim_system.project_patterns_and_take_images(pattern_deg, 10000, delay_sec = 0.0)
        self.ref_frame = frames[0]

    def grating_pattern(self):
        params = self.get_params()
        grating_distance_x = params["grating_distance_x"]
        grating_distance_y = params["grating_distance_y"]
        orientation_deg = params["orientation_deg"]

        grating_repeats = 5
        grating_dots_x = np.linspace(-grating_distance_x * (grating_repeats - 1) / 2, grating_distance_x * (grating_repeats - 1) / 2, grating_repeats)
        grating_dots_y = np.linspace(-grating_distance_y * (grating_repeats - 1) / 2, grating_distance_y * (grating_repeats - 1) / 2, grating_repeats)
        dots_xx, dots_yy = np.meshgrid(grating_dots_x, grating_dots_y)
        dots_xy = np.zeros((grating_repeats, grating_repeats, 2))
        dots_xy[:, :, 0] = dots_xx
        dots_xy[:, :, 1] = dots_yy
        dots_xy = dots_xy.reshape((-1, 2))

        # Apply orientation
        orientation_rad = np.deg2rad(orientation_deg)
        c, s = np.cos(orientation_rad), np.sin(orientation_rad)
        R = np.array(((c, -s), (s, c)))
        dots_xy = np.dot(dots_xy, R.T)

        return dots_xy

    def take_image(self):
        #params = self.get_params()
        #pattern_deg = np.zeros((1, 2, 2))
        #pattern_deg[0, 0] = [params["grating_distance_x"], 0]
        #pattern_deg[0, 1] = [0, params["grating_distance_y"]]

        ## Apply orientation
        #orientation_rad = np.deg2rad(params["orientation_deg"])
        #c, s = np.cos(orientation_rad), np.sin(orientation_rad)
        #R = np.array(((c, -s), (s, c)))
        #pattern_deg = np.dot(pattern_deg, R.T)

        #pattern_deg = np.repeat(pattern_deg, 500, axis=1)
        pattern_deg = self.grating_pattern()[np.newaxis, :, :]

        frames = self.sim_system.project_patterns_and_take_images(pattern_deg, 10000, delay_sec = 0.0)
        self.adj_frame = frames[0]

        # Stddev is not a good metric
        q = np.std(self.adj_frame[10:-10, 10:-10])
        self.quality_hist.append(q)
        self.update_view()

    def switch_image(self):
        self.show_adj_image = not self.show_adj_image
        #self.update_view()

    def update_view(self):
        #frame = self.adj_frame if self.show_adj_image else self.ref_frame
        #self.frame_u8 = (np.clip(frame / frame.max() * 255, 0, 255)).astype(np.uint8)
        themax = np.max(self.adj_frame[10:-10, 10:-10])
        adj_clip = (np.clip(self.adj_frame / themax * 255, 0, 255)).astype(np.uint8)
        #ref_clip = (np.clip(self.ref_frame / themax * 255, 0, 255)).astype(np.uint8)
        #z = np.zeros(ref_clip.shape, dtype=np.uint8)
        #self.frame_u8 = np.stack([adj_clip, ref_clip, z], axis=-1)
        self.frame_u8 = adj_clip
        self.image_widget.set_numpy_array(self.frame_u8)

        self.quality_plot.axes.clear()
        self.quality_plot.axes.plot(self.quality_hist)
        self.quality_plot.draw()

    @staticmethod
    def run_dialog(sim_system, params, parent=None):
        dialog = MeasureGratingDialog(sim_system, params, parent)
        result = dialog.exec_()
        if result > 0:
            return dialog.get_params()
        else:
            return None
