from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

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


class ImageView(QtWidgets.QGraphicsView):
    color_palette = [QtGui.qRgb(i, i, i) for i in range(256)]

    def __init__(self):
        super(ImageView, self).__init__()
        self.graphics_scene = QtWidgets.QGraphicsScene()
        self.setScene(self.graphics_scene)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(200, 200, 200)))

        self.text_item = None

        self.pix = None
        self.pixmap_graphics_item = None

    def fit_view_to_pixmap(self):
        if self.pix is not None:
            s = self.pix.size()
            self.fitInView(QtCore.QRectF(0, 0, s.width(), s.height()), QtCore.Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        self.fit_view_to_pixmap()
        super(ImageView, self).resizeEvent(event)

    def clear_display(self):
        self.graphics_scene.clear()
        self.pix = None

    def set_pixmap(self, pixmap):
        self.pix = pixmap
        if not self.pixmap_graphics_item is None:
            self.graphics_scene.removeItem(self.pixmap_graphics_item)
        self.pixmap_graphics_item = self.graphics_scene.addPixmap(self.pix)
        self.setSceneRect(self.pixmap_graphics_item.boundingRect())
        self.fit_view_to_pixmap()

    def set_centered_text(self, text):
        # TODO Center the text
        self.clear_centered_text()
        self.text_item = self.graphics_scene.addText(text, font=QtWidgets.QFont('Helvetica', 40))
        self.text_item.setDefaultTextColor(QtGui.QColor(255, 0, 0))
        self.text_item.setZValue(10)

    def clear_centered_text(self):
        if not self.text_item is None:
            self.graphics_scene.removeItem(self.text_item)
            self.text_item = None

    def set_numpy_array(self, a):
        assert a.dtype == np.uint8
        if len(a.shape) == 2:
            qimg = QtGui.QImage(a, a.shape[1], a.shape[0], QtGui.QImage.Format_Indexed8)
            qimg.setColorTable(self.color_palette)
        elif len(a.shape) == 3:
            qimg = QtGui.QImage(a, a.shape[1], a.shape[0], QtGui.QImage.Format_RGB888)
        self.set_pixmap(QtGui.QPixmap.fromImage(qimg))

