from PySide2 import QtCore, QtGui, QtWidgets
import numpy as np

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

