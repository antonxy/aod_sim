from PySide2 import QtCore, QtGui, QtWidgets
import widgets
import numpy as np


class DetectOrientationDialog(QtWidgets.QDialog):
    def __init__(self, frame, parent=None):
        super(DetectOrientationDialog, self).__init__(parent)


        self.setWindowFlags(QtCore.Qt.Window |
                            QtCore.Qt.WindowMaximizeButtonHint |
                            QtCore.Qt.WindowCloseButtonHint)
        self.setWindowTitle('Detect Orientation')

        layout = QtWidgets.QVBoxLayout()
        self.image_widget = OrientationImageView()
        layout.addWidget(self.image_widget)

        hlayout = QtWidgets.QHBoxLayout()
        adjust_button = QtWidgets.QPushButton('Adjust Line')
        adjust_button.pressed.connect(self.adjust)
        hlayout.addWidget(adjust_button)
        ok_button = QtWidgets.QPushButton('OK')
        ok_button.pressed.connect(self.accept)
        hlayout.addWidget(ok_button)
        cancel_button = QtWidgets.QPushButton('Cancel')
        cancel_button.pressed.connect(self.reject)
        hlayout.addWidget(cancel_button)
        layout.addLayout(hlayout)

        self.setLayout(layout)

        self.frame_u8 = (np.clip(frame / frame.max() * 255, 0, 255)).astype(np.uint8)

        self.image_widget.set_numpy_array(self.frame_u8)
        self.frame = frame

    def local_max(self, x, y, radius):
        neigh = self.frame[y-radius:y+radius, x-radius:x+radius]
        dy, dx = np.unravel_index(np.argmax(neigh), neigh.shape)
        return x + dx - radius, y + dy - radius

    def adjust(self):
        p1 = self.image_widget.line.p1()
        p1x, p1y = self.local_max(int(p1.x()), int(p1.y()), 10)
        self.image_widget.line.setP1(QtCore.QPointF(p1x, p1y))
        p2 = self.image_widget.line.p2()
        p2x, p2y = self.local_max(int(p2.x()), int(p2.y()), 10)
        self.image_widget.line.setP2(QtCore.QPointF(p2x, p2y))
        self.image_widget.updateLine()

    def get_orientation(self):
        angle = -self.image_widget.line.angle()
        if angle < 180:
            angle += 360
        if angle > 180:
            angle -= 360
        return angle

    @staticmethod
    def run_calibration_dialog(frame, parent=None):
        dialog = DetectOrientationDialog(frame, parent)
        result = dialog.exec_()
        if result > 0:
            return dialog.get_orientation()
        else:
            return None


class OrientationImageView(widgets.ImageView):
    def __init__(self):
        super(OrientationImageView, self).__init__()
        pen = QtGui.QPen(QtCore.Qt.red, 1)
        self.line = QtCore.QLineF(0, 0, 0, 0)
        self.line_graphics_item = self.graphics_scene.addLine(self.line, pen)
        self.line_graphics_item.setZValue(100)

    def mousePressEvent(self, event):
        self.line.setP1(self.point_to_image_coordinates(event.pos()))
        self.line.setP2(self.point_to_image_coordinates(event.pos()))
        self.updateLine()

    def mouseMoveEvent(self, event):
        self.line.setP2(self.point_to_image_coordinates(event.pos()))
        self.updateLine()

    def updateLine(self):
        self.line_graphics_item.setLine(self.line)

    def point_to_image_coordinates(self, point):
        return self.pixmap_graphics_item.mapFromScene(self.mapToScene(point))


if __name__ == '__main__':
    import tifffile
    import os
    import sys
    in_file_path = os.path.join(os.path.dirname(__file__), 'test.tiff')
    frames = tifffile.imread(in_file_path)
    frame = frames[0]

    app = QtWidgets.QApplication(sys.argv)
    mainWin = QtWidgets.QMainWindow()
    mainWin.show()
    print(DetectOrientationDialog.run_calibration_dialog(frame, mainWin))

    sys.exit(app.exec_())

