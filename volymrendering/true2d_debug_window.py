import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QSurfaceFormat

from widget_factory import WidgetFactory, WidgetType
from gl_true2d_test_widget import GLTrue2DTestWidget


def make_test_widgets():
    widgets = []

    # Diagonal-ish red Gaussian: tests real joint fx/fy lookup.
    widgets.append(
        WidgetFactory.create_widget(
            WidgetType.GAUSSIAN,
            center_intensity=128,
            center_gradient=128,
            intensity_std=22,
            gradient_std=22,
            opacity=1.0,
            color=(1.0, 0.1, 0.1),
            blend_mode="max",
        )
    )

    # Off-center green Gaussian: proves x/y are not collapsed into one 1D TF.
    widgets.append(
        WidgetFactory.create_widget(
            WidgetType.GAUSSIAN,
            center_intensity=70,
            center_gradient=190,
            intensity_std=18,
            gradient_std=18,
            opacity=0.9,
            color=(0.1, 1.0, 0.1),
            blend_mode="max",
        )
    )

    # Blue rectangle: makes a hard 2D region.
    widgets.append(
        WidgetFactory.create_widget(
            WidgetType.RECTANGULAR,
            center_intensity=190,
            center_gradient=70,
            intensity_width=45,
            gradient_height=45,
            falloff=4.0,
            opacity=0.8,
            color=(0.1, 0.2, 1.0),
            blend_mode="max",
        )
    )

    return widgets


class TestWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TRUE 2D TF OpenGL Debug Test")
        self.resize(1000, 800)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        instructions = QtWidgets.QLabel(
            "<b>True 2D Transfer Function Debug Test</b><br>"
            "Keys: 0=True 2D TF, 1=Feature X, 2=Feature Y, "
            "3=Joint feature color, R=Force red, +/-=Sampling"
        )
        layout.addWidget(instructions)

        self.gl_widget = GLTrue2DTestWidget(make_test_widgets())
        layout.addWidget(self.gl_widget, 1)

        self.setCentralWidget(central)


if __name__ == "__main__":
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QtWidgets.QApplication(sys.argv)

    win = TestWindow()
    win.show()

    sys.exit(app.exec_())