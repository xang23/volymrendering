from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt


class TFCanvasWidget(QtWidgets.QWidget):
    def __init__(self, canvas, parent=None, label="Reset View"):
        super().__init__(parent)
        self.canvas = canvas
        self.reset_btn = QtWidgets.QPushButton(label)
        self.reset_btn.setToolTip("Reset this canvas view (hold Shift and click to reset all canvases)")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

        h = QtWidgets.QHBoxLayout()
        h.addStretch(1)
        h.addWidget(self.reset_btn)
        lay.addLayout(h)

        self.reset_btn.clicked.connect(self._on_reset_clicked)

    def _on_reset_clicked(self):
        mods = QtWidgets.QApplication.keyboardModifiers()
        shift_held = bool(mods & Qt.ShiftModifier)
        if shift_held:
            w = self.parent()
            while w is not None and not hasattr(w, 'reset_all_views'):
                w = w.parent()
            if w is not None and hasattr(w, 'reset_all_views'):
                w.reset_all_views()
                return
        try:
            self.canvas.reset_view()
        except Exception:
            try:
                self.canvas.draw()
            except Exception:
                pass