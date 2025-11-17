# main_app.py
# Entry point - create the UI and wire modules together

import sys
import os
import vtk
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from dataset_loader import load_volume
from transfer_function_plot import TransferFunctionPlot, TransferFunction2D, TFCanvasWidget
from tf_manager import TransferFunctionManager
from volume_renderer import VolumeRenderer

from vtk.util import numpy_support

import numpy as np

class VolumeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Renderer with TF Editor (modular)")
        self.resize(1100, 800)

        # Central widget layout
        frame = QtWidgets.QFrame()
        vlay = QtWidgets.QVBoxLayout(frame)

        # VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(frame)
        vlay.addWidget(self.vtkWidget)

        # toolbar: log checkbox, toggle, buttons
        toolbar_h = QtWidgets.QHBoxLayout()
        self.log_checkbox = QtWidgets.QCheckBox("Log Histogram")
        toolbar_h.addWidget(self.log_checkbox)
        toolbar_h.addStretch(1)

        self.btn_new_tf = QtWidgets.QPushButton("New TF (start fresh)")
        self.btn_new_tf.setToolTip("Create a new TF for the currently loaded dataset")
        toolbar_h.addWidget(self.btn_new_tf)

        vlay.addLayout(toolbar_h)

        # TF canvases - 1D and 2D stacked + toggle button
        self.tf_stack = QtWidgets.QStackedWidget()
        vlay.addWidget(self.tf_stack)

        self.view_toggle = QtWidgets.QPushButton("Switch to 2D TF")
        self.view_toggle.setCheckable(True)
        vlay.addWidget(self.view_toggle)

        # TF manager & controls
        tf_ctrl_h = QtWidgets.QHBoxLayout()
        self.tf_manager = TransferFunctionManager()
        self.tf_selector = QtWidgets.QComboBox()
        tf_ctrl_h.addWidget(self.tf_selector)
        self.save_tf_btn = QtWidgets.QPushButton("Save TF")
        tf_ctrl_h.addWidget(self.save_tf_btn)
        vlay.addLayout(tf_ctrl_h)

        # assemble main widget
        frame.setLayout(vlay)
        self.setCentralWidget(frame)

        # renderer + volume renderer object
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.vol_renderer = VolumeRenderer()
        self.renderer.AddVolume(self.vol_renderer.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCamera()
        self.interactor.Initialize()

        # default empty data arrays (will be replaced by loader)
        empty = np.zeros((1,), dtype=np.float32)
        self.normalized_scalars = empty
        self.gradient_normalized = empty
        self.intensity_range = (0.0, 1.0)
        self.gradient_range = (0.0, 1.0)

        # create TF canvases (they will read hist_data from self.normalized_scalars below)
        self.tf1d = TransferFunctionPlot(self.on_tf_changed, self.normalized_scalars, self.log_checkbox)
        self.tf1d_widget = TFCanvasWidget(self.tf1d, parent=self, label="Reset 1D View")
        self.tf2d = TransferFunction2D(np.zeros((256,256)), self.intensity_range, self.gradient_range, self.log_checkbox)
        self.tf2d_widget = TFCanvasWidget(self.tf2d, parent=self, label="Reset 2D View")

        self.tf_stack.addWidget(self.tf1d_widget)
        self.tf_stack.addWidget(self.tf2d_widget)

        # connect toggles and buttons
        self.view_toggle.toggled.connect(self.on_toggle_view)
        self.log_checkbox.stateChanged.connect(self.on_toggle_log)
        self.save_tf_btn.clicked.connect(self.do_save_tf)
        self.tf_selector.currentIndexChanged.connect(self.on_load_selected_tf)
        self.btn_new_tf.clicked.connect(self.new_tf_for_current_dataset)

        # menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QtWidgets.QAction("Open Dataset", self)
        open_action.triggered.connect(self.open_dataset)
        file_menu.addAction(open_action)
        reset_action = QtWidgets.QAction("Reset TF", self)
        reset_action.triggered.connect(self.reset_tf_to_default)
        file_menu.addAction(reset_action)

        # populate TF selector from disk
        self.tf_manager.load_from_disk()
        self._refresh_tf_selector()

        # call update to ensure UI shows default TF
        self.update_canvases_from_tf(self.tf_manager.get_default_tf())

    # UI callbacks ---------------------------------------------------
    def on_toggle_view(self, checked):
        idx = 1 if checked else 0
        self.tf_stack.setCurrentIndex(idx)
        self.view_toggle.setText("Switch to 1D TF" if checked else "Switch to 2D TF")

    def on_toggle_log(self, state):
        try:
            self.tf1d._draw_plot()
        except Exception:
            pass
        try:
            self.tf2d._on_log_toggled(state)
        except Exception:
            pass

    def open_dataset(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Dataset", "", 
            "Volume Files (*.vti *.mhd *.vol *.raw *.nii *.nii.gz);;All Files (*)")
        if not fname:
            return
        try:
            image_data, reader = load_volume(fname)  # returns (vtkImageData, maybe-reader)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Failed", f"Failed to load {fname}:\n{e}")
            return

        # give the volume renderer the input (it will call SetInputConnection or SetInputData)
        self.vol_renderer.set_input(image_data=image_data, reader=reader)

        # update histogram inputs for canvases
        scalars = image_data.GetPointData().GetScalars()
        arr = numpy_support.vtk_to_numpy(scalars).astype(np.float32)
        raw_int_min, raw_int_max = arr.min(), arr.max()
        self.intensity_range = (raw_int_min, raw_int_max)
        if raw_int_max - raw_int_min == 0:
            self.normalized_scalars = np.zeros_like(arr)
        else:
            self.normalized_scalars = 255.0 * (arr - raw_int_min) / (raw_int_max - raw_int_min)

        # gradient
        try:
            grad_filter = vtk.vtkImageGradientMagnitude()
            if reader is not None:
                grad_filter.SetInputConnection(reader.GetOutputPort())
            else:
                grad_filter.SetInputData(image_data)
            grad_filter.Update()
            g = numpy_support.vtk_to_numpy(grad_filter.GetOutput().GetPointData().GetScalars()).astype(np.float32)
            raw_gmin, raw_gmax = g.min(), g.max()
            self.gradient_range = (raw_gmin, raw_gmax)
            if raw_gmax - raw_gmin == 0:
                self.gradient_normalized = np.zeros_like(g)
            else:
                self.gradient_normalized = 255.0 * (g - raw_gmin) / (raw_gmax - raw_gmin)
        except Exception:
            self.gradient_normalized = np.zeros_like(self.normalized_scalars)

        # update canvases with new hist data and 2D histogram
        try:
            self.tf1d.hist_data = self.normalized_scalars
            self.tf1d._draw_plot()
        except Exception:
            pass
        try:
            h2, _, _ = np.histogram2d(self.normalized_scalars, self.gradient_normalized, bins=(256,256), range=((0,255),(0,255)))
            self.tf2d.raw = h2
            if self.log_checkbox.isChecked():
                self.tf2d._on_log_toggled(True)
            else:
                self.tf2d._draw()
        except Exception:
            pass

        # update renderer
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def new_tf_for_current_dataset(self):
        """Clear the current TF GUI state (start fresh)"""
        self.tf1d.points_x = [0.0, 255.0]
        self.tf1d.points_y = [0.0, 1.0]
        self.tf1d.colors = [(1.0,1.0,1.0), (1.0,1.0,1.0)]
        self.tf1d._draw_plot()
        self.on_tf_changed(self.tf1d.points_x, self.tf1d.points_y, self.tf1d.colors)

    def reset_tf_to_default(self):
        default = self.tf_manager.get_default_tf()
        self.update_canvases_from_tf(default)
        self.on_tf_changed(*default)

    def _refresh_tf_selector(self):
        self.tf_selector.blockSignals(True)
        self.tf_selector.clear()
        for name in self.tf_manager.list_names():
            self.tf_selector.addItem(name)
        self.tf_selector.blockSignals(False)

    # TF save/load callbacks ----------------------------------------
    def do_save_tf(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save TF", "Name:")
        if not ok or not name:
            return
        xs = list(self.tf1d.points_x)
        ys = list(self.tf1d.points_y)
        colors = [tuple(c) for c in self.tf1d.colors]
        self.tf_manager.save(name, xs, ys, colors)
        self.tf_manager.save_to_disk()
        self._refresh_tf_selector()
        self.tf_selector.setCurrentText(name)

    def on_load_selected_tf(self, idx):
        # triggered by combobox change
        if idx < 0:
            return
        name = self.tf_selector.itemText(idx)
        if name not in self.tf_manager.saved_tfs:
            return
        xs, ys, colors = self.tf_manager.saved_tfs[name]
        self.update_canvases_from_tf((xs, ys, colors))
        self.on_tf_changed(xs, ys, colors)

    def update_canvases_from_tf(self, tftuple):
        xs, ys, colors = tftuple
        self.tf1d.points_x = list(xs)
        self.tf1d.points_y = list(ys)
        self.tf1d.colors = list(colors)
        self.tf1d._draw_plot()
        try:
            self.tf2d.set_tf_state(xs, ys, colors)
        except Exception:
            pass

    # called from TF canvas when user edits TF
    def on_tf_changed(self, xs, ys, colors):
        # update VTK transfer functions on renderer
        self.vol_renderer.update_transfer_function(xs, ys, colors, intensity_range=self.intensity_range)
        self.vtkWidget.GetRenderWindow().Render()
        # keep 2D overlay consistent
        try:
            self.tf2d.set_tf_state(xs, ys, colors)
        except Exception:
            pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = VolumeApp()
    win.show()
    sys.exit(app.exec_())
