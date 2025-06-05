# -*- coding: utf-8 -*-
import sys
import numpy as np
from vtk.util import numpy_support
import vtk
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm


class TransferFunctionPlot(FigureCanvas):
    """
    1D transfer‐function editor with a histogram of intensities (0..255).
    Exactly as before: histogram over [original_min..original_max], 100 bins,
    then remapped to [0..255] on the x‐axis.  Supports linear vs log toggle.
    """
    def __init__(self, update_callback, scalar_data, log_toggle_checkbox=None):
        self.fig = Figure(figsize=(5, 2))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

        # Initial TF control points in (x=0..255, y=0..1)
        self.points_x = [0, 64, 192, 255]
        self.points_y = [0.0, 0.3, 0.6, 1.0]
        self.colors = [(1.0, 1.0, 1.0)] * len(self.points_x)

        self.selected_index = None
        self.update_callback = update_callback
        self.dragging = False

        # Now we assign the RAW scalar array (not normalized!)
        self.hist_data = scalar_data
        self.log_toggle_checkbox = log_toggle_checkbox

        # Draw initial 1D histogram + TF curve
        self._draw_plot()

        # Mouse events for editing TF control points
        self.mpl_connect("button_press_event", self.on_press)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("button_release_event", self.on_release)

    def _draw_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Intensity")

        # ─── Exactly the “old” histogram code, over raw data range ───────────────
        original_min = float(self.hist_data.min())
        original_max = float(self.hist_data.max())
        hist, bin_edges = np.histogram(
            self.hist_data,
            bins=100,
            range=(original_min, original_max)
        )

        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            hist = np.log1p(hist)
            self.ax.set_ylabel("log(1 + count)")
        else:
            self.ax.set_ylabel("Normalized Count")

        if hist.max() > 0:
            hist = hist.astype(np.float64) / hist.max()

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_centers = 255 * (bin_centers - original_min) / (original_max - original_min)

        self.ax.plot(bin_centers, hist, color='gray', linewidth=1, alpha=0.4)
        self.ax.fill_between(bin_centers, hist, color='lightgray', alpha=0.5)
        # ─────────────────────────────────────────────────────────────────────────

        # Draw the transfer‐function curve on top
        self.ax.set_title("1D Transfer Function with Histogram")
        self.ax.grid(True)
        for i in range(len(self.points_x) - 1):
            self.ax.plot(
                self.points_x[i : i + 2],
                self.points_y[i : i + 2],
                color='orange',
                linewidth=2
            )
        for x, y, color in zip(self.points_x, self.points_y, self.colors):
            self.ax.plot(x, y, 'o', color=color, markersize=8)

        self.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.figure.set_size_inches(
            self.width() / self.figure.dpi,
            self.height() / self.figure.dpi
        )
        self.draw()

    def _closest_point(self, event):
        if event.inaxes != self.ax:
            return None
        distances = [
            np.hypot(event.xdata - x, event.ydata - y)
            for x, y in zip(self.points_x, self.points_y)
        ]
        if not distances:
            return None
        idx = int(np.argmin(distances))
        if distances[idx] < 10:
            return idx
        return None

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        idx = self._closest_point(event)

        # Double‐click: add new control point
        if event.dblclick:
            new_x = max(0, min(255, event.xdata))
            new_y = max(0.0, min(1.0, event.ydata))
            color = (1.0, 1.0, 1.0)
            if event.guiEvent.modifiers() & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
            self.points_x.append(new_x)
            self.points_y.append(new_y)
            self.colors.append(color)
            zipped = sorted(zip(self.points_x, self.points_y, self.colors), key=lambda p: p[0])
            self.points_x, self.points_y, self.colors = map(list, zip(*zipped))
            self._draw_plot()
            self.update_callback(self.points_x, self.points_y, self.colors)
            return

        # Right‐click: delete a point (if not an endpoint)
        if idx is not None and event.button == 3:
            if idx not in (0, len(self.points_x) - 1):
                self.points_x.pop(idx)
                self.points_y.pop(idx)
                self.colors.pop(idx)
                self._draw_plot()
                self.update_callback(self.points_x, self.points_y, self.colors)
            return

        # Left‐click: start drag or color‐change if Shift
        if idx is not None and event.button == 1:
            if event.guiEvent.modifiers() & Qt.ShiftModifier:
                # Shift+click changes color
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    self.colors[idx] = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                    self._draw_plot()
                    self.update_callback(self.points_x, self.points_y, self.colors)
            else:
                self.selected_index = idx
                self.dragging = True

    def on_motion(self, event):
        if not self.dragging or self.selected_index is None or event.inaxes != self.ax:
            return

        new_x = max(0, min(255, event.xdata))
        new_y = max(0.0, min(1.0, event.ydata))

        if self.selected_index == 0:
            new_x = 0
        elif self.selected_index == len(self.points_x) - 1:
            new_x = 255

        self.points_x[self.selected_index] = new_x
        self.points_y[self.selected_index] = new_y

        zipped = sorted(zip(self.points_x, self.points_y, self.colors), key=lambda p: p[0])
        self.points_x, self.points_y, self.colors = map(list, zip(*zipped))

        try:
            self.selected_index = self.points_x.index(new_x)
        except ValueError:
            self.selected_index = None

        self._draw_plot()
        self.update_callback(self.points_x, self.points_y, self.colors)

    def on_release(self, event):
        self.selected_index = None
        self.dragging = False



class TransferFunction2D(FigureCanvas):
    """
    2D histogram of (intensity vs gradient magnitude), using a 256×256 bin grid
    over [0..255]×[0..255].  “Log Histogram” toggles linear vs log color scaling.
    """
    def __init__(self, raw_hist2d, log_toggle_checkbox=None):
        fig = Figure(figsize=(5, 5), dpi=100)
        super().__init__(fig)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry()

        self.ax = fig.add_subplot(111)
        self.figure.tight_layout()

        # Store the raw 256×256 counts (over full [0..255]×[0..255])
        self.raw = raw_hist2d
        self.log_checkbox = log_toggle_checkbox

        # Compute initial display (linear or log) and show it
        disp = self._get_display_data()
        norm = LogNorm() if (self.log_checkbox and self.log_checkbox.isChecked()) else None

        self.im = self.ax.imshow(
            disp.T,
            origin='lower',
            cmap='hot',
            norm=norm,
            interpolation='nearest',
            extent=(0, 255, 0, 255),
            aspect='auto'
        )
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.set_title("2D Transfer Function (Intensity vs Gradient)")
        self.ax.set_xlabel("Intensity (0..255)")
        self.ax.set_ylabel("Gradient Magnitude (0..255)")

        # When “Log Histogram” toggles, redraw
        if self.log_checkbox is not None:
            self.log_checkbox.stateChanged.connect(self._on_log_toggled)

    def _get_display_data(self):
        arr = self.raw.astype(np.float64)
        if self.log_checkbox is not None and self.log_checkbox.isChecked():
            arr = np.log1p(arr)
        maxval = arr.max()
        if maxval > 0:
            arr /= maxval
        return arr

    def _on_log_toggled(self, state):
        disp = self._get_display_data()
        if self.log_checkbox and self.log_checkbox.isChecked():
            self.im.set_norm(LogNorm())
        else:
            self.im.set_norm(None)
        self.im.set_data(disp.T)
        self.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.figure.set_size_inches(
            self.width() / self.figure.dpi,
            self.height() / self.figure.dpi
        )
        self.draw()



class VolumeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK Volume + Interactive 1D/2D Transfer Function")

        self.frame = QtWidgets.QFrame()
        self.layout = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtkWidget)

        # ─── “Log Histogram” Checkbox (affects both 1D & 2D) ───────────────
        self.log_checkbox = QtWidgets.QCheckBox("Log Histogram")
        self.log_checkbox.setChecked(False)
        self.log_checkbox.stateChanged.connect(self.toggle_log_histogram)
        self.layout.addWidget(self.log_checkbox)

        # ─── StackedWidget: index 0 = 1D, index 1 = 2D ──────────────────────
        self.canvas_container = QtWidgets.QStackedWidget()
        self.layout.addWidget(self.canvas_container)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

        # ─── VTK Setup ───────────────────────────────────────────────────────
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Load the .vti file
        path = r"C:\Users\josde002\source\repos\volymrendering\data\head-binary-zlib.vti"
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(path)
        reader.Update()

        image_data = reader.GetOutput()
        scalars = image_data.GetPointData().GetScalars()
        np_scalars = numpy_support.vtk_to_numpy(scalars)

        # Compute gradient magnitude
        gradient_filter = vtk.vtkImageGradientMagnitude()
        gradient_filter.SetInputConnection(reader.GetOutputPort())
        gradient_filter.Update()

        gradient_data = gradient_filter.GetOutput().GetPointData().GetScalars()
        np_gradient = numpy_support.vtk_to_numpy(gradient_data)

        # ─── STEP 1: store the RAW scalar array for the 1D histogram ─────────
        self.raw_scalars = np_scalars.astype(np.float32)

        # ─── STEP 2: normalize for 2D (0..255)
        normalized_scalars = 255 * (np_scalars - np_scalars.min()) / (np_scalars.max() - np_scalars.min())
        self.normalized_scalars = normalized_scalars.astype(np.float32)

        gradient_normalized = 255 * (np_gradient - np_gradient.min()) / (np_gradient.max() - np_gradient.min())
        self.gradient_normalized = gradient_normalized.astype(np.float32)

        # ─── 1D TransferFunctionPlot ─────────────────────────────────────────
        self.plot_canvas = TransferFunctionPlot(
            self.update_opacity_function,
            self.raw_scalars,      # <-- pass raw data here
            self.log_checkbox
        )

        # ─── Build initial full‐range 2D histogram (256×256) ────────────────
        hist2d_full, xedges, yedges = np.histogram2d(
            self.normalized_scalars,
            self.gradient_normalized,
            bins=(256, 256),
            range=((0, 255), (0, 255))
        )
        self.raw_hist2d = hist2d_full

        # ─── 2D TransferFunction Canvas ─────────────────────────────────────
        self.tf2d_canvas = TransferFunction2D(self.raw_hist2d, self.log_checkbox)

        # Put the 1D and 2D canvases into the stacked widget
        self.canvas_container.addWidget(self.plot_canvas)  # index 0 = 1D
        self.canvas_container.addWidget(self.tf2d_canvas)  # index 1 = 2D
        self.canvas_container.setCurrentIndex(0)           # start in 1D

        # ─── VTK Volume‐Rendering Pipeline ─────────────────────────────────
        self.reader = reader
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.mapper.SetInputConnection(reader.GetOutputPort())

        raw_int_min = float(np_scalars.min())
        raw_int_max = float(np_scalars.max())

        self.color_function = vtk.vtkColorTransferFunction()
        self.color_function.AddRGBPoint(raw_int_min,                                  0.0, 0.0, 0.0)
        self.color_function.AddRGBPoint(raw_int_min + 0.25 * (raw_int_max - raw_int_min),  1.0, 0.5, 0.3)
        self.color_function.AddRGBPoint(raw_int_min + 0.50 * (raw_int_max - raw_int_min),  1.0, 1.0, 0.9)
        self.color_function.AddRGBPoint(raw_int_min + 0.75 * (raw_int_max - raw_int_min),  0.5, 1.0, 0.5)
        self.color_function.AddRGBPoint(raw_int_max,                                  1.0, 1.0, 1.0)

        self.opacity_function = vtk.vtkPiecewiseFunction()
        # Initialize opacity from the 1D TF’s default control points
        for x, y in zip(self.plot_canvas.points_x, self.plot_canvas.points_y):
            abs_val = raw_int_min + (x / 255.0) * (raw_int_max - raw_int_min)
            self.opacity_function.AddPoint(abs_val, y)

        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()

        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.volume_property)

        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        self.renderer.ResetCamera()

        self.interactor.Initialize()
        self.interactor.Start()

        # ─── Button to switch between 1D ↔ 2D ───────────────────────────────────
        self.view_toggle = QtWidgets.QPushButton("Switch to 2D TF")
        self.view_toggle.setCheckable(True)
        self.view_toggle.setChecked(False)
        self.view_toggle.toggled.connect(self.toggle_tf_view)
        self.layout.addWidget(self.view_toggle)

    def toggle_tf_view(self, checked):
        """
        Swap between the 1D (index 0) and 2D (index 1) canvases.
        """
        if checked:
            self.canvas_container.setCurrentIndex(1)
            self.view_toggle.setText("Switch to 1D TF")
        else:
            self.canvas_container.setCurrentIndex(0)
            self.view_toggle.setText("Switch to 2D TF")

    def update_opacity_function(self, xs, ys, colors):
        """
        Called whenever the 1D TF curve changes.  Rebuild and re-render the VTK volume’s
        opacity + color functions (in the raw scalar range).
        """
        self.color_function.RemoveAllPoints()
        self.opacity_function.RemoveAllPoints()

        raw_int_min = float(self.raw_scalars.min())
        raw_int_max = float(self.raw_scalars.max())

        for x, y, c in zip(xs, ys, colors):
            abs_val = raw_int_min + (x / 255.0) * (raw_int_max - raw_int_min)
            self.opacity_function.AddPoint(abs_val, y)
            self.color_function.AddRGBPoint(abs_val, *c)

        self.vtkWidget.GetRenderWindow().Render()

    def toggle_log_histogram(self):
        """
        Redraw the 1D histogram.  The 2D canvas listens to the same checkbox
        and will redraw itself via its own _on_log_toggled().
        """
        self.plot_canvas._draw_plot()
        self.tf2d_canvas._on_log_toggled(None)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolumeApp()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec_())
