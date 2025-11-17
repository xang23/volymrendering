# -*- coding: utf-8 -*-
"""
Volume App - Patched v3 (Load Dataset button + .vol support + remember last folder)
Interactive 1D and 2D transfer functions with consistent point sorting and TF save/load.
Minimal changes from your v3 version: added Load Dataset button, load_volume_dialog, load_volume,
and small .vol/raw loader dialog. Remembers last folder in a small settings file.
"""

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
import json
import os


# --------------------------- Helper: Canvas container with reset button ---------------------------
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


# --------------------------- TransferFunctionPlot (1D) ---------------------------
class TransferFunctionPlot(FigureCanvas):
    def __init__(self, update_callback, scalar_data, log_toggle_checkbox=None):
        fig = Figure(figsize=(5, 2))
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

        # TF control state
        hist, bins = np.histogram(scalar_data, bins=256, range=(0, 255))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # find peaks or thresholds
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        peaks = np.where(hist > hist.max() * 0.05)[0]  # top 5% of voxel counts
        points_x = bin_centers[peaks]
        points_y = np.clip(hist[peaks] / hist.max(), 0.0, 1.0) # normalize for opacity
        colors = [(1.0, 1.0, 1.0) for _ in points_x]# start with white, user can edit
        self.points_x = list(points_x)
        self.points_y = list(points_y)
        self.colors = list(colors)

        self.update_callback = update_callback
        self.log_toggle_checkbox = log_toggle_checkbox

        # interaction state
        self.selected_index = None
        self.dragging = False
        self.histogram_scaling = False
        self._hist_scale_start_y = None

        # keep last limits
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 1.0)
        self._reset_view_requested = False

        # histogram data
        self.hist_data = scalar_data

        # connect events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

        self._draw_plot()

    def _data_to_display(self, x):
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            return 255.0 * (np.log1p(x) / np.log1p(255.0))
        return float(x)

    def _display_to_data(self, x_disp):
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            frac = float(x_disp) / 255.0
            val = np.expm1(frac * np.log1p(255.0))
            return float(np.clip(val, 0.0, 255.0))
        return float(np.clip(x_disp, 0.0, 255.0))

    def reset_view(self):
        self._reset_view_requested = True
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 1.0)
        self._draw_plot()

    def _draw_plot(self):
        curr_xlim = self._cached_xlim
        curr_ylim = self._cached_ylim
        self.ax.clear()

        # histogram
        hist, bin_edges = np.histogram(self.hist_data, bins=150, range=(0.0, 255.0))
        hist = hist.astype(np.float64)
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            hist = np.log1p(hist)
            self.ax.set_ylabel('log(1 + count)')
        else:
            self.ax.set_ylabel('Normalized Count')
        if hist.max() > 0:
            hist /= hist.max()
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.ax.plot(bin_centers, hist, color='gray', linewidth=1, alpha=0.4)
        self.ax.fill_between(bin_centers, hist, color='lightgray', alpha=0.5)

        # TF curve
        display_xs = [self._data_to_display(x) for x in self.points_x]
        for i in range(len(display_xs) - 1):
            self.ax.plot(display_xs[i:i+2], self.points_y[i:i+2], color='orange', linewidth=2)
        for xd, y, c in zip(display_xs, self.points_y, self.colors):
            self.ax.plot(xd, y, 'o', color=c, markersize=8, mec='k')

        self.ax.set_title('1D Transfer Function with Histogram')
        self.ax.set_xlabel('Intensity (display)')
        self.ax.grid(True)

        if self._reset_view_requested:
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 1)
            self._reset_view_requested = False
        else:
            self.ax.set_xlim(*curr_xlim)
            self.ax.set_ylim(*curr_ylim)

        self._cached_xlim = self.ax.get_xlim()
        self._cached_ylim = self.ax.get_ylim()

        self.draw()

    def _closest_point(self, event, pixel_tol=10):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        display_xs = [self._data_to_display(x) for x in self.points_x]
        pts = np.array([[x, y] for x, y in zip(display_xs, self.points_y)])
        pix = self.ax.transData.transform(pts)
        ev = np.array([event.x, event.y])
        dists = np.hypot(pix[:,0]-ev[0], pix[:,1]-ev[1])
        if len(dists) == 0:
            return None
        idx = int(np.argmin(dists))
        if dists[idx] <= pixel_tol:
            return idx
        return None

    def _sort_points_with_colors(self):
        zipped = sorted(zip(self.points_x, self.points_y, self.colors), key=lambda p: p[0])
        self.points_x, self.points_y, self.colors = map(list, zip(*zipped))

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.x is not None and event.x <= 40:
            self.histogram_scaling = True
            self._hist_scale_start_y = event.y
            return

        idx = self._closest_point(event)

        # double click -> add
        if getattr(event, 'dblclick', False):
            if event.xdata is None:
                return
            new_x = self._display_to_data(event.xdata)
            new_y = float(np.clip(event.ydata if event.ydata is not None else 0.0, 0.0, 1.0))
            color = (1.0, 1.0, 1.0)
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
            if mods & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
            self.points_x.append(new_x)
            self.points_y.append(new_y)
            self.colors.append(color)
            self._sort_points_with_colors()
            self._draw_plot()
            self.update_callback(self.points_x, self.points_y, self.colors)
            return

        # right click -> delete
        if idx is not None and getattr(event, 'button', None) == 3:
            if idx not in (0, len(self.points_x)-1):
                self.points_x.pop(idx)
                self.points_y.pop(idx)
                self.colors.pop(idx)
                self._sort_points_with_colors()
                self._draw_plot()
                self.update_callback(self.points_x, self.points_y, self.colors)
            return

        # left click -> drag
        if idx is not None and getattr(event, 'button', None) == 1:
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
            if mods & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    self.colors[idx] = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                    self._draw_plot()
                    self.update_callback(self.points_x, self.points_y, self.colors)
            else:
                self.selected_index = idx
                self.dragging = True

    def on_motion(self, event):
        if self.histogram_scaling:
            if event.y is None or self._hist_scale_start_y is None:
                return
            dy = event.y - self._hist_scale_start_y
            factor = np.exp(dy / 200.0)
            self._hist_scale_start_y = event.y
            self._draw_plot()
            return

        if not self.dragging or self.selected_index is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        new_x = self._display_to_data(event.xdata)
        new_y = float(np.clip(event.ydata, 0.0, 1.0))
        if self.selected_index == 0:
            new_x = 0.0
        elif self.selected_index == len(self.points_x)-1:
            new_x = 255.0
        self.points_x[self.selected_index] = new_x
        self.points_y[self.selected_index] = new_y
        self._sort_points_with_colors()
        try:
            self.selected_index = min(range(len(self.points_x)), key=lambda i: abs(self.points_x[i] - new_x))
        except Exception:
            self.selected_index = None
        self._cached_xlim = self.ax.get_xlim()
        self._cached_ylim = self.ax.get_ylim()
        self._draw_plot()
        self.update_callback(self.points_x, self.points_y, self.colors)

    def on_release(self, event):
        self.selected_index = None
        self.dragging = False
        self.histogram_scaling = False
        self._hist_scale_start_y = None

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        step = getattr(event, 'step', None)
        if step is None:
            step = 1 if getattr(event, 'button', None) == 'up' else -1
        base = 0.9
        scale = base ** step
        try:
            shift = bool(event.guiEvent.modifiers() & Qt.ShiftModifier)
        except Exception:
            shift = False
        if shift:
            ydata = event.ydata if event.ydata is not None else 0.5 * sum(self.ax.get_ylim())
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(ydata + (ymin - ydata) * scale, ydata + (ymax - ydata) * scale)
        else:
            xdata_disp = event.xdata if event.xdata is not None else 0.5 * sum(self.ax.get_xlim())
            xmin, xmax = self.ax.get_xlim()
            new_xmin = max(0.0, xdata_disp + (xmin - xdata_disp) * scale)
            new_xmax = min(255.0, xdata_disp + (xmax - xdata_disp) * scale)
            if new_xmax - new_xmin > 1e-6:
                self.ax.set_xlim(new_xmin, new_xmax)
        self._cached_xlim = self.ax.get_xlim()
        self._cached_ylim = self.ax.get_ylim()
        self.draw()


# --------------------------- TransferFunction2D (2D) ---------------------------
class TransferFunction2D(FigureCanvas):
    """ Interactive 2D histogram (intensity vs gradient) with editable TF overlay.
    Dragging points updates stored TF and VTK immediately.
    """
    def __init__(self, raw_hist2d, intensity_range, gradient_range, log_toggle_checkbox=None):
        fig = Figure(figsize=(5, 5), dpi=100)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

        self.raw = raw_hist2d
        self.int_range = intensity_range
        self.grad_range = gradient_range
        self.log_checkbox = log_toggle_checkbox

        # TF state
        self.points_x = []
        self.points_y = []
        self.colors = []

        # interaction
        self.selected_index = None
        self.dragging = False
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 255.0)
        self._reset_view_requested = False

        disp = self._get_display_data()
        norm = LogNorm() if (self.log_checkbox and self.log_checkbox.isChecked()) else None
        self.im = self.ax.imshow(
            disp.T, origin='lower', cmap='hot', norm=norm,
            interpolation='nearest', extent=(0,255,0,255), aspect='auto'
        )

        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # overlay
        self.tf_line, = self.ax.plot([], [], color='orange', linewidth=2)
        self.tf_scatter = self.ax.scatter([], [], s=40, edgecolor='k', zorder=10)

        # events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)

        if self.log_checkbox is not None:
            self.log_checkbox.stateChanged.connect(self._on_log_toggled)

        self._draw()

    def set_tf_state(self, xs, ys, colors):
        """Set TF state from 1D canvas or saved TF."""
        self.points_x = list(xs)
        self.points_y = list(ys)
        self.colors = list(colors)
        self._draw()

    def _get_display_data(self):
        arr = self.raw.astype(np.float64)
        if self.log_checkbox and self.log_checkbox.isChecked():
            arr = np.log1p(arr)
        m = arr.max()
        if m > 0:
            arr /= m
        return arr

    def _on_log_toggled(self, state):
        disp = self._get_display_data()
        if self.log_checkbox and self.log_checkbox.isChecked():
            self.im.set_norm(LogNorm())
        else:
            self.im.set_norm(None)
        self.im.set_data(disp.T)
        self._draw()

    def reset_view(self):
        self._reset_view_requested = True
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 255.0)
        self._draw()

    def _draw(self):
        curr_xlim = self._cached_xlim
        curr_ylim = self._cached_ylim

        # update image
        self.im.set_data(self._get_display_data().T)

        # update TF overlay
        if len(self.points_x) == 0:
            self.tf_line.set_data([], [])
            self.tf_scatter.set_offsets(np.empty((0, 2)))
        else:
            x = np.array(self.points_x)
            y = 255.0 * np.array(self.points_y)
            self.tf_line.set_data(x, y)
            offsets = np.column_stack([x, y])
            self.tf_scatter.set_offsets(offsets)
            if self.colors and len(self.colors) == len(x):
                self.tf_scatter.set_facecolor(self.colors)
            else:
                self.tf_scatter.set_facecolor([(1.0,1.0,1.0)] * len(x))

        # restore or cache limits
        if self._reset_view_requested:
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 255)
            self._reset_view_requested = False
        else:
            self.ax.set_xlim(*curr_xlim)
            self.ax.set_ylim(*curr_ylim)

        self._cached_xlim = self.ax.get_xlim()
        self._cached_ylim = self.ax.get_ylim()
        self.draw()

    def _closest_point(self, event, pixel_tol=10):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        pts = np.array([[x, 255.0*y] for x,y in zip(self.points_x, self.points_y)])
        pix = self.ax.transData.transform(pts)
        ev = np.array([event.x, event.y])
        dists = np.hypot(pix[:,0]-ev[0], pix[:,1]-ev[1])
        if len(dists) == 0:
            return None
        idx = int(np.argmin(dists))
        if dists[idx] <= pixel_tol:
            return idx
        return None

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        idx = self._closest_point(event)

        # double-click: add point
        if getattr(event, 'dblclick', False):
            if event.xdata is None or event.ydata is None:
                return
            new_x = float(np.clip(event.xdata, 0.0, 255.0))
            new_y = float(np.clip(event.ydata / 255.0, 0.0, 1.0))
            color = (1.0, 1.0, 1.0)
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
            if mods & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
            self.points_x.append(new_x)
            self.points_y.append(new_y)
            self.colors.append(color)
            zipped = sorted(zip(self.points_x, self.points_y, self.colors), key=lambda p:p[0])
            self.points_x, self.points_y, self.colors = map(list, zip(*zipped))
            self._draw()
            self._notify_app()
            return

        # right-click: delete
        if idx is not None and getattr(event, 'button', None) == 3:
            if idx not in (0, len(self.points_x)-1):
                self.points_x.pop(idx)
                self.points_y.pop(idx)
                self.colors.pop(idx)
                self._draw()
                self._notify_app()
            return

        # left-click: drag
        if idx is not None and getattr(event, 'button', None) == 1:
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
            if mods & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    self.colors[idx] = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                    self._draw()
                    self._notify_app()
            else:
                self.selected_index = idx
                self.dragging = True

    def on_motion(self, event):
        if not self.dragging or self.selected_index is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        new_x = float(np.clip(event.xdata, 0.0, 255.0))
        new_y = float(np.clip(event.ydata / 255.0, 0.0, 1.0))

        # lock endpoints
        if self.selected_index == 0:
            new_x = 0.0
        elif self.selected_index == len(self.points_x)-1:
            new_x = 255.0

        self.points_x[self.selected_index] = new_x
        self.points_y[self.selected_index] = new_y
        zipped = sorted(zip(self.points_x, self.points_y, self.colors), key=lambda p:p[0])
        self.points_x, self.points_y, self.colors = map(list, zip(*zipped))
        self._cached_xlim = self.ax.get_xlim()
        self._cached_ylim = self.ax.get_ylim()
        self._draw()
        self._notify_app()

    def on_release(self, event):
        self.selected_index = None
        self.dragging = False

    def _notify_app(self):
        """Bubble up to VolumeApp to update VTK TF."""
        w = self.parent()
        while w is not None and not hasattr(w, 'update_opacity_function'):
            w = w.parent()
        if w is not None and hasattr(w, 'update_opacity_function'):
            w.update_opacity_function(self.points_x, self.points_y, self.colors)


# --------------------------- Main Application ---------------------------
class VolumeApp(QtWidgets.QMainWindow):
    TF_SAVE_FILE = "saved_tfs.json"  # default TF storage
    LAST_DIR_FILE = ".last_open_dir"  # small helper to remember last folder

    def __init__(self):
        super().__init__()
        self.setWindowTitle('VTK Volume + Interactive 1D/2D Transfer Function (patched v3)')

        self.frame = QtWidgets.QFrame()
        vlay = QtWidgets.QVBoxLayout(self.frame)

        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        vlay.addWidget(self.vtkWidget)

        toolbar = QtWidgets.QHBoxLayout()
        self.log_checkbox = QtWidgets.QCheckBox('Log Histogram')
        self.log_checkbox.stateChanged.connect(self.toggle_log_histogram)
        toolbar.addWidget(self.log_checkbox)
        toolbar.addStretch(1)


        # --- NEW: Load Dataset button (minimal change) ---
        self.load_data_btn = QtWidgets.QPushButton("Load Dataset")
        self.load_data_btn.setToolTip("Choose a .vti or .vol file to load (remembers last folder).")
        self.load_data_btn.clicked.connect(self.load_volume_dialog)
        toolbar.addWidget(self.load_data_btn)

        self.tf_selector = QtWidgets.QComboBox()
        self.tf_selector.currentIndexChanged.connect(self.load_selected_tf)
        toolbar.addWidget(self.tf_selector)

        self.save_tf_btn = QtWidgets.QPushButton("Save TF")
        self.save_tf_btn.clicked.connect(self.save_current_tf)
        toolbar.addWidget(self.save_tf_btn)

        vlay.addLayout(toolbar)

        self.canvas_container = QtWidgets.QStackedWidget()
        vlay.addWidget(self.canvas_container)

        self.frame.setLayout(vlay)
        self.setCentralWidget(self.frame)

        # ----------------- original initial reader replaced by load_volume() -------------
        # use the same default file you had (keeps behavior identical)
        default_path = r"C:\Users\josde002\source\repos\volymrendering\data\head-binary-zlib.vti"
        # try to load default (load_volume is safe to call here)
        try:
            self.load_volume(default_path)
        except Exception as e:
            print("Failed to load default dataset:", e)
            # If default fails, create empty small arrays to allow UI to initialize
            empty = np.zeros((1,), dtype=np.float32)
            self.normalized_scalars = empty
            self.gradient_normalized = empty
            self.intensity_range = (0.0, 1.0)
            self.gradient_range = (0.0, 1.0)

        # 1D canvas (if load_volume created data this will show meaningful histogram)
        self.plot_canvas = TransferFunctionPlot(self.update_opacity_function, self.normalized_scalars, self.log_checkbox)
        self.tf1d_widget = TFCanvasWidget(self.plot_canvas, parent=self, label='Reset 1D View')
        self.canvas_container.addWidget(self.tf1d_widget)

        # 2D canvas
        hist2d, _, _ = np.histogram2d(self.normalized_scalars, self.gradient_normalized, bins=(256,256), range=((0,255),(0,255)))
        self.tf2d_canvas = TransferFunction2D(hist2d, self.intensity_range, self.gradient_range, self.log_checkbox)
        self.tf2d_widget = TFCanvasWidget(self.tf2d_canvas, parent=self, label='Reset 2D View')
        self.canvas_container.addWidget(self.tf2d_widget)

        # link 2D to 1D points
        try:
            self.tf2d_canvas.set_tf_state(self.plot_canvas.points_x, self.plot_canvas.points_y, self.plot_canvas.colors)
        except Exception:
            pass

        # --- FIX: initialize saved_tfs BEFORE updating combo box
        self.saved_tfs = {}
        self.current_tf_name = "Default"
        self.saved_tfs[self.current_tf_name] = (
            list(self.plot_canvas.points_x),
            list(self.plot_canvas.points_y),
            list(self.plot_canvas.colors)
        )
        self.update_tf_selector()

        self.canvas_container.setCurrentIndex(0)

        # VTK volume setup (same as before)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        # if load_volume set self.reader and it was a VTK reader, connect mapper here:
        try:
            if hasattr(self, 'reader') and self.reader is not None:
                self.mapper.SetInputConnection(self.reader.GetOutputPort())
        except Exception:
            pass

        self.color_function = vtk.vtkColorTransferFunction()
        # use intensity_range to compute color points if available
        try:
            raw_int_min, raw_int_max = self.intensity_range
        except Exception:
            raw_int_min, raw_int_max = 0.0, 1.0
        self.color_function.AddRGBPoint(raw_int_min, 0,0,0)
        self.color_function.AddRGBPoint(raw_int_min + 0.25*(raw_int_max-raw_int_min), 1,0.5,0.3)
        self.color_function.AddRGBPoint(raw_int_min + 0.50*(raw_int_max-raw_int_min), 1,1,0.9)
        self.color_function.AddRGBPoint(raw_int_min + 0.75*(raw_int_max-raw_int_min), 0.5,1,0.5)
        self.color_function.AddRGBPoint(raw_int_max, 1,1,1)

        self.opacity_function = vtk.vtkPiecewiseFunction()
        for x,y in zip(self.plot_canvas.points_x, self.plot_canvas.points_y):
            abs_val = raw_int_min + (x/255.0)*(raw_int_max-raw_int_min)
            self.opacity_function.AddPoint(abs_val, y)

        # Start from default 1D TF points
        for x, y, c in zip(self.plot_canvas.points_x, self.plot_canvas.points_y, self.plot_canvas.colors):
            abs_val = raw_int_min + (x / 255.0) * (raw_int_max - raw_int_min)
            self.opacity_function.AddPoint(abs_val, y)
            self.color_function.AddRGBPoint(abs_val, *c)

        vp = vtk.vtkVolumeProperty()
        vp.SetColor(self.color_function)
        vp.SetScalarOpacity(self.opacity_function)
        vp.ShadeOn()
        vp.SetInterpolationTypeToLinear()

        volume = vtk.vtkVolume()
        volume.SetMapper(self.mapper)
        volume.SetProperty(vp)
        self.renderer.AddVolume(volume)
        self.renderer.SetBackground(0.1,0.1,0.1)
        self.renderer.ResetCamera()
        self.interactor.Initialize()
        # don't call Start() here if embedding in some event loops, but to keep identical to your v3:
        self.interactor.Start()

        self.view_toggle = QtWidgets.QPushButton('Switch to 2D TF')
        self.view_toggle.setCheckable(True)
        self.view_toggle.toggled.connect(self.toggle_tf_view)
        vlay.addWidget(self.view_toggle)

        # initialize saved_tfs
        self.saved_tfs = {}
        self.load_tfs_from_disk()  # try loading existing TFs
        if not self.saved_tfs:
            # initialize default TF if none loaded
            self.current_tf_name = "Default"
            self.saved_tfs[self.current_tf_name] = (
                list(self.plot_canvas.points_x),
                list(self.plot_canvas.points_y()),
                list(self.plot_canvas.colors)
            )
        self.update_tf_selector()

    def toggle_tf_view(self, checked):
        idx = 1 if checked else 0
        self.canvas_container.setCurrentIndex(idx)
        self.view_toggle.setText('Switch to 1D TF' if checked else 'Switch to 2D TF')

    def update_opacity_function(self, xs, ys, colors):
        raw_int_min, raw_int_max = self.intensity_range
        #Clear old points
        self.color_function.RemoveAllPoints()
        self.opacity_function.RemoveAllPoints()

        #Add new points
        for x, y, c in zip(xs, ys, colors):
            abs_val = raw_int_min + (x/255.0)*(raw_int_max-raw_int_min)
            self.opacity_function.AddPoint(abs_val, y)
            self.color_function.AddRGBPoint(abs_val, *c)

         # Sync 2D overlay if exists
        try:
            self.tf2d_canvas.set_tf_state(xs, ys, colors)
        except Exception:
            pass

        #Trigger render
        self.vtkWidget.GetRenderWindow().Render()
        try:
            self.tf2d_canvas.set_tf_state(xs, ys, colors)
        except Exception:
            pass

    def toggle_log_histogram(self, state):
        try:
            self.plot_canvas._draw_plot()
        except Exception:
            pass
        try:
            self.tf2d_canvas._on_log_toggled(state)
        except Exception:
            pass

    def reset_all_views(self):
        try:
            self.plot_canvas.reset_view()
            self.tf2d_canvas.reset_view()
        except Exception:
            pass

    # --- TF saving/loading ---
    def save_current_tf(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Transfer Function", "Name:")
        if ok and name:
            # store a **deep copy** of the current canvas
            xs = list(self.plot_canvas.points_x)
            ys = list(self.plot_canvas.points_y)
            colors = [tuple(c) for c in self.plot_canvas.colors]
            self.saved_tfs[name] = (xs, ys, colors)
            self.update_tf_selector()
            self.tf_selector.setCurrentText(name)
            self.save_tfs_to_disk()

    def save_tfs_to_disk(self):
        """Serialize all TFs to JSON file."""
        out = {}
        for name, (xs, ys, colors) in self.saved_tfs.items():
            out[name] = {
                "x": list(xs),
                "y": list(ys),
                "colors": [list(c) for c in colors]
            }
        try:
            with open(self.TF_SAVE_FILE, "w") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            print(f"Failed to save TFs: {e}")

    def load_tfs_from_disk(self):
        """Load TFs from JSON file if it exists."""
        if os.path.exists(self.TF_SAVE_FILE):
            try:
                with open(self.TF_SAVE_FILE, "r") as f:
                    data = json.load(f)
                for name, tf in data.items():
                    xs = tf["x"]
                    ys = tf["y"]
                    colors = [tuple(c) for c in tf["colors"]]
                    self.saved_tfs[name] = (xs, ys, colors)
            except Exception as e:
                print(f"Failed to load TFs: {e}")

    def load_selected_tf(self, idx):
        if idx < 0:
            return
        name = self.tf_selector.itemText(idx)
        if name in self.saved_tfs:
            xs, ys, colors = self.saved_tfs[name]
            self.plot_canvas.points_x = list(xs)
            self.plot_canvas.points_y = list(ys)
            self.plot_canvas.colors = [tuple(c) for c in colors]
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw_plot()
            self.update_opacity_function(xs, ys, colors)

    def update_tf_selector(self):
        """Refresh the combo box with saved TF names."""
        self.tf_selector.blockSignals(True)  # prevent triggering currentIndexChanged
        self.tf_selector.clear()
        for name in self.saved_tfs.keys():
            self.tf_selector.addItem(name)
        self.tf_selector.blockSignals(False)

    # ------------------ New: Load dataset UI + logic ------------------
    def load_volume_dialog(self):
        """Open file dialog and load selected dataset. Remembers last folder."""
        start_dir = ""
        last_file = os.path.join(os.path.dirname(__file__), self.LAST_DIR_FILE)
        if os.path.exists(last_file):
            try:
                with open(last_file, "r") as f:
                    start_dir = f.read().strip()
            except Exception:
                start_dir = ""
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Volume Dataset", start_dir or "",
            "VTI Files (*.vti);;VOL/RAW Files (*.vol *.raw);;MHD Files (*.mhd);;All Files (*)"
        )
        if not file_name:
            return
        # store last dir
        try:
            with open(last_file, "w") as f:
                f.write(os.path.dirname(file_name))
        except Exception:
            pass
        # load
        try:
            self.load_volume(file_name)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Failed", f"Failed to load {file_name}:\n{e}")

    def _ask_raw_settings(self, fname):
        """
        Ask the user for raw/.vol settings: dims and dtype and byte order.
        Returns tuple (dims, dtype, byte_order) or None if cancelled.
        dtype -> numpy dtype string, byte_order -> 'little'/'big'
        """
        # dims
        dims_text, ok = QtWidgets.QInputDialog.getText(
            self, "Raw / .vol settings",
            "Enter dimensions as width,height,depth (e.g. 256,256,113):"
        )
        if not ok or not dims_text:
            return None
        try:
            parts = [int(p.strip()) for p in dims_text.split(",")]
            if len(parts) != 3:
                raise ValueError("Expected three integers")
            dims = tuple(parts)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Invalid input", f"Invalid dimensions: {e}")
            return None
        # dtype
        dtype_items = ["uint8", "uint16", "float32"]
        dtype_choice, ok = QtWidgets.QInputDialog.getItem(self, "Raw / .vol settings", "Data type:", dtype_items, 0, False)
        if not ok:
            return None
        dtype = dtype_choice
        # byte order
        bo_items = ["little", "big"]
        bo_choice, ok = QtWidgets.QInputDialog.getItem(self, "Raw / .vol settings", "Byte order:", bo_items, 0, False)
        if not ok:
            return None
        byte_order = bo_choice
        return dims, dtype, byte_order

    def load_volume(self, file_path):
        """
        Load .vti, .mhd, .raw, .vol (raw) datasets.
        Sets self.reader (if VTK reader used) or self.image_data (for raw created vtkImageData).
        Updates normalized scalars, gradient arrays, and updates canvases and mapper if present.
        """
        ext = os.path.splitext(file_path)[1].lower()
        image_data = None
        reader = None

        if ext == ".vti":
            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()
            self.reader = reader

        elif ext == ".mhd":
            reader = vtk.vtkMetaImageReader()
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()
            self.reader = reader

        elif ext in (".raw", ".vol"):
            # Ask user for dims/dtype/byte-order
            settings = self._ask_raw_settings(file_path)
            if settings is None:
                raise RuntimeError("Raw/.vol load cancelled or invalid settings.")
            dims, dtype_str, byte_order = settings
            dtype = np.dtype(dtype_str)
            # read file
            with open(file_path, "rb") as f:
                data = f.read()
            arr = np.frombuffer(data, dtype=dtype)
            expected = dims[0] * dims[1] * dims[2]
            if arr.size != expected:
                # try swapping byte order if endianness mismatch for 16/32-bit
                if dtype.itemsize > 1:
                    if byte_order == "big":
                        arr = arr.byteswap().newbyteorder()
                if arr.size != expected:
                    raise RuntimeError(f"Data size mismatch: expected {expected} elements, got {arr.size}. Check dims/type.")
            arr = arr.reshape(dims[::-1])  # VTK expects z-fast? keep consistent with your prior usage
            # create vtkImageData
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(dims[0], dims[1], dims[2])
            # assume single component
            if dtype_str == "uint8":
                vtk_type = vtk.VTK_UNSIGNED_CHAR
            elif dtype_str == "uint16":
                vtk_type = vtk.VTK_UNSIGNED_SHORT
            elif dtype_str == "float32":
                vtk_type = vtk.VTK_FLOAT
            else:
                vtk_type = vtk.VTK_UNSIGNED_CHAR
            vtk_data.AllocateScalars(vtk_type, 1)
            # convert and set scalars
            flat = np.ascontiguousarray(arr.ravel(order='C'))
            vtk_arr = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=None)
            vtk_data.GetPointData().SetScalars(vtk_arr)
            image_data = vtk_data
            self.reader = None  # no VTK reader in this case

        else:
            raise RuntimeError(f"Unsupported extension: {ext}")

        # compute numpy scalars and gradient arrays
        try:
            np_scalars = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars()).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to extract scalars: {e}")

        # gradient: prefer to use a VTK filter that accepts vtkImageData
        grad_filter = vtk.vtkImageGradientMagnitude()
        # if we have a reader, feed from its output port; else feed from image_data
        try:
            if hasattr(self, 'reader') and self.reader is not None:
                grad_filter.SetInputConnection(self.reader.GetOutputPort())
            else:
                grad_filter.SetInputData(image_data)
            grad_filter.Update()
            np_gradient = numpy_support.vtk_to_numpy(grad_filter.GetOutput().GetPointData().GetScalars()).astype(np.float32)
        except Exception:
            # fallback to zeros
            np_gradient = np.zeros_like(np_scalars, dtype=np.float32)

        # store for later usage
        self.raw_scalars = np_scalars
        raw_int_min, raw_int_max = np_scalars.min(), np_scalars.max()
        self.intensity_range = (raw_int_min, raw_int_max)
        if raw_int_max - raw_int_min == 0:
            self.normalized_scalars = np.zeros_like(np_scalars)
        else:
            self.normalized_scalars = 255.0 * (np_scalars - raw_int_min) / (raw_int_max - raw_int_min)

        raw_grad_min, raw_grad_max = np_gradient.min(), np_gradient.max()
        self.gradient_range = (raw_grad_min, raw_grad_max)
        if raw_grad_max - raw_grad_min == 0:
            self.gradient_normalized = np.zeros_like(np_gradient)
        else:
            self.gradient_normalized = 255.0 * (np_gradient - raw_grad_min) / (raw_grad_max - raw_grad_min)

        # update canvases if they already exist (runtime load), otherwise callers will create them
        try:
            if hasattr(self, 'plot_canvas') and self.plot_canvas is not None:
                self.plot_canvas.hist_data = self.normalized_scalars
                self.plot_canvas._draw_plot()
        except Exception:
            pass

        try:
            hist2d, _, _ = np.histogram2d(self.normalized_scalars, self.gradient_normalized, bins=(256,256), range=((0,255),(0,255)))
            if hasattr(self, 'tf2d_canvas') and self.tf2d_canvas is not None:
                self.tf2d_canvas.raw = hist2d
                # update display (respecting log toggle)
                if self.log_checkbox.isChecked():
                    self.tf2d_canvas._on_log_toggled(True)
                else:
                    self.tf2d_canvas._draw()
        except Exception:
            pass

        # connect mapper if present
        try:
            if hasattr(self, 'mapper') and self.mapper is not None and hasattr(self, 'reader') and self.reader is not None:
                self.mapper.SetInputConnection(self.reader.GetOutputPort())
            elif hasattr(self, 'mapper') and self.mapper is not None and image_data is not None and self.reader is None:
                # raw-created vtkImageData -> set input data
                try:
                    self.mapper.SetInputData(image_data)
                except Exception:
                    pass
        except Exception:
            pass

        # refresh renderer
        try:
            self.renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            pass

        # hold references
        self.image_data = image_data

        return True


# --------------------------- Main ---------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolumeApp()
    window.show()
    sys.exit(app.exec_())
