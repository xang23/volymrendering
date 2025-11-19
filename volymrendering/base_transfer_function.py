import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt


class BaseTransferFunction(FigureCanvas):
    """Base class for all transfer function widgets with common functionality."""
    
    def __init__(self, figsize=(5, 3), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        
        # Common TF state
        self.points_x = []
        self.points_y = []
        self.colors = []
        
        # Common interaction state
        self.selected_index = None
        self.dragging = False
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 255.0)
        self._reset_view_requested = False
        
        # Common events
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('scroll_event', self.on_scroll)

    # ===== COMMON TF STATE MANAGEMENT =====
    
    def set_tf_state(self, xs, ys, colors):
        """Set TF state from points and colors."""
        self.points_x = list(xs)
        self.points_y = list(ys)
        self.colors = list(colors)
        self._sort_points_with_colors()
        self._draw()

    def get_tf_state(self):
        """Get current TF state."""
        return self.points_x, self.points_y, self.colors

    def _sort_points_with_colors(self):
        """Sort points by x-coordinate while maintaining color association."""
        if self.points_x:
            zipped = sorted(zip(self.points_x, self.points_y, self.colors), key=lambda p: p[0])
            self.points_x, self.points_y, self.colors = map(list, zip(*zipped))

    def add_point(self, x, y, color=None):
        """Add a new control point."""
        if color is None:
            color = (1.0, 1.0, 1.0)  # Default white
        self.points_x.append(x)
        self.points_y.append(y)
        self.colors.append(color)
        self._sort_points_with_colors()
        self._draw()
        self._notify_app()

    def remove_point(self, index):
        """Remove a control point by index."""
        if 0 <= index < len(self.points_x) and index not in (0, len(self.points_x)-1):
            self.points_x.pop(index)
            self.points_y.pop(index)
            self.colors.pop(index)
            self._draw()
            self._notify_app()

    def update_point(self, index, x, y):
        """Update position of a control point."""
        if 0 <= index < len(self.points_x):
            # Lock endpoints to boundaries
            if index == 0:
                x = 0.0
            elif index == len(self.points_x) - 1:
                x = 255.0
                
            self.points_x[index] = x
            self.points_y[index] = y
            self._sort_points_with_colors()
            self._draw()
            self._notify_app()

    def update_point_color(self, index, color):
        """Update color of a control point."""
        if 0 <= index < len(self.colors):
            self.colors[index] = color
            self._draw()
            self._notify_app()

    # ===== COMMON VIEW MANAGEMENT =====
    
    def reset_view(self):
        """Reset to default view."""
        self._reset_view_requested = True
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 255.0)
        self._draw()

    def _update_view_limits(self):
        """Update cached view limits."""
        self._cached_xlim = self.ax.get_xlim()
        self._cached_ylim = self.ax.get_ylim()

    def _apply_view_limits(self):
        """Apply cached or reset view limits."""
        if self._reset_view_requested:
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 255)
            self._reset_view_requested = False
        else:
            self.ax.set_xlim(*self._cached_xlim)
            self.ax.set_ylim(*self._cached_ylim)

    # ===== COMMON POINT INTERACTION =====
    
    def _closest_point(self, event, pixel_tol=10):
        """Find closest control point to mouse position."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
            
        display_points = self._get_display_points()
        if display_points is None or len(display_points) == 0:
            return None
            
        pix = self.ax.transData.transform(display_points)
        ev = np.array([event.x, event.y])
        dists = np.hypot(pix[:,0]-ev[0], pix[:,1]-ev[1])
        
        idx = int(np.argmin(dists))
        if dists[idx] <= pixel_tol:
            return idx
        return None

    def _get_display_coords(self, x, y):
        """Convert data coordinates to display coordinates."""
        # Default implementation - can be overridden for coordinate transformations
        return x, y

    def _get_data_coords(self, x_disp, y_disp):
        """Convert display coordinates to data coordinates."""
        # Default implementation - can be overridden for coordinate transformations
        return x_disp, y_disp

    def _get_display_points(self):
        """Convert TF points to display coordinates for point picking."""
        if not self.points_x:
            return np.empty((0, 2))
        display_points = []
        for x, y in zip(self.points_x, self.points_y):
            dx, dy = self._get_display_coords(x, y)
            display_points.append([dx, dy])
        return np.array(display_points)

    # ===== COMMON EVENT HANDLERS =====
    
    def on_press(self, event):
        """Handle mouse press events."""
        if event.inaxes != self.ax:
            return

        idx = self._closest_point(event)

        # double-click: add point
        if getattr(event, 'dblclick', False):
            if event.xdata is None or event.ydata is None:
                return
                
            x_data, y_data = self._get_data_coords(event.xdata, event.ydata)
            x_clipped = float(np.clip(x_data, 0.0, 255.0))
            y_clipped = float(np.clip(y_data, 0.0, 1.0))
            
            color = (1.0, 1.0, 1.0)
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
                
            if mods & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                    
            self.add_point(x_clipped, y_clipped, color)
            return

        # right-click: delete point
        if idx is not None and getattr(event, 'button', None) == 3:
            self.remove_point(idx)
            return

        # left-click: select point for dragging or change color
        if idx is not None and getattr(event, 'button', None) == 1:
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
                
            if mods & Qt.ShiftModifier:
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    self.update_point_color(idx, (qcolor.redF(), qcolor.greenF(), qcolor.blueF()))
            else:
                self.selected_index = idx
                self.dragging = True

    def on_motion(self, event):
        """Handle mouse motion events."""
        if not self.dragging or self.selected_index is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        x_data, y_data = self._get_data_coords(event.xdata, event.ydata)
        x_clipped = float(np.clip(x_data, 0.0, 255.0))
        y_clipped = float(np.clip(y_data, 0.0, 1.0))
        
        self.update_point(self.selected_index, x_clipped, y_clipped)
        
        # Update selected index after sorting
        try:
            self.selected_index = min(range(len(self.points_x)), 
                                    key=lambda i: abs(self.points_x[i] - x_clipped))
        except Exception:
            self.selected_index = None
            
        self._update_view_limits()

    def on_release(self, event):
        """Handle mouse release events."""
        self.selected_index = None
        self.dragging = False

    def on_scroll(self, event):
        """Handle scroll events for zooming."""
        if event.inaxes != self.ax:
            return
            
        step = getattr(event, 'step', None)
        if step is None:
            step = 1 if getattr(event, 'button', None) == 'up' else -1
            
        base = 0.9
        scale = base ** step
        
        try:
            shift = bool(event.guiEvent.modifiers() & Qt.ShiftModifier)
            ctrl = bool(event.guiEvent.modifiers() & Qt.ControlModifier)
        except Exception:
            shift = False
            ctrl = False
            
        if event.xdata is None or event.ydata is None:
            return
            
        if shift:
            # Vertical zoom only
            ydata = event.ydata
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(ydata + (ymin - ydata) * scale, 
                            ydata + (ymax - ydata) * scale)
        elif ctrl:
            # Both axes zoom
            xdata = event.xdata
            ydata = event.ydata
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_xlim(xdata + (xmin - xdata) * scale, 
                            xdata + (xmax - xdata) * scale)
            self.ax.set_ylim(ydata + (ymin - ydata) * scale, 
                            ydata + (ymax - ydata) * scale)
        else:
            # Horizontal zoom only (default)
            xdata = event.xdata
            xmin, xmax = self.ax.get_xlim()
            self.ax.set_xlim(xdata + (xmin - xdata) * scale, 
                            xdata + (xmax - xdata) * scale)

        self._update_view_limits()
        self._draw()

    # ===== COMMON UTILITIES =====
    
    def _format_ticks(self, x_range, y_range):
        """Format ticks based on current zoom level."""
        if x_range <= 10:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        elif x_range <= 50:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        elif x_range <= 100:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        else:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(50))
        if y_range <= 10:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        elif y_range <= 50:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        elif y_range <= 100:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        else:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(50))

    def _notify_app(self):
        """Notify the main app about TF changes."""
        w = self.parent()
        while w is not None and not hasattr(w, 'update_opacity_function'):
            w = w.parent()
        if w is not None and hasattr(w, 'update_opacity_function'):
            w.update_opacity_function(self.points_x, self.points_y, self.colors)

    # ===== ABSTRACT METHODS =====
    
    def _draw(self):
        """Draw the canvas - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _draw")