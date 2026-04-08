from base_transfer_function import BaseTransferFunction
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt


class TransferFunctionPlot(BaseTransferFunction):
    def __init__(self, update_callback, scalar_data, log_toggle_checkbox=None):
        super().__init__(figsize=(5, 2))
        
        self.update_callback = update_callback
        self.log_toggle_checkbox = log_toggle_checkbox
        self.histogram_scaling = False
        self._hist_scale_start_y = None

        # Store original data
        self.hist_data = scalar_data
        
        # Initialize with histogram-based points
        self._initialize_from_histogram()

        # Set proper initial y-limits for 1D TF
        self._cached_ylim = (0.0, 1.0)
        
        # Connect log checkbox if provided
        if self.log_toggle_checkbox is not None:
            self.log_toggle_checkbox.stateChanged.connect(self._on_log_toggled)
        
        self._draw()

    def _initialize_from_histogram(self):
        """Initialize TF points from data histogram."""
        hist, bins = np.histogram(self.hist_data, bins=256, range=(0, 255))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        peaks = np.where(hist > hist.max() * 0.05)[0]
        
        if len(peaks) < 2:
            self.points_x = [0.0, 255.0]
            self.points_y = [0.0, 1.0]
            self.colors = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
        else:
            self.points_x = list(bin_centers[peaks])
            self.points_y = list(np.clip(hist[peaks] / hist.max(), 0.0, 1.0))
            self.colors = [(1.0, 1.0, 1.0) for _ in self.points_x]

    def _on_log_toggled(self, state):
        """Handle log scale toggle - redraw without changing point positions"""
        self._draw()

    # ===== COORDINATE TRANSFORMATIONS =====
    # CRITICAL FIX: Keep display coordinates consistent with 2D view!
    
    def _get_display_coords(self, x, y):
        """Convert data coordinates to display coordinates.
        For 1D TF, we want to see opacity directly, not scaled.
        """
        # x: intensity [0,255] - keep as is
        # y: opacity [0,1] - keep as is
        return x, y

    def _get_data_coords(self, x_disp, y_disp):
        """Convert display coordinates to data coordinates."""
        # x_disp: displayed intensity [0,255] - keep as is
        # y_disp: displayed opacity [0,1] - keep as is
        return float(np.clip(x_disp, 0.0, 255.0)), float(np.clip(y_disp, 0.0, 1.0))

    # ===== 1D-SPECIFIC VIEW MANAGEMENT =====
    
    def reset_view(self):
        """Reset to default view."""
        self._reset_view_requested = True
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 1.0)
        self._draw()

    def _apply_view_limits(self):
        """Apply view limits."""
        if self._reset_view_requested:
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 1)
            self._reset_view_requested = False
        else:
            self.ax.set_xlim(*self._cached_xlim)
            self.ax.set_ylim(*self._cached_ylim)

    # ===== 1D-SPECIFIC DRAWING =====
    
    def _draw(self):
        """Draw the 1D transfer function with histogram."""
        # Store current view
        curr_xlim = self._cached_xlim
        curr_ylim = self._cached_ylim
        
        self.ax.clear()

        # Draw histogram (background)
        self._draw_histogram()

        # Draw TF curve and points
        self._draw_tf_curve()

        # Apply view limits
        self._apply_view_limits()
        self._update_view_limits()

        # Add labels and grid
        self.ax.set_title('1D Transfer Function')
        self.ax.set_xlabel('Intensity')
        self.ax.set_ylabel('Opacity')
        self.ax.grid(True, alpha=0.3)

        # Format ticks
        self._format_1d_ticks()

        self.draw()

    def _draw_histogram(self):
        """Draw the histogram background."""
        if self.hist_data is None:
            return
            
        # Always compute histogram in intensity space [0,255]
        hist, bin_edges = np.histogram(self.hist_data, bins=256, range=(0.0, 255.0))
        hist = hist.astype(np.float64)
        
        # Apply log scaling if requested (for display only!)
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            hist = np.log1p(hist)
            
        # Normalize for display
        if hist.max() > 0:
            hist = hist / hist.max()
            
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Draw histogram
        self.ax.fill_between(bin_centers, 0, hist, alpha=0.3, color='gray')
        self.ax.plot(bin_centers, hist, alpha=0.5, color='black', linewidth=1)

    def _draw_tf_curve(self):
        """Draw the TF curve and control points."""
        if len(self.points_x) < 2:
            return
        
        # Sort points by x
        sorted_pairs = sorted(zip(self.points_x, self.points_y, self.colors), 
                             key=lambda p: p[0])
        xs, ys, colors = zip(*sorted_pairs)
        
        # Draw lines between points
        self.ax.plot(xs, ys, '-', color='orange', linewidth=2, zorder=5)
        
        # Draw control points
        for x, y, c in zip(xs, ys, colors):
            self.ax.plot(x, y, 'o', color=c, markersize=8, 
                        markeredgecolor='black', zorder=10)

    def _format_1d_ticks(self):
        """Format ticks for 1D transfer function."""
        x_range = self._cached_xlim[1] - self._cached_xlim[0]
        y_range = self._cached_ylim[1] - self._cached_ylim[0]
        
        # X-axis formatting (intensity)
        if x_range <= 10:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        elif x_range <= 50:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        elif x_range <= 100:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        else:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            
        # Y-axis formatting (opacity)
        if y_range <= 0.1:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
        elif y_range <= 0.2:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        elif y_range <= 0.5:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        else:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    # ===== 1D-SPECIFIC EVENT HANDLING =====
    
    def on_press(self, event):
        """Handle mouse press events for 1D TF with shift+click color picker."""
        if event.inaxes != self.ax:
            return

        # Find closest point
        idx = self._closest_point(event)

        # Check for shift+click for color picking (using guiEvent modifiers)
        try:
            mods = event.guiEvent.modifiers()
            shift_pressed = (mods & Qt.ShiftModifier)
        except:
            shift_pressed = False
    
        # SHIFT+CLICK on any point - change its color
        if shift_pressed and idx is not None:
            # Open color dialog
            from PyQt5 import QtWidgets
            qcolor = QtWidgets.QColorDialog.getColor()
            if qcolor.isValid():
                new_color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                self.update_point_color(idx, new_color)
                self._draw()
                self._notify_app()
            return

        # Double-click: add point (without shift)
        if getattr(event, 'dblclick', False):
            if event.xdata is None or event.ydata is None:
                return
            
            x_clipped = float(np.clip(event.xdata, 0.0, 255.0))
            y_clipped = float(np.clip(event.ydata, 0.0, 1.0))
        
            # Default white color for new points
            color = (1.0, 1.0, 1.0)
        
            # If shift is also pressed during double-click, open color picker for new point
            if shift_pressed:
                from PyQt5 import QtWidgets
                qcolor = QtWidgets.QColorDialog.getColor()
                if qcolor.isValid():
                    color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
        
            self.add_point(x_clipped, y_clipped, color)
            self._draw()
            self._notify_app()
            return

        # Right-click: delete point (but not endpoints)
        if idx is not None and getattr(event, 'button', None) == 3:
            if idx not in (0, len(self.points_x)-1):
                self.remove_point(idx)
                self._draw()
                self._notify_app()
            return

        # Left-click: select point for dragging (without shift)
        if idx is not None and getattr(event, 'button', None) == 1:
            self.selected_index = idx
            self.dragging = True
            self._draw()
            return
    
        # Click on empty space - deselect
        self.selected_index = None
        self.dragging = False
        self._draw()

    def on_motion(self, event):
        """Handle mouse motion for point dragging."""
        if not self.dragging or self.selected_index is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
            
        x_clipped = float(np.clip(event.xdata, 0.0, 255.0))
        y_clipped = float(np.clip(event.ydata, 0.0, 1.0))
        
        self.update_point(self.selected_index, x_clipped, y_clipped)
        
        # Update selected index after sorting
        try:
            self.selected_index = min(range(len(self.points_x)), 
                                    key=lambda i: abs(self.points_x[i] - x_clipped))
        except:
            self.selected_index = None
            
        self._update_view_limits()

    # ===== 1D-SPECIFIC NOTIFICATION =====
    
    def _notify_app(self):
        """Notify app about TF changes."""
        if self.update_callback:
            self.update_callback(self.points_x, self.points_y, self.colors)

    def _closest_point(self, event):
        """Find closest point to mouse click."""
        if not self.points_x:
            return None
        
        min_dist = 10  # 10 pixels threshold
        closest_idx = None
    
        for i, (x, y) in enumerate(zip(self.points_x, self.points_y)):
            # Convert to display coordinates for distance check
            x_disp, y_disp = self._get_display_coords(x, y)
            dist = np.sqrt((event.xdata - x_disp)**2 + (event.ydata - y_disp)**2)
        
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
    
        return closest_idx

    def update_point_color(self, idx, new_color):
        """Update color of a specific point."""
        if 0 <= idx < len(self.colors):
            self.colors[idx] = new_color
            self._draw()
            self._notify_app()