from base_transfer_function import BaseTransferFunction
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt  # ADD THIS IMPORT


class TransferFunctionPlot(BaseTransferFunction):
    def __init__(self, update_callback, scalar_data, log_toggle_checkbox=None):
        super().__init__(figsize=(5, 2))
        
        self.update_callback = update_callback
        self.log_toggle_checkbox = log_toggle_checkbox
        self.histogram_scaling = False
        self._hist_scale_start_y = None

        # Initialize with histogram-based points
        self.hist_data = scalar_data
        self._initialize_from_histogram()

        # Set proper initial y-limits for 1D TF
        self._cached_ylim = (0.0, 1.0)
        
        self._draw()

    def _initialize_from_histogram(self):
        """Initialize TF points from data histogram."""
        hist, bins = np.histogram(self.hist_data, bins=256, range=(0, 255))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        peaks = np.where(hist > hist.max() * 0.05)[0]
        self.points_x = list(bin_centers[peaks])
        self.points_y = list(np.clip(hist[peaks] / hist.max(), 0.0, 1.0))
        self.colors = [(1.0, 1.0, 1.0) for _ in self.points_x]

    # ===== COORDINATE TRANSFORMATIONS =====
    
    def _data_to_display(self, x):
        """Convert data intensity to display coordinates."""
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            return 255.0 * (np.log1p(x) / np.log1p(255.0))
        return float(x)

    def _display_to_data(self, x_disp):
        """Convert display coordinates to data intensity."""
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            frac = float(x_disp) / 255.0
            val = np.expm1(frac * np.log1p(255.0))
            return float(np.clip(val, 0.0, 255.0))
        return float(np.clip(x_disp, 0.0, 255.0))

    def _get_display_coords(self, x, y):
        """Override for 1D coordinate transformation."""
        return self._data_to_display(x), y

    def _get_data_coords(self, x_disp, y_disp):
        """Override for 1D coordinate transformation."""
        return self._display_to_data(x_disp), y_disp

    # ===== 1D-SPECIFIC VIEW MANAGEMENT =====
    
    def reset_view(self):
        """Override to set proper 1D y-limits."""
        self._reset_view_requested = True
        self._cached_xlim = (0.0, 255.0)
        self._cached_ylim = (0.0, 1.0)  # 1D TF has y-range 0-1
        self._draw()

    def _apply_view_limits(self):
        """Override to apply 1D-specific view limits."""
        if self._reset_view_requested:
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 1)  # 1D TF has y-range 0-1
            self._reset_view_requested = False
        else:
            self.ax.set_xlim(*self._cached_xlim)
            self.ax.set_ylim(*self._cached_ylim)

    # ===== 1D-SPECIFIC DRAWING =====
    
    def _draw(self):
        """Draw the 1D transfer function with histogram."""
        curr_xlim = self._cached_xlim
        curr_ylim = self._cached_ylim
        
        self.ax.clear()

        # Draw histogram
        self._draw_histogram()

        # Draw TF curve and points
        self._draw_tf_curve()

        # Apply view limits and formatting
        self._apply_view_limits()
        self._update_view_limits()

        # Add labels and grid
        self.ax.set_title('1D Transfer Function with Histogram')
        self.ax.set_xlabel('Intensity (display)')
        
        # Set proper y-label based on log scale
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            self.ax.set_ylabel('log(1 + count)')
        else:
            self.ax.set_ylabel('Normalized Count')
            
        self.ax.grid(True)

        # Format ticks - use different logic for 1D
        self._format_1d_ticks()

        self.draw()

    def _draw_histogram(self):
        """Draw the histogram background."""
        hist, bin_edges = np.histogram(self.hist_data, bins=150, range=(0.0, 255.0))
        hist = hist.astype(np.float64)
        if self.log_toggle_checkbox and self.log_toggle_checkbox.isChecked():
            hist = np.log1p(hist)
            # Normalize log histogram for display
            if hist.max() > 0:
                hist /= hist.max()
        else:
            # Normalize regular histogram
            if hist.max() > 0:
                hist /= hist.max()
                
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.ax.plot(bin_centers, hist, color='gray', linewidth=1, alpha=0.4)
        self.ax.fill_between(bin_centers, hist, color='lightgray', alpha=0.5)

    def _draw_tf_curve(self):
        """Draw the TF curve and control points."""
        if not self.points_x:
            return
            
        display_xs = [self._data_to_display(x) for x in self.points_x]
        
        # Draw lines between points
        for i in range(len(display_xs) - 1):
            self.ax.plot(display_xs[i:i+2], self.points_y[i:i+2], color='orange', linewidth=2)
        
        # Draw control points
        for xd, y, c in zip(display_xs, self.points_y, self.colors):
            self.ax.plot(xd, y, 'o', color=c, markersize=8, mec='k')

    def _format_1d_ticks(self):
        """Format ticks specifically for 1D transfer function."""
        x_range = self._cached_xlim[1] - self._cached_xlim[0]
        y_range = self._cached_ylim[1] - self._cached_ylim[0]
        
        # X-axis formatting (same as base)
        if x_range <= 10:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        elif x_range <= 50:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        elif x_range <= 100:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        else:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            
        # Y-axis formatting specific to 1D (range 0-1)
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
        """Override to handle 1D-specific histogram scaling."""
        if event.inaxes != self.ax:
            return
            
        # 1D-specific histogram scaling behavior
        if event.x is not None and event.x <= 40:
            self.histogram_scaling = True
            self._hist_scale_start_y = event.y
            return

        # Use base class for point interactions
        super().on_press(event)

    def on_motion(self, event):
        """Override to handle 1D-specific histogram scaling."""
        if self.histogram_scaling:
            if event.y is None or self._hist_scale_start_y is None:
                return
            dy = event.y - self._hist_scale_start_y
            factor = np.exp(dy / 200.0)
            self._hist_scale_start_y = event.y
            
            # Adjust y-limits for histogram scaling
            ymin, ymax = self.ax.get_ylim()
            self.ax.set_ylim(ymin * factor, ymax * factor)
            self._update_view_limits()
            self.draw()
            return

        # Use base class for point dragging
        super().on_motion(event)

    def on_release(self, event):
        """Override to handle 1D-specific histogram scaling."""
        super().on_release(event)
        self.histogram_scaling = False
        self._hist_scale_start_y = None

    # ===== 1D-SPECIFIC NOTIFICATION =====
    
    def _notify_app(self):
        """Override to use the provided callback."""
        if self.update_callback:
            self.update_callback(self.points_x, self.points_y, self.colors)
        else:
            super()._notify_app()

    def on_scroll(self, event):
        """Override scroll for 1D-specific behavior."""
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
            
        if event.xdata is None or event.ydata is None:
            return
            
        if shift:
            # Vertical zoom only - but constrained to reasonable values
            ydata = event.ydata if event.ydata is not None else 0.5
            ymin, ymax = self.ax.get_ylim()
            new_ymin = ydata + (ymin - ydata) * scale
            new_ymax = ydata + (ymax - ydata) * scale
            
            # Constrain y-limits to prevent extreme values
            if new_ymax - new_ymin > 0.001:  # Minimum range
                self.ax.set_ylim(new_ymin, new_ymax)
        else:
            # Horizontal zoom only (default)
            xdata = event.xdata
            xmin, xmax = self.ax.get_xlim()
            new_xmin = max(0.0, xdata + (xmin - xdata) * scale)
            new_xmax = min(255.0, xdata + (xmax - xdata) * scale)
            if new_xmax - new_xmin > 1e-6:
                self.ax.set_xlim(new_xmin, new_xmax)

        self._update_view_limits()
        self._draw()