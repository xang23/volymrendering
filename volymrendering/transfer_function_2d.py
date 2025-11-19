from base_transfer_function import BaseTransferFunction
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt  # Add this import


class TransferFunction2D(BaseTransferFunction):
    def __init__(self, raw_hist2d, intensity_range, gradient_range, log_toggle_checkbox=None):
        super().__init__(figsize=(5, 5))
    
        self.raw = raw_hist2d
        # FORCE 0-255 RANGES REGARDLESS OF INPUT
        self.int_range = (0, 255)
        self.grad_range = (0, 255)
        self.log_checkbox = log_toggle_checkbox

        # Setup the 2D histogram display
        self._setup_histogram_display()

        if self.log_checkbox is not None:
            self.log_checkbox.stateChanged.connect(self._on_log_toggled)

        # SET INITIAL VIEW TO 0-255
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
    
        self._draw()

    def _setup_histogram_display(self):
        """Setup the 2D histogram visualization."""
        disp = self._get_display_data()
        norm = LogNorm() if (self.log_checkbox and self.log_checkbox.isChecked()) else None
        self.im = self.ax.imshow(
            disp.T, origin='lower', cmap='hot', norm=norm,
            interpolation='nearest', extent=(0,255,0,255), aspect='auto'
        )

        # Add proper axes and labels
        self.ax.set_xlabel('Intensity')
        self.ax.set_ylabel('Gradient Magnitude')
        self.ax.set_title('2D Transfer Function')
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Setup TF overlay
        self.tf_line, = self.ax.plot([], [], color='orange', linewidth=2)
        self.tf_scatter = self.ax.scatter([], [], s=40, edgecolor='k', zorder=10)

    def _get_display_data(self):
        """Prepare data for display with optional log scaling."""
        arr = self.raw.astype(np.float64)
        if self.log_checkbox and self.log_checkbox.isChecked():
            arr = np.log1p(arr)
        m = arr.max()
        if m > 0:
            arr /= m
        return arr

    def _on_log_toggled(self, state):
        """Handle log scale toggle."""
        disp = self._get_display_data()
        if self.log_checkbox and self.log_checkbox.isChecked():
            self.im.set_norm(LogNorm())
        else:
            self.im.set_norm(None)
        self.im.set_data(disp.T)
        self._draw()

    # ===== 2D-SPECIFIC COORDINATE TRANSFORMATIONS =====
    
    def _get_display_coords(self, x, y):
        """Convert 2D TF points to display coordinates."""
        return x, 255.0 * y  # Scale y from [0,1] to [0,255] for display

    def _get_data_coords(self, x_disp, y_disp):
        """Convert display coordinates to 2D TF data coordinates."""
        return x_disp, y_disp / 255.0  # Scale y from [0,255] to [0,1] for data

    # ===== 2D-SPECIFIC DRAWING =====
    
    def _draw(self):
        """Draw the 2D transfer function."""
        curr_xlim = self._cached_xlim
        curr_ylim = self._cached_ylim
    
        # Update histogram display
        self.im.set_data(self._get_display_data().T)

        # Update TF overlay
        self._draw_tf_overlay()

        # Apply view limits WITH ZOOM CONSTRAINTS
        self._apply_view_limits()
        self._update_view_limits()

        # Adjust grid based on zoom level - IMPROVED LOGIC
        x_range = self._cached_xlim[1] - self._cached_xlim[0]
        y_range = self._cached_ylim[1] - self._cached_ylim[0]
    
        # Always show grid, but adjust density
        if x_range <= 50 and y_range <= 50:
            self.ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        elif x_range <= 100 and y_range <= 100:
            self.ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        else:
            self.ax.grid(True, alpha=0.1, linestyle=':', linewidth=0.2)  # Keep grid for large zooms

        # Format ticks
        self._format_ticks(x_range, y_range)

        self.draw()

    def _draw_tf_overlay(self):
        """Draw the TF curve and points overlay."""
        if len(self.points_x) == 0:
            self.tf_line.set_data([], [])
            self.tf_scatter.set_offsets(np.empty((0, 2)))
        else:
            # Convert to display coordinates for drawing
            display_points = self._get_display_points()
            x = display_points[:, 0]
            y = display_points[:, 1]
            
            self.tf_line.set_data(x, y)
            self.tf_scatter.set_offsets(display_points)
            
            if self.colors and len(self.colors) == len(x):
                self.tf_scatter.set_facecolor(self.colors)
            else:
                self.tf_scatter.set_facecolor([(1.0, 1.0, 1.0)] * len(x))

    def _format_ticks(self, x_range, y_range):
        """Format ticks for 2D view - ADD THIS METHOD"""
        # X-axis formatting
        if x_range <= 10:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        elif x_range <= 50:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        elif x_range <= 100:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        else:
            self.ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            
        # Y-axis formatting
        if y_range <= 10:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        elif y_range <= 50:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        elif y_range <= 100:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        else:
            self.ax.yaxis.set_major_locator(plt.MultipleLocator(50))

    # ===== 2D-SPECIFIC EVENT HANDLING =====
    

    def set_tf_state(self, xs, ys, colors):
        """Override to ensure proper state setting and drawing."""
        self.points_x = list(xs)
        self.points_y = list(ys)
        self.colors = list(colors)
        self._sort_points_with_colors()
        self._draw()

    def _apply_view_limits(self):
        """Only prevent excessive zoom out, allow unlimited zoom in"""
        super()._apply_view_limits()
    
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        # ONLY PREVENT ZOOM OUT BEYOND 3x DATA RANGE
        max_zoom_out = 3.0
        max_range = 255 * max_zoom_out
    
        current_x_range = xlim[1] - xlim[0]
        current_y_range = ylim[1] - ylim[0]
    
        needs_adjustment = False
        new_xlim, new_ylim = xlim, ylim
    
        if current_x_range > max_range:
            center_x = (xlim[0] + xlim[1]) / 2
            new_xlim = (center_x - max_range/2, center_x + max_range/2)
            needs_adjustment = True
        
        if current_y_range > max_range:
            center_y = (ylim[0] + ylim[1]) / 2
            new_ylim = (center_y - max_range/2, center_y + max_range/2)
            needs_adjustment = True
        
        if needs_adjustment:
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)

    def reset_view(self):
        """Reset view to show full 0-255 range"""
        self.ax.set_xlim(0, 255)
        self.ax.set_ylim(0, 255)
        self._draw()