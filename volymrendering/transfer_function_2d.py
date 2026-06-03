from base_transfer_function import BaseTransferFunction
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
import PyQt5.QtWidgets as QtWidgets


class TransferFunction2D(BaseTransferFunction):
    def __init__(self, raw_hist2d, intensity_range, gradient_range, log_toggle_checkbox=None, x_label="Intensity", y_label="Gradient Magnitude"):
        super().__init__(figsize=(5, 5))
    
        self.raw = raw_hist2d
        # FORCE 0-255 RANGES REGARDLESS OF INPUT
        self.int_range = (0, 255)
        self.grad_range = (0, 255)
        self.log_checkbox = log_toggle_checkbox
        self.x_label = x_label
        self.y_label = y_label


        # Initialize with default TF
        self.points_x = [0.0, 255.0]
        self.points_y = [0.0, 1.0]
        self.colors = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]

        # Setup the 2D histogram display
        self._setup_histogram_display()

        if self.log_checkbox is not None:
            self.log_checkbox.stateChanged.connect(self._on_log_toggled)

        # SET INITIAL VIEW TO 0-255
        self.ax.set_xlim(0, 256)
        self.ax.set_ylim(0, 256)

    
        self._draw()

    def _setup_histogram_display(self):
        """Setup the 2D histogram visualization."""
        disp = self._get_display_data()
        self.im = self.ax.imshow(
            disp.T, origin='lower', cmap='hot',
            interpolation='bilinear', extent=(0,255,0,255), aspect='equal'
        )

        # Add proper axes and labels
        self.ax.set_xlabel('Intensity')
        self.ax.set_ylabel('Gradient Magnitude')
        self.ax.set_title('2D Transfer Function')
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Setup TF overlay
        self.tf_line, = self.ax.plot([], [], color='orange', linewidth=2, zorder=5)
        self.tf_scatter = self.ax.scatter([], [], s=40, edgecolor='k', zorder=10)

    def _get_display_data(self):
        """Prepare data for display with optional log scaling."""
        arr = self.raw.astype(np.float64)
        if self.log_checkbox and self.log_checkbox.isChecked():
            arr = np.log1p(arr)
        
        arr_min = arr.min()
        arr_max = arr.max()

        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)

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
        """Convert data coordinates to display coordinates.
        x: intensity [0,255] -> display x [0,255]
        y: opacity [0,1] -> display y [0,255] (mapped to gradient axis)
        """
        return x, y * 255.0

    def _get_data_coords(self, x_disp, y_disp):
        """Convert display coordinates to data coordinates.
        x_disp: display x [0,255] -> intensity [0,255]
        y_disp: display y [0,255] -> opacity [0,1]
        """
        # ===== ADD DEBUG HERE =====
        print(f"  🔄 _get_data_coords: disp=({x_disp:.1f}, {y_disp:.1f})", end="")
    
        result_x = float(np.clip(x_disp, 0.0, 255.0))
        result_y = float(np.clip(y_disp / 255.0, 0.0, 1.0))
    
        # ===== ADD DEBUG HERE =====
        print(f" → data=({result_x:.1f}, {result_y:.3f})")
    
        return result_x, result_y

    def _get_display_points(self):
        """Convert TF points to display coordinates for drawing."""
        if not self.points_x:
            return np.empty((0, 2))
        
        display_points = []
        for x, y in zip(self.points_x, self.points_y):
            dx, dy = self._get_display_coords(x, y)
            display_points.append([dx, dy])
        
        # DEBUG: Print first few points
        if len(display_points) > 0:
            print(f"2D View drawing: ({display_points[0][0]:.1f}, {display_points[0][1]:.1f})")
            
        return np.array(display_points)

    # ===== OVERRIDE: Point handling with DEBUG =====
    
    def add_point(self, x, y, color=None):
        """Add a point in DATA coordinates."""
        x = float(np.clip(x, 0.0, 255.0))
        y = float(np.clip(y, 0.0, 1.0))
        print(f"2D View ADD: intensity={x:.1f}, opacity={y:.3f}")
        super().add_point(x, y, color)

    def update_point(self, index, x, y):
        """Update point in DATA coordinates."""
        if 0 <= index < len(self.points_x):
            # ===== ADD DEBUG HERE =====
            print(f"  🔍 2D view update_point RAW: index={index}, x={x:.1f}, y={y:.3f}")
        
            x = float(np.clip(x, 0.0, 255.0))
            y = float(np.clip(y, 0.0, 1.0))
        
            # ===== ADD MORE DEBUG HERE =====
            print(f"  🔍 2D view update_point CLIPPED: x={x:.1f}, y={y:.3f}")
        
            # Lock endpoints
            if index == 0:
                x = 0.0
                print(f"  🔒 Locked endpoint 0 to x=0.0")
            elif index == len(self.points_x) - 1:
                x = 255.0
                print(f"  🔒 Locked endpoint {index} to x=255.0")
        
            print(f"  ✅ 2D view STORING: intensity={x:.1f}, opacity={y:.3f}")
        
            self.points_x[index] = x
            self.points_y[index] = y
            self._sort_points_with_colors()
            self._draw()
            self._notify_app()

    # ===== 2D-SPECIFIC DRAWING =====
    
    def _draw(self):
        """Draw the 2D transfer function."""
        # Update histogram display
        self.im.set_data(self._get_display_data().T)

        # Update TF overlay
        self._draw_tf_overlay()

        # Apply view limits
        self._apply_view_limits()
        self._update_view_limits()

        # Adjust grid based on zoom level
        x_range = self._cached_xlim[1] - self._cached_xlim[0]
        y_range = self._cached_ylim[1] - self._cached_ylim[0]
        if x_range > 10 and y_range > 10:
            self.ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        else:
            self.ax.grid(False)

        # Format ticks
        self._format_ticks(x_range, y_range)

        self.draw()

    def _draw_tf_overlay(self):
        """Draw the TF curve and points overlay."""
        if len(self.points_x) == 0:
            self.tf_line.set_data([], [])
            self.tf_scatter.set_offsets(np.empty((0, 2)))
            return
            
        # Sort points by x for proper line drawing
        sorted_pairs = sorted(zip(self.points_x, self.points_y, self.colors), 
                             key=lambda p: p[0])
        xs, ys, colors = zip(*sorted_pairs)
        
        # Convert to display coordinates for drawing
        display_points = []
        for x, y in zip(xs, ys):
            dx, dy = self._get_display_coords(x, y)
            display_points.append([dx, dy])
        
        display_array = np.array(display_points)
        
        # Draw line connecting points
        self.tf_line.set_data(display_array[:, 0], display_array[:, 1])
        
        # Draw scatter points
        self.tf_scatter.set_offsets(display_array)
        
        # Set colors
        if colors:
            self.tf_scatter.set_facecolor(list(colors))

    def _apply_view_limits(self):
        """Force zoom to stay within 0-255 range."""
        super()._apply_view_limits()
    
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
    
        needs_adjustment = False
        new_xlim = list(xlim)
        new_ylim = list(ylim)
    
        if new_xlim[0] < 0 or new_xlim[1] > 255:
            new_xlim[0] = max(0, new_xlim[0])
            new_xlim[1] = min(255, new_xlim[1])
            needs_adjustment = True
        
        if new_ylim[0] < 0 or new_ylim[1] > 255:
            new_ylim[0] = max(0, new_ylim[0])
            new_ylim[1] = min(255, new_ylim[1])
            needs_adjustment = True
    
        if new_xlim[1] - new_xlim[0] < 5:
            center = (new_xlim[0] + new_xlim[1]) / 2
            new_xlim[0] = center - 2.5
            new_xlim[1] = center + 2.5
            needs_adjustment = True
        
        if new_ylim[1] - new_ylim[0] < 5:
            center = (new_ylim[0] + new_ylim[1]) / 2
            new_ylim[0] = center - 2.5
            new_ylim[1] = center + 2.5
            needs_adjustment = True
    
        if needs_adjustment:
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)

    # ===== CRITICAL: Override notification to update 1D view =====
    def update_labels(self, x_label, y_label):
        """Update the axis labels"""
        self.x_label = x_label
        self.y_label = y_label
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.draw()

    def _notify_app(self):
        """Notify the main app about TF changes."""
        w = self.parent()
        while w is not None and not hasattr(w, 'update_opacity_function_from_2d'):
            w = w.parent()
        if w is not None and hasattr(w, 'update_opacity_function_from_2d'):
            # Pass the SAME data - this should update 1D view
            print(f"2D View notifying app: {len(self.points_x)} points")
            w.update_opacity_function_from_2d(self.points_x, self.points_y, self.colors)