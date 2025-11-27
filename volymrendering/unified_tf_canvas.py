# unified_tf_canvas.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon
from base_transfer_function import BaseTransferFunction
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from widget_factory import WidgetType

class UnifiedTFCanvas(BaseTransferFunction):
    def __init__(self, tf_type='2d', data=None, gradient_data=None, update_callback=None):
        figsize = (8, 6) if tf_type == '2d' else (8, 4)
        super().__init__(figsize=figsize)
        
        self.tf_type = tf_type
        self.data = data
        self.gradient_data = gradient_data
        self.update_callback = update_callback
        
        # ADD DATA RANGE TRACKING
        self.intensity_range = (0.0, 255.0)  # Default, but will be updated
        self.gradient_range = (0.0, 255.0)   # Default, but will be updated
        
        self.widgets = []
        self.active_widget = None
        self.dragging_widget = False
        
        # Initialize with actual data ranges
        self._update_data_ranges()
        self._setup_canvas()
    
    def _update_data_ranges(self):
        """ALWAYS use 0-255 range - don't calculate from data"""
        self.intensity_range = (0.0, 255.0)
        self.gradient_range = (0.0, 255.0)

    def _setup_canvas(self):
        """Setup canvas - ALWAYS use 0-255 range"""
        self.ax.clear()
    
        if self.tf_type == '2d' and self.data is not None and self.gradient_data is not None:
            # ALWAYS use 0-255 range for histogram
            hist2d, x_edges, y_edges = np.histogram2d(
                self.data, self.gradient_data, 
                bins=256, 
                range=((0, 255), (0, 255))  # ← HARDCODED 0-255!
            )
            self.mesh = self.ax.pcolormesh(
                x_edges, y_edges, np.log1p(hist2d.T),
                cmap='hot', alpha=0.7, shading='auto'
            )
        
            # Set initial view to show full 0-255 range
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 255)
        
        elif self.tf_type == '1d' and self.data is not None:
            # ALWAYS use 0-255 range for 1D
            hist, bins = np.histogram(self.data, bins=256, range=(0, 255))  # ← HARDCODED 0-255!
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            self.ax.plot(bin_centers, hist / hist.max(), color='gray', alpha=0.5)
            self.ax.fill_between(bin_centers, hist / hist.max(), color='lightgray', alpha=0.3)
            self.ax.set_xlim(0, 255)
            self.ax.set_ylim(0, 1)
    
        self.ax.set_xlabel('Intensity')
        self.ax.set_ylabel('Gradient Magnitude' if self.tf_type == '2d' else 'Opacity')
        self.ax.set_title(f'{self.tf_type.upper()} Transfer Function with Widgets')
        self.ax.grid(True, alpha=0.3)


    def canvas_to_data_coords(self, x_canvas, y_canvas):
        """Convert canvas coordinates (0-1) to data coordinates"""
        data_x = x_canvas * (self.intensity_range[1] - self.intensity_range[0]) + self.intensity_range[0]
        data_y = y_canvas * (self.gradient_range[1] - self.gradient_range[0]) + self.gradient_range[0]
        return data_x, data_y

    def data_to_canvas_coords(self, x_data, y_data):
        """Convert data coordinates to canvas coordinates (0-1)"""
        canvas_x = (x_data - self.intensity_range[0]) / (self.intensity_range[1] - self.intensity_range[0])
        canvas_y = (y_data - self.gradient_range[0]) / (self.gradient_range[1] - self.gradient_range[0])
        return canvas_x, canvas_y

    def add_widget(self, widget):
        """Add a widget to the canvas"""
        self.widgets.append(widget)
        self._draw()
        self._notify_app()
        
    def remove_widget(self, widget):
        """Remove a widget from the canvas"""
        if widget in self.widgets:
            self.widgets.remove(widget)
            self._draw()
            self._notify_app()
            return True
        return False
            
    def clear_widgets(self):  # ← THIS WAS MISSING!
        """Remove all widgets"""
        self.widgets.clear()
        self._draw()
        self._notify_app()
        
    
    def sample_for_vtk(self, num_samples=256):
        """Proper 2D sampling that respects widget height"""
        # Get the ACTUAL gradient distribution from your data
        if hasattr(self, 'gradient_normalized'):
            # Use the real gradient values from your volume data
            unique_gradients = np.unique(self.gradient_normalized.astype(int))
            # Sample representative gradients (every 10th to keep it fast)
            data_gradients = unique_gradients[::len(unique_gradients)//20 + 1]
        else:
            # Fallback
            data_gradients = list(range(0, 256, 10))
    
        print(f"📊 Sampling at gradients: {len(data_gradients)} points from data")
    
        # For each intensity, we need to consider which GRADIENTS actually exist there
        intensity_opacity = np.zeros(256)
        intensity_color = np.ones((256, 3))
    
        for widget in self.widgets:
            # For each ACTUAL gradient in our data...
            for data_gradient in data_gradients:
                # Check if this widget affects THIS SPECIFIC gradient
                widget_opacity = widget.calculate_opacity(widget.center_intensity, data_gradient)
            
                if widget_opacity > 0:
                    # This widget affects data at this gradient!
                    # Now find which intensities get affected at this gradient
                    if widget.widget_type == WidgetType.RECTANGULAR:
                        intensity_range = (
                            max(0, int(widget.center_intensity - widget.intensity_width/2)),
                            min(255, int(widget.center_intensity + widget.intensity_width/2))
                        )
                    elif widget.widget_type == WidgetType.GAUSSIAN:
                        intensity_range = (
                            max(0, int(widget.center_intensity - 3 * widget.intensity_std)),
                            min(255, int(widget.center_intensity + 3 * widget.intensity_std))
                        )
                    else:
                        intensity_range = (max(0, widget.center_intensity-25), min(255, widget.center_intensity+25))
                
                    # Apply the widget to intensities in its range
                    for intensity in range(intensity_range[0], intensity_range[1] + 1):
                        opacity = widget.calculate_opacity(intensity, data_gradient)
                        if opacity > intensity_opacity[intensity]:
                            intensity_opacity[intensity] = opacity
                            intensity_color[intensity] = widget.color
    
        samples = [(i, intensity_opacity[i], tuple(intensity_color[i])) for i in range(256)]
        return samples
    
    def _draw(self):
        """Draw the canvas with widgets"""
        self.ax.clear()
        self._setup_canvas()
        
        # Draw widgets
        for i, widget in enumerate(self.widgets):
            self._draw_widget(widget, i == self.active_widget)
            
        self.draw()
        
    def _draw_widget(self, widget, is_active=False):
        """Draw a single widget"""
        color = 'red' if is_active else widget.color
        linewidth = 3 if is_active else 2
    
        if widget.widget_type == WidgetType.GAUSSIAN:
            self._draw_gaussian_widget(widget, color, linewidth)
        elif widget.widget_type == WidgetType.TRIANGULAR:
            self._draw_triangular_widget(widget, color, linewidth)
        elif widget.widget_type == WidgetType.RECTANGULAR:
            self._draw_rectangular_widget(widget, color, linewidth)
        elif widget.widget_type == WidgetType.ELLIPSOID:
            self._draw_ellipsoid_widget(widget, color, linewidth)
        elif widget.widget_type == WidgetType.DIAMOND:
            self._draw_diamond_widget(widget, color, linewidth)
    
        # Draw center point for all widgets
        self.ax.plot(widget.center_intensity, widget.center_gradient, 'o', 
                    color=color, markersize=8, markeredgecolor='black')
        
    def _draw_gaussian_widget(self, widget, color, linewidth):
        """Draw Gaussian widget as contour"""
        ellipse = Ellipse(
            (widget.center_intensity, widget.center_gradient),
            width=widget.intensity_std * 2,
            height=widget.gradient_std * 2,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8
        )
        self.ax.add_patch(ellipse)

    def _draw_triangular_widget(self, widget, color, linewidth):
        """Draw Triangular widget as actual triangle"""
        from matplotlib.patches import Polygon
    
        if widget.direction == 'up':
            points = [
                (widget.center_intensity, widget.center_gradient),  # Bottom center
                (widget.center_intensity - widget.intensity_width/2, widget.center_gradient + widget.gradient_height),  # Top left
                (widget.center_intensity + widget.intensity_width/2, widget.center_gradient + widget.gradient_height)   # Top right
            ]
        elif widget.direction == 'down':
            points = [
                (widget.center_intensity, widget.center_gradient),  # Top center  
                (widget.center_intensity - widget.intensity_width/2, widget.center_gradient - widget.gradient_height),  # Bottom left
                (widget.center_intensity + widget.intensity_width/2, widget.center_gradient - widget.gradient_height)   # Bottom right
            ]
        else:  # symmetric
            points = [
                (widget.center_intensity, widget.center_gradient + widget.gradient_height/2),  # Top
                (widget.center_intensity - widget.intensity_width/2, widget.center_gradient - widget.gradient_height/2),  # Bottom left
                (widget.center_intensity + widget.intensity_width/2, widget.center_gradient - widget.gradient_height/2)   # Bottom right
            ]
    
        polygon = Polygon(
            points, 
            fill=False, 
            edgecolor=color, 
            linewidth=linewidth, 
            alpha=0.8
        )
        self.ax.add_patch(polygon)
        
    def _draw_rectangular_widget(self, widget, color, linewidth):
        """Draw Rectangular widget as rectangle"""
        from matplotlib.patches import Rectangle
    
        rect = Rectangle(
            (widget.center_intensity - widget.intensity_width/2.0, 
             widget.center_gradient - widget.gradient_height/2.0),
            widget.intensity_width, 
            widget.gradient_height,
            fill=False, 
            edgecolor=color, 
            linewidth=linewidth, 
            alpha=0.8
        )
        self.ax.add_patch(rect)

    def _draw_ellipsoid_widget(self, widget, color, linewidth):
        """Draw Ellipsoid widget as ellipse"""
        from matplotlib.patches import Ellipse
    
        ellipse = Ellipse(
            (widget.center_intensity, widget.center_gradient),
            width=widget.intensity_radius * 2.0,
            height=widget.gradient_radius * 2.0,
            fill=False, 
            edgecolor=color, 
            linewidth=linewidth, 
            alpha=0.8
        )
        self.ax.add_patch(ellipse)

    def _draw_diamond_widget(self, widget, color, linewidth):
        """Draw Diamond widget as diamond"""
        from matplotlib.patches import Polygon
    
        points = [
            (widget.center_intensity, widget.center_gradient - widget.gradient_height/2),  # Bottom
            (widget.center_intensity + widget.intensity_width/2, widget.center_gradient),  # Right
            (widget.center_intensity, widget.center_gradient + widget.gradient_height/2),  # Top  
            (widget.center_intensity - widget.intensity_width/2, widget.center_gradient)   # Left
        ]
        polygon = Polygon(
            points, 
            fill=False, 
            edgecolor=color, 
            linewidth=linewidth, 
            alpha=0.8
        )
        self.ax.add_patch(polygon)
    
    def on_press(self, event):
        """Handle mouse press for widget interaction - FIXED COORDINATES"""
        if event.inaxes != self.ax:
            return
    
        # Convert click coordinates to data space
        click_x_data = event.xdata
        click_y_data = event.ydata
    
        # Check for Shift+click for color change
        if getattr(event, 'button', None) == 1:  # Left click
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
        
            if mods & Qt.ShiftModifier:
                for i, widget in enumerate(self.widgets):
                    # Use data coordinates for distance calculation
                    distance = np.sqrt((click_x_data - widget.center_intensity)**2 + 
                                     (click_y_data - widget.center_gradient)**2)
                    if distance < (self.intensity_range[1] - self.intensity_range[0]) * 0.05:  # 5% of range
                        qcolor = QtWidgets.QColorDialog.getColor()
                        if qcolor.isValid():
                            widget.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                            self._draw()
                            self._notify_app()
                        return

        # Existing widget dragging code - using data coordinates
        for i, widget in enumerate(self.widgets):
            distance = np.sqrt((click_x_data - widget.center_intensity)**2 + 
                             (click_y_data - widget.center_gradient)**2)
            threshold = (self.intensity_range[1] - self.intensity_range[0]) * 0.05  # 5% of range
            if distance < threshold:
                self.active_widget = i
                self.dragging_widget = True
                self._draw()
                return
        
        # If no widget clicked, use base class behavior for point addition
        super().on_press(event)
    
    def on_motion(self, event):
        """Handle mouse motion for widget dragging - FIXED COORDINATES"""
        if self.dragging_widget and event.inaxes == self.ax and self.active_widget is not None:
            widget = self.widgets[self.active_widget]
            # Use data coordinates directly
            widget.center_intensity = event.xdata
            widget.center_gradient = event.ydata
            self._draw()
            self._notify_app()
        else:
            super().on_motion(event)
            
    def on_release(self, event):
        """Handle mouse release"""
        self.dragging_widget = False
        super().on_release(event)
        
    def _notify_app(self):
        """Notify application about TF changes"""
        if self.update_callback:
            self.update_callback()

    def set_tf_type(self, tf_type):
        """Switch between 1D and 2D mode - ADD THIS METHOD"""
        self.tf_type = tf_type
        # Update figure size based on mode
        if tf_type == '1d':
            self.fig.set_size_inches(8, 4)
        else:
            self.fig.set_size_inches(8, 6)
        self._setup_canvas()
        self._draw()

    def calculate_combined_opacity(self, intensity, gradient):
        """Combine opacity from all widgets using blend modes"""
        if not self.widgets:
            return 0.0
            
        final_opacity = 0.0
        
        for widget in self.widgets:
            widget_opacity = widget.calculate_opacity(intensity, gradient)
            
            if widget.blend_mode == 'add':
                final_opacity += widget_opacity
            elif widget.blend_mode == 'multiply':
                final_opacity = final_opacity * (1 - widget_opacity) + widget_opacity
            else:  # 'max' - default
                final_opacity = max(final_opacity, widget_opacity)
                
        return min(1.0, final_opacity)  # Clamp to [0,1]

    #ND
    def set_feature_pair(self, feature_x, feature_y, feature_data_x, feature_data_y):
        """Dynamically switch what features are displayed"""
        self.current_features = (feature_x, feature_y)
    
        # Update the data attributes that your existing code uses
        self.data = feature_data_x
        self.gradient_data = feature_data_y
    
        # Update the canvas
        self._setup_canvas()  # This will use the new data
        self._draw()  # Redraw with existing widgets
    
        print(f"🔄 TF Canvas updated: {feature_x} vs {feature_y}")

    # In UnifiedTFCanvas class
    def reset_view(self):
        """Reset the view to show full 0-255 range"""
        if hasattr(self, 'ax'):
            self.ax.set_xlim(0, 255)
            if self.tf_type == '2d':
                self.ax.set_ylim(0, 255)
            else:
                self.ax.set_ylim(0, 1)
            self.draw()