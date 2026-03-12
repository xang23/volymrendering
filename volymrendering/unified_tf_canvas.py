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
            
    def clear_widgets(self):
        """Remove all widgets safely without crashes"""
        try:
            # Store count for debugging
            widget_count = len(self.widgets)
        
            # Clear the list
            self.widgets.clear()
        
            # Redraw canvas
            self._draw()
        
            # Notify application
            self._notify_app()
        
            print(f"✅ Cleared {widget_count} widgets safely")
        
        except Exception as e:
            print(f"❌ Error clearing widgets: {e}")
            import traceback
            traceback.print_exc()
        
    
    def sample_for_vtk(self):
        """Main entry point for VTK sampling - properly handles both 1D and 2D"""
        print(f"🔍 sample_for_vtk called, tf_type={self.tf_type}, widgets={len(self.widgets)}")
    
        if self.tf_type == '1d':
            result = self._sample_1d_for_vtk()
            print(f"📊 1D mode: returning {len(result)} samples")
            return result
        else:  # 2D mode
            result = self._sample_2d_for_vtk_dual_functions()
            print(f"📊 2D mode: returning {len(result)} samples")
        
            # Debug: Check if gradient was stored
            if hasattr(self, '_cached_gradient_opacity'):
                grad_op = self._cached_gradient_opacity
                if isinstance(grad_op, np.ndarray):
                    non_zero = np.sum(grad_op > 0.01)
                    print(f"📊 Stored gradient opacity has {non_zero} non-zero values")
        
            return result

    def _sample_1d_for_vtk(self):
        """Sample 1D transfer function for VTK (KEEP your existing 1D logic)"""
        intensity_opacity = np.zeros(256)
        intensity_color = np.ones((256, 3))
    
        for widget in self.widgets:
            intensity_range = self._get_widget_intensity_range(widget)
            for intensity in intensity_range:
                # Use gradient=0 for 1D mode
                opacity = widget.calculate_opacity(intensity, 0)
                if opacity > intensity_opacity[intensity]:
                    intensity_opacity[intensity] = opacity
                    intensity_color[intensity] = widget.color
    
        return self._create_vtk_samples(intensity_opacity, intensity_color)

    def _sample_2d_for_vtk_dual_functions(self):
        """NEW: Create dual 1D functions for VTK (scalar + gradient opacity)"""
        # Create separate arrays for scalar and gradient influence
        scalar_influence = np.zeros(256)  # Intensity -> opacity
        gradient_influence = np.zeros(256)  # Gradient -> opacity  
        color_influence = np.ones((256, 3))  # Intensity -> color
    
        print(f"🎯 Creating 2D TF with {len(self.widgets)} widgets")
    
        # Sample each widget's influence in 2D space
        for widget_idx, widget in enumerate(self.widgets):
            print(f"  Processing {widget.widget_type.value} widget...")
        
            # Determine widget's effective ranges
            if hasattr(widget, 'intensity_width'):
                intensity_min = max(0, int(widget.center_intensity - widget.intensity_width/2))
                intensity_max = min(255, int(widget.center_intensity + widget.intensity_width/2))
            else:
                intensity_min = 0
                intensity_max = 255
            
            if hasattr(widget, 'gradient_height'):
                gradient_min = max(0, int(widget.center_gradient - widget.gradient_height/2))
                gradient_max = min(255, int(widget.center_gradient + widget.gradient_height/2))
            else:
                gradient_min = 0
                gradient_max = 255
        
            # For each intensity in widget's range, find max opacity across gradients
            for intensity in range(intensity_min, intensity_max + 1):
                max_opacity = 0
                # Sample gradients within widget's range (step by 4 for speed)
                for gradient in range(gradient_min, gradient_max + 1, 4):
                    opacity = widget.calculate_opacity(intensity, gradient)
                    max_opacity = max(max_opacity, opacity)
            
                if max_opacity > 0:
                    # Apply blending
                    if widget.blend_mode == 'max':
                        if max_opacity > scalar_influence[intensity]:
                            scalar_influence[intensity] = max_opacity
                            color_influence[intensity] = widget.color
                    elif widget.blend_mode == 'add':
                        scalar_influence[intensity] += max_opacity
                        # Blend colors based on contribution
                        if max_opacity > 0:
                            weight = max_opacity / (scalar_influence[intensity] + 1e-6)
                            color_influence[intensity] = (
                                weight * np.array(widget.color) + 
                                (1 - weight) * color_influence[intensity]
                            )
        
            # For each gradient in widget's range, find max opacity across intensities
            for gradient in range(gradient_min, gradient_max + 1):
                max_opacity = 0
                for intensity in range(intensity_min, intensity_max + 1, 4):
                    opacity = widget.calculate_opacity(intensity, gradient)
                    max_opacity = max(max_opacity, opacity)
            
                if max_opacity > 0:
                    if widget.blend_mode == 'max':
                        gradient_influence[gradient] = max(
                            gradient_influence[gradient], max_opacity
                        )
                    elif widget.blend_mode == 'add':
                        gradient_influence[gradient] += max_opacity
    
        # Clamp values
        scalar_influence = np.clip(scalar_influence, 0, 1)
        gradient_influence = np.clip(gradient_influence, 0, 1)
    
        # Create VTK-compatible samples (format compatible with existing code)
        return self._create_dual_function_samples(scalar_influence, gradient_influence, color_influence)

    def _create_dual_function_samples(self, scalar_opacity, gradient_opacity, colors):
        """Convert dual functions to VTK samples format"""
        # Create samples in the SAME format as before for compatibility
        samples = []
    
        # Sample every 4th intensity for efficiency
        for intensity in range(0, 256, 4):
            opacity = scalar_opacity[intensity]
            if opacity > 0.01:  # Only include significant points
                samples.append((intensity, opacity, tuple(colors[intensity])))
    
        print(f"📊 Generated {len(samples)} samples for VTK")
    
        # CRITICAL FIX: ALWAYS store gradient opacity (not conditional!)
        self._cached_gradient_opacity = gradient_opacity
        print(f"📊 Stored gradient opacity array with shape: {gradient_opacity.shape}")
    
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

    def _get_widget_intensity_range(self, widget):
        """Get the intensity range affected by a widget"""
        if hasattr(widget, 'intensity_width'):
            min_int = max(0, int(widget.center_intensity - widget.intensity_width/2))
            max_int = min(255, int(widget.center_intensity + widget.intensity_width/2))
            return range(min_int, max_int + 1)
        else:
            # Default to widget center ±10
            center = int(widget.center_intensity)
            return range(max(0, center-10), min(255, center+10) + 1)

    def set_projection_features(self, feat_x, feat_y):
        """Set which features this canvas is showing"""
        self.projection_x = feat_x
        self.projection_y = feat_y

    def set_nd_callback(self, callback):
        """Set callback for nD updates"""
        self.nd_update_callback = callback

    def on_motion(self, event):
        """Modified to update nD coordinates"""
        if self.dragging_widget and event.inaxes == self.ax and self.active_widget is not None:
            widget = self.widgets[self.active_widget]
        
            # Update 2D position
            widget.center_intensity = event.xdata
            widget.center_gradient = event.ydata
        
            # NEW: Update nD coordinates if this is a projection
            if hasattr(widget, 'nd_ref') and hasattr(self, 'projection_x'):
                if hasattr(self, 'nd_update_callback'):
                    self.nd_update_callback(
                        widget, 
                        self.projection_x, self.projection_y,
                        event.xdata, event.ydata
                    )
        
            self._draw()
            self._notify_app()
        else:
            super().on_motion(event)

    def on_scroll(self, event):
        """Handle mouse wheel for zooming"""
        if event.inaxes != self.ax:
            return
    
        # Get current limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
    
        # Zoom factor (scroll up = zoom in, scroll down = zoom out)
        scale = 0.9 if event.button == 'up' else 1.1
    
        # Calculate new limits centered on mouse position
        x_range = (x_max - x_min) * scale
        y_range = (y_max - y_min) * scale
    
        new_x_min = event.xdata - (event.xdata - x_min) * scale
        new_x_max = event.xdata + (x_max - event.xdata) * scale
        new_y_min = event.ydata - (event.ydata - y_min) * scale
        new_y_max = event.ydata + (y_max - event.ydata) * scale
    
        # Apply new limits
        self.ax.set_xlim(new_x_min, new_x_max)
        self.ax.set_ylim(new_y_min, new_y_max)
    
        self.draw()