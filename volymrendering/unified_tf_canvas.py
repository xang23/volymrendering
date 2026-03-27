import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon
from base_transfer_function import BaseTransferFunction
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets
from widget_factory import WidgetType

class UnifiedTFCanvas(BaseTransferFunction):
    def __init__(self, tf_type='2d', data=None, gradient_data=None, update_callback=None,
                 x_label="Intensity", y_label="Gradient Magnitude"):
        figsize = (8, 6) if tf_type == '2d' else (8, 4)
        super().__init__(figsize=figsize)
        
        self.tf_type = tf_type
        self.data = data
        self.gradient_data = gradient_data
        self.update_callback = update_callback
        self.x_label = x_label
        self.y_label = y_label
        
        # ADD DATA RANGE TRACKING
        self.intensity_range = (0.0, 255.0)  # Default, but will be updated
        self.gradient_range = (0.0, 255.0)   # Default, but will be updated
        
        self.widgets = []
        self.active_widget = None
        self.dragging_widget = False
        
        # Initialize with actual data ranges
        self._update_data_ranges()
        self._setup_canvas()

    def update_axis_labels(self, x_label, y_label):
        """Update the axis labels on the 2D canvas"""
        self.x_label = x_label
        self.y_label = y_label
        
        # Update the current canvas axes
        if hasattr(self, 'ax') and self.ax:
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)
            self.draw()
        
        # If we have a 2D TF object, also update its labels
        if hasattr(self, 'tf_2d') and self.tf_2d:
            if hasattr(self.tf_2d, 'update_labels'):
                self.tf_2d.update_labels(x_label, y_label)
    
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
    
        # Use the stored labels
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label if self.tf_type == '2d' else 'Opacity')
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
            widget_count = len(self.widgets)
            self.widgets.clear()
            self._draw()
            self._notify_app()
            print(f"✅ Cleared {widget_count} widgets safely")
        except Exception as e:
            print(f"❌ Error clearing widgets: {e}")
            import traceback
            traceback.print_exc()
        
    
    def sample_for_vtk(self):
        """Main entry point for VTK sampling"""
        print(f"🔍 sample_for_vtk called, tf_type={self.tf_type}, widgets={len(self.widgets)}")
    
        if self.tf_type == '1d':
            result = self._sample_1d_for_vtk()
            print(f"📊 1D mode: returning {len(result)} samples")
            return result
        else:  # 2D mode
            result = self._sample_2d_for_vtk_dual_functions()
            print(f"📊 2D mode: returning {len(result)} samples")
        
            if hasattr(self, '_cached_gradient_opacity'):
                grad_op = self._cached_gradient_opacity
                if isinstance(grad_op, np.ndarray):
                    non_zero = np.sum(grad_op > 0.01)
                    print(f"📊 Stored gradient opacity has {non_zero} non-zero values")
        
            return result

    def _sample_1d_for_vtk(self):
        """Sample 1D transfer function for VTK"""
        intensity_opacity = np.zeros(256)
        intensity_color = np.ones((256, 3))
    
        for widget in self.widgets:
            intensity_range = self._get_widget_intensity_range(widget)
            for intensity in intensity_range:
                opacity = widget.calculate_opacity(intensity, 0)
                if opacity > intensity_opacity[intensity]:
                    intensity_opacity[intensity] = opacity
                    intensity_color[intensity] = widget.color
    
        return self._create_vtk_samples(intensity_opacity, intensity_color)

    def _sample_2d_for_vtk_dual_functions(self):
        """Create dual 1D functions for VTK (scalar + gradient opacity)"""
        scalar_influence = np.zeros(256)
        gradient_influence = np.zeros(256)  
        color_influence = np.ones((256, 3))
    
        print(f"🎯 Creating 2D TF with {len(self.widgets)} widgets")
    
        for widget_idx, widget in enumerate(self.widgets):
            print(f"  Processing {widget.widget_type.value} widget...")
        
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
        
            for intensity in range(intensity_min, intensity_max + 1):
                max_opacity = 0
                for gradient in range(gradient_min, gradient_max + 1, 4):
                    opacity = widget.calculate_opacity(intensity, gradient)
                    max_opacity = max(max_opacity, opacity)
            
                if max_opacity > 0:
                    if widget.blend_mode == 'max':
                        if max_opacity > scalar_influence[intensity]:
                            scalar_influence[intensity] = max_opacity
                            color_influence[intensity] = widget.color
                    elif widget.blend_mode == 'add':
                        scalar_influence[intensity] += max_opacity
                        if max_opacity > 0:
                            weight = max_opacity / (scalar_influence[intensity] + 1e-6)
                            color_influence[intensity] = (
                                weight * np.array(widget.color) + 
                                (1 - weight) * color_influence[intensity]
                            )
        
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
    
        scalar_influence = np.clip(scalar_influence, 0, 1)
        gradient_influence = np.clip(gradient_influence, 0, 1)
    
        return self._create_dual_function_samples(scalar_influence, gradient_influence, color_influence)

    def _create_dual_function_samples(self, scalar_opacity, gradient_opacity, colors):
        """Convert dual functions to VTK samples format"""
        samples = []
    
        for intensity in range(0, 256, 4):
            opacity = scalar_opacity[intensity]
            if opacity > 0.01:
                samples.append((intensity, opacity, tuple(colors[intensity])))
    
        print(f"📊 Generated {len(samples)} samples for VTK")
        self._cached_gradient_opacity = gradient_opacity
        print(f"📊 Stored gradient opacity array with shape: {gradient_opacity.shape}")
    
        return samples
    
    def _draw(self):
        """Draw the canvas with widgets"""
        self.ax.clear()
        self._setup_canvas()
        
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
    
        self.ax.plot(widget.center_intensity, widget.center_gradient, 'o', 
                    color=color, markersize=8, markeredgecolor='black')
        
    def _draw_gaussian_widget(self, widget, color, linewidth):
        ellipse = Ellipse(
            (widget.center_intensity, widget.center_gradient),
            width=widget.intensity_std * 2,
            height=widget.gradient_std * 2,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8
        )
        self.ax.add_patch(ellipse)

    def _draw_triangular_widget(self, widget, color, linewidth):
        from matplotlib.patches import Polygon
    
        if widget.direction == 'up':
            points = [
                (widget.center_intensity, widget.center_gradient),
                (widget.center_intensity - widget.intensity_width/2, widget.center_gradient + widget.gradient_height),
                (widget.center_intensity + widget.intensity_width/2, widget.center_gradient + widget.gradient_height)
            ]
        elif widget.direction == 'down':
            points = [
                (widget.center_intensity, widget.center_gradient),
                (widget.center_intensity - widget.intensity_width/2, widget.center_gradient - widget.gradient_height),
                (widget.center_intensity + widget.intensity_width/2, widget.center_gradient - widget.gradient_height)
            ]
        else:
            points = [
                (widget.center_intensity, widget.center_gradient + widget.gradient_height/2),
                (widget.center_intensity - widget.intensity_width/2, widget.center_gradient - widget.gradient_height/2),
                (widget.center_intensity + widget.intensity_width/2, widget.center_gradient - widget.gradient_height/2)
            ]
    
        polygon = Polygon(points, fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8)
        self.ax.add_patch(polygon)
        
    def _draw_rectangular_widget(self, widget, color, linewidth):
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (widget.center_intensity - widget.intensity_width/2.0, 
             widget.center_gradient - widget.gradient_height/2.0),
            widget.intensity_width, widget.gradient_height,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8
        )
        self.ax.add_patch(rect)

    def _draw_ellipsoid_widget(self, widget, color, linewidth):
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(
            (widget.center_intensity, widget.center_gradient),
            width=widget.intensity_radius * 2.0,
            height=widget.gradient_radius * 2.0,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8
        )
        self.ax.add_patch(ellipse)

    def _draw_diamond_widget(self, widget, color, linewidth):
        from matplotlib.patches import Polygon
        points = [
            (widget.center_intensity, widget.center_gradient - widget.gradient_height/2),
            (widget.center_intensity + widget.intensity_width/2, widget.center_gradient),
            (widget.center_intensity, widget.center_gradient + widget.gradient_height/2),
            (widget.center_intensity - widget.intensity_width/2, widget.center_gradient)
        ]
        polygon = Polygon(points, fill=False, edgecolor=color, linewidth=linewidth, alpha=0.8)
        self.ax.add_patch(polygon)
    
    def on_press(self, event):
        """Handle mouse press for widget interaction"""
        if event.inaxes != self.ax:
            return

        click_x_data = event.xdata
        click_y_data = event.ydata

        if getattr(event, 'button', None) == 1:
            try:
                mods = event.guiEvent.modifiers()
            except Exception:
                mods = 0
    
            if mods & Qt.ShiftModifier:
                # Color picker mode - find the closest widget
                closest_idx = None
                closest_dist = float('inf')
                for i, widget in enumerate(self.widgets):
                    distance = np.sqrt((click_x_data - widget.center_intensity)**2 + 
                                     (click_y_data - widget.center_gradient)**2)
                    if distance < 15 and distance < closest_dist:
                        closest_idx = i
                        closest_dist = distance
            
                if closest_idx is not None:
                    widget = self.widgets[closest_idx]
                    qcolor = QtWidgets.QColorDialog.getColor()
                    if qcolor.isValid():
                        widget.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                        self._draw()
                        self._notify_app()
                    return

        # Find the closest widget for selection/dragging
        closest_widget_idx = None
        closest_distance = float('inf')
        threshold = 15  # 15 pixels in data coordinates
    
        for i, widget in enumerate(self.widgets):
            distance = np.sqrt((click_x_data - widget.center_intensity)**2 + 
                             (click_y_data - widget.center_gradient)**2)
            if distance < threshold and distance < closest_distance:
                closest_widget_idx = i
                closest_distance = distance
    
        if closest_widget_idx is not None:
            self.active_widget = closest_widget_idx
            self.dragging_widget = True
            self._draw()
            print(f"✅ Selected widget {closest_widget_idx} at distance {closest_distance:.1f}")
            return
    
        super().on_press(event)

    def on_motion(self, event):
        """Handle mouse motion for widget dragging"""
        if self.dragging_widget and event.inaxes == self.ax and self.active_widget is not None:
            # Store the active widget reference before potential list modifications
            active_widget = self.widgets[self.active_widget]
        
            # Update position
            new_x = event.xdata
            new_y = event.ydata
        
            # Clamp to canvas bounds
            new_x = max(0, min(255, new_x))
            new_y = max(0, min(255, new_y))
        
            active_widget.center_intensity = new_x
            active_widget.center_gradient = new_y
    
            # Notify nD manager about the move
            if hasattr(active_widget, 'nd_ref') and hasattr(self, 'projection_x'):
                if hasattr(self, 'nd_update_callback'):
                    self.nd_update_callback(
                        active_widget, 
                        self.projection_x, self.projection_y,
                        new_x, new_y
                    )
        
            self._draw()
            self._notify_app()
        else:
            super().on_motion(event)
    
    def on_motion(self, event):
        """Handle mouse motion for widget dragging"""
        if self.dragging_widget and event.inaxes == self.ax and self.active_widget is not None:
            widget = self.widgets[self.active_widget]
            widget.center_intensity = event.xdata
            widget.center_gradient = event.ydata
        
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
            
    def on_release(self, event):
        """Handle mouse release"""
        self.dragging_widget = False
        super().on_release(event)
        
    def _notify_app(self):
        """Notify application about TF changes"""
        if self.update_callback:
            self.update_callback()

    def set_tf_type(self, tf_type):
        """Switch between 1D and 2D mode"""
        self.tf_type = tf_type
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
            else:
                final_opacity = max(final_opacity, widget_opacity)
                
        return min(1.0, final_opacity)

    def set_feature_pair(self, feature_x, feature_y, feature_data_x, feature_data_y):
        """Dynamically switch what features are displayed"""
        self.current_features = (feature_x, feature_y)
        self.data = feature_data_x
        self.gradient_data = feature_data_y
        self._setup_canvas()
        self._draw()
        print(f"🔄 TF Canvas updated: {feature_x} vs {feature_y}")

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
            center = int(widget.center_intensity)
            return range(max(0, center-10), min(255, center+10) + 1)

    def set_projection_features(self, feat_x, feat_y):
        """Set which features this canvas is showing"""
        self.projection_x = feat_x
        self.projection_y = feat_y

    def set_nd_callback(self, callback):
        """Set callback for nD updates"""
        self.nd_update_callback = callback

    def on_scroll(self, event):
        """Handle mouse wheel for zooming"""
        if event.inaxes != self.ax:
            return
    
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
    
        scale = 0.9 if event.button == 'up' else 1.1
    
        new_x_min = event.xdata - (event.xdata - x_min) * scale
        new_x_max = event.xdata + (x_max - event.xdata) * scale
        new_y_min = event.ydata - (event.ydata - y_min) * scale
        new_y_max = event.ydata + (y_max - event.ydata) * scale
    
        self.ax.set_xlim(new_x_min, new_x_max)
        self.ax.set_ylim(new_y_min, new_y_max)
        self.draw()