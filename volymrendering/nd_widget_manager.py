# nd_widget_manager.py
from copy import copy
import numpy as np

class NDWidgetManager:
    """Manages nD widget positions with proper scaling"""
    
    def __init__(self):
        self.widgets = []  # Original nD widgets
        self.feature_names = ['Intensity', 'Gradient']  # Default features
        self.feature_ranges = {}  # Store min/max for each feature
        
    def update_features(self, feature_names, feature_data=None):
        """Update available features and their ranges from loaded data"""
        # Add any new features
        for feat in feature_names:
            if feat not in self.feature_names:
                self.add_feature(feat)
        
        # Store ranges if data provided
        if feature_data:
            for name, data in feature_data.items():
                if data is not None and len(data) > 0:
                    self.feature_ranges[name] = (float(np.min(data)), 
                                                float(np.max(data)))
        
        self.feature_names = feature_names
    
    def add_feature(self, name):
        """Add a new feature dimension"""
        if name not in self.feature_names:
            self.feature_names.append(name)
            # Initialize existing widgets for this feature
            for widget in self.widgets:
                if not hasattr(widget, 'nd_coords'):
                    widget.nd_coords = {}
                    widget.nd_scales = {}
                # Default values for new feature (middle of 0-255 range)
                widget.nd_coords[name] = 128
                widget.nd_scales[name] = 30
    
    def add_widget(self, widget):
        """Add widget to nD management"""
        # Initialize nD coordinates from 2D position (in display space)
        widget.nd_coords = {
            'Intensity': widget.center_intensity,
            'Gradient': widget.center_gradient
        }
        # Initialize scales
        widget.nd_scales = {}
        if hasattr(widget, 'intensity_std'):
            widget.nd_scales['Intensity'] = widget.intensity_std
        if hasattr(widget, 'gradient_std'):
            widget.nd_scales['Gradient'] = widget.gradient_std
        
        # Set defaults for any other features
        for feat in self.feature_names:
            if feat not in widget.nd_coords:
                widget.nd_coords[feat] = 128
            if feat not in widget.nd_scales:
                widget.nd_scales[feat] = 30
        
        self.widgets.append(widget)
        return widget
    
    def project_to_2d(self, feat_x, feat_y):
        """Project nD widgets to 2D plane - returns RAW coordinates"""
        projected = []

        for nd_widget in self.widgets:
            widget_2d = copy(nd_widget)
        
            # ===== CRITICAL: Store reference to original nD widget =====
            widget_2d.nd_ref = nd_widget  # ← ADD THIS!
            widget_2d.projection = (feat_x, feat_y)  # ← ADD THIS!
            # =========================================================
    
            # Get display coordinates (0-255) from nD storage
            display_x = nd_widget.nd_coords.get(feat_x, 128)
            display_y = nd_widget.nd_coords.get(feat_y, 128)
    
            # Convert to RAW data coordinates using stored ranges
            if feat_x in self.feature_ranges:
                x_min, x_max = self.feature_ranges[feat_x]
                # Convert 0-255 → raw data
                raw_x = x_min + (display_x / 255.0) * (x_max - x_min)
                widget_2d.center_intensity = raw_x
        
                # Scale size too
                if hasattr(widget_2d, 'intensity_std'):
                    std_display = nd_widget.nd_scales.get(feat_x, 30)
                    widget_2d.intensity_std = (std_display / 255.0) * (x_max - x_min)
    
            if feat_y in self.feature_ranges:
                y_min, y_max = self.feature_ranges[feat_y]
                raw_y = y_min + (display_y / 255.0) * (y_max - y_min)
                widget_2d.center_gradient = raw_y
        
                if hasattr(widget_2d, 'gradient_std'):
                    std_display = nd_widget.nd_scales.get(feat_y, 30)
                    widget_2d.gradient_std = (std_display / 255.0) * (y_max - y_min)
    
            projected.append(widget_2d)

        return projected
    
    def update_nd_position(self, widget_2d, new_x, new_y):
        """Update nD coordinates when widget moves in 2D (with inverse scaling)"""
        if hasattr(widget_2d, 'nd_ref'):
            feat_x, feat_y = widget_2d.projection
            
            # Check if we need to inverse scale
            if feat_x in self.feature_ranges and feat_y in self.feature_ranges:
                x_min, x_max = self.feature_ranges[feat_x]
                y_min, y_max = self.feature_ranges[feat_y]
                
                # Inverse scale back to display space (0-255)
                display_x = 255.0 * (new_x - x_min) / (x_max - x_min)
                display_y = 255.0 * (new_y - y_min) / (y_max - y_min)
                
                # Clamp to 0-255
                display_x = max(0, min(255, display_x))
                display_y = max(0, min(255, display_y))
                
                widget_2d.nd_ref.nd_coords[feat_x] = display_x
                widget_2d.nd_ref.nd_coords[feat_y] = display_y
            else:
                # No scaling needed
                widget_2d.nd_ref.nd_coords[feat_x] = new_x
                widget_2d.nd_ref.nd_coords[feat_y] = new_y