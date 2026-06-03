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
        """Add widget to nD management without duplicating it."""

        if widget in self.widgets:
            return widget

        if not hasattr(widget, "nd_coords"):
            widget.nd_coords = {}

        if not hasattr(widget, "nd_scales"):
            widget.nd_scales = {}

        widget.nd_coords.setdefault("Intensity", widget.center_intensity)
        widget.nd_coords.setdefault("Gradient", widget.center_gradient)

        if hasattr(widget, "intensity_std"):
            widget.nd_scales.setdefault("Intensity", widget.intensity_std)

        if hasattr(widget, "gradient_std"):
            widget.nd_scales.setdefault("Gradient", widget.gradient_std)

        for feat in self.feature_names:
            widget.nd_coords.setdefault(feat, 128)
            widget.nd_scales.setdefault(feat, 30)

        self.widgets.append(widget)
        return widget
    
    def project_to_2d(self, feat_x, feat_y):
        """Project nD widgets to 2D plane - returns DISPLAY coordinates (0-255)."""
        projected = []

        for nd_widget in self.widgets:
            widget_2d = copy(nd_widget)

            widget_2d.nd_ref = nd_widget
            widget_2d.projection = (feat_x, feat_y)

            display_x = nd_widget.nd_coords.get(feat_x, nd_widget.center_intensity)
            display_y = nd_widget.nd_coords.get(feat_y, nd_widget.center_gradient)

            widget_2d.center_intensity = display_x
            widget_2d.center_gradient = display_y

            print(f"   Projected: {feat_x}={display_x:.1f}, {feat_y}={display_y:.1f}")

            # IMPORTANT:
            # Do not overwrite widget sizes here.
            # Size attributes are copied from nd_widget by copy(nd_widget).
            # That preserves:
            # - intensity_std / gradient_std
            # - intensity_width / gradient_height
            # - intensity_radius / gradient_radius

            projected.append(widget_2d)

        return projected
    
    """def update_nd_position(self, widget_2d, new_x, new_y):
        """"""Update nD coordinates when widget moves in 2D (with inverse scaling)""""""
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
                widget_2d.nd_ref.nd_coords[feat_y] = new_y"""
    def update_nd_position(self, widget, x, y, feature_x=None, feature_y=None):
        """
        Store nD widget coordinates in canonical display-space 0..255.
        """

        def to_display(val, feat):
            fv = float(val)

            # Normalized coordinate
            if 0.0 <= fv <= 1.0:
                return 255.0 * fv

            # Display-space coordinate
            if 0.0 <= fv <= 255.0:
                return fv

            # Raw feature-space coordinate
            if hasattr(self, "feature_ranges") and feat in self.feature_ranges:
                vmin, vmax = self.feature_ranges[feat]
                if vmax > vmin:
                    return max(0.0, min(255.0, 255.0 * (fv - vmin) / (vmax - vmin)))

            return fv

        target = getattr(widget, "nd_ref", None)

        if target is True or target is None:
            target = widget

        if not hasattr(target, "nd_coords"):
            target.nd_coords = {}

        if not hasattr(target, "nd_scales"):
            target.nd_scales = {}

        if feature_x is not None and feature_y is not None:
            dx = to_display(x, feature_x)
            dy = to_display(y, feature_y)

            target.nd_coords[feature_x] = dx
            target.nd_coords[feature_y] = dy

            print(f"   Updated nd_coords: {feature_x}={dx:.1f}, {feature_y}={dy:.1f}")
            return

        for feat in self.feature_names:
            target.nd_coords.setdefault(feat, 128)
            target.nd_scales.setdefault(feat, 30)


    def debug_widgets(self):
        print(f"\n🔍 Current widgets in nd_manager ({len(self.widgets)}):")
        for i, w in enumerate(self.widgets):
            print(f"   Widget {i}: {w.nd_coords}")