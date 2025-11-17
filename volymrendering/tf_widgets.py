import numpy as np
from enum import Enum

class WidgetType(Enum):
    TRIANGULAR = "triangular"
    RECTANGULAR = "rectangular" 
    GAUSSIAN = "gaussian"
    ELLIPSOID = "ellipsoid"

class TFWidget:
    # Base widget class with common functionality
    def __init__(self, widget_type, center_intensity=128, center_gradient=128, opacity=1.0, color=(1.0, 1.0, 1.0)):
        self.widget_type = widget_type
        self.center_intensity = center_intensity
        self.center_gradient = center_gradient
        self.opacity = opacity
        self.color = color
        self.selected = False
        
    def calculate_opacity(self, intensity, gradient):
        raise NotImplementedError
        
    def sample_for_vtk(self, num_samples=50):
        raise NotImplementedError
        
    def get_parameters(self):
        """Return parameters for UI controls"""
        raise NotImplementedError
        
    def set_parameter(self, name, value):
        """Update a parameter value"""
        raise NotImplementedError

class GaussianWidget(TFWidget):
    def __init__(self, center_intensity=128, center_gradient=128, intensity_std=30, gradient_std=30, opacity=1.0, color=(1.0, 1.0, 1.0)):
        super().__init__(WidgetType.GAUSSIAN, center_intensity, center_gradient, opacity, color)
        self.intensity_std = intensity_std
        self.gradient_std = gradient_std
        
    def calculate_opacity(self, intensity, gradient):
        dx = (intensity - self.center_intensity) / self.intensity_std
        dy = (gradient - self.center_gradient) / self.gradient_std
        distance_sq = dx*dx + dy*dy
        return self.opacity * np.exp(-distance_sq / 2)
    
    def sample_for_vtk(self, num_samples=50):
        samples = []
        for i in range(num_samples):
            intensity = (i / num_samples) * 255
            opacity = self.calculate_opacity(intensity, self.center_gradient)
            if opacity > 0.001:
                samples.append((intensity, opacity, self.color))
        return samples
    
    def get_parameters(self):
        return {
            "center_intensity": {"value": self.center_intensity, "range": (0, 255), "type": "slider"},
            "center_gradient": {"value": self.center_gradient, "range": (0, 255), "type": "slider"},
            "intensity_std": {"value": self.intensity_std, "range": (5, 100), "type": "slider"},
            "gradient_std": {"value": self.gradient_std, "range": (5, 100), "type": "slider"},
            "opacity": {"value": self.opacity, "range": (0, 1), "type": "slider"}
        }
    
    def set_parameter(self, name, value):
        if name == "center_intensity":
            self.center_intensity = value
        elif name == "center_gradient":
            self.center_gradient = value
        elif name == "intensity_std":
            self.intensity_std = value
        elif name == "gradient_std":
            self.gradient_std = value
        elif name == "opacity":
            self.opacity = value

# Add other widget classes (Triangular, Rectangular, Ellipsoid) similarly...