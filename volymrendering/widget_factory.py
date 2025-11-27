# widget_factory.py
import numpy as np
from enum import Enum

class WidgetType(Enum):
    GAUSSIAN = "gaussian"
    TRIANGULAR = "triangular" 
    RECTANGULAR = "rectangular"
    ELLIPSOID = "ellipsoid"
    DIAMOND = "diamond"

class TFWidget:
    def __init__(self, widget_type, center_intensity=128, center_gradient=128, 
                 opacity=1.0, color=(1.0, 1.0, 1.0), blend_mode='max'):
        self.widget_type = widget_type
        self.center_intensity = center_intensity
        self.center_gradient = center_gradient
        self.opacity = opacity
        self.color = color
        self.blend_mode = blend_mode
        self.selected = False
        
    def calculate_opacity(self, intensity, gradient):
        raise NotImplementedError
        
    def get_parameters(self):
        return {
            "center_intensity": {"value": self.center_intensity, "range": (0, 255), "type": "slider", "step": 1},
            "center_gradient": {"value": self.center_gradient, "range": (0, 255), "type": "slider", "step": 1},
            "opacity": {"value": self.opacity, "range": (0, 1), "type": "slider", "step": 0.01},
            "blend_mode": {"value": self.blend_mode, "options": ["max", "add", "multiply"], "type": "combo"}
        }
        
    def set_parameter(self, name, value):
        if name == "center_intensity":
            self.center_intensity = max(0, min(255, int(value)))
        elif name == "center_gradient":
            self.center_gradient = max(0, min(255, int(value)))
        elif name == "opacity":
            self.opacity = max(0.0, min(1.0, float(value)))
        elif name == "blend_mode":
            self.blend_mode = value

class GaussianWidget(TFWidget):
    def __init__(self, center_intensity=128, center_gradient=128, 
                 intensity_std=30, gradient_std=30, falloff_power=2.0,
                 opacity=1.0, color=(1.0, 1.0, 1.0), blend_mode='max'):
        super().__init__(WidgetType.GAUSSIAN, center_intensity, center_gradient, opacity, color, blend_mode)
        self.intensity_std = max(1, intensity_std)  # Prevent division by zero
        self.gradient_std = max(1, gradient_std)
        self.falloff_power = falloff_power
        
    def calculate_opacity(self, intensity, gradient):
        # Proper 2D Gaussian
        dx = (intensity - self.center_intensity) / self.intensity_std
        dy = (gradient - self.center_gradient) / self.gradient_std
        distance_sq = dx*dx + dy*dy
        
        return self.opacity * np.exp(-distance_sq / 2)
    
    def get_parameters(self):
        base_params = super().get_parameters()
        base_params.update({
            "intensity_std": {"value": self.intensity_std, "range": (1, 100), "type": "slider", "step": 1},
            "gradient_std": {"value": self.gradient_std, "range": (1, 100), "type": "slider", "step": 1},
            "falloff_power": {"value": self.falloff_power, "range": (0.5, 5.0), "type": "slider", "step": 0.1},
        })
        return base_params
    
    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_std":
            self.intensity_std = max(1, int(value))  # Minimum 1 to avoid division by zero
        elif name == "gradient_std":
            self.gradient_std = max(1, int(value))
        elif name == "falloff_power":
            self.falloff_power = max(0.1, float(value))

class TriangularWidget(TFWidget):
    def __init__(self, center_intensity=128, center_gradient=128, 
                 intensity_width=50, gradient_height=50, direction='symmetric',
                 opacity=1.0, color=(1.0, 1.0, 1.0), blend_mode='max'):
        super().__init__(WidgetType.TRIANGULAR, center_intensity, center_gradient, opacity, color, blend_mode)
        self.intensity_width = max(1, intensity_width)
        self.gradient_height = max(1, gradient_height)
        self.direction = direction
        
    def calculate_opacity(self, intensity, gradient):
        # Calculate normalized distances from center
        dx = abs(intensity - self.center_intensity) / (self.intensity_width / 2)
        dy = abs(gradient - self.center_gradient) / (self.gradient_height / 2)
        
        if self.direction == 'up':
            # Only affects points above center
            if gradient < self.center_gradient:
                return 0.0
            relative_height = (gradient - self.center_gradient) / (self.gradient_height / 2)
            if dx > 1 or relative_height > 1:
                return 0.0
            return self.opacity * max(0, 1 - dx - relative_height)
            
        elif self.direction == 'down':
            # Only affects points below center
            if gradient > self.center_gradient:
                return 0.0
            relative_height = (self.center_gradient - gradient) / (self.gradient_height / 2)
            if dx > 1 or relative_height > 1:
                return 0.0
            return self.opacity * max(0, 1 - dx - relative_height)
            
        else:  # symmetric (default)
            # Classic diamond-like triangular shape
            if dx + dy > 1:
                return 0.0
            return self.opacity * max(0, 1 - dx - dy)
    
    def get_parameters(self):
        base_params = super().get_parameters()
        base_params.update({
            "intensity_width": {"value": self.intensity_width, "range": (10, 200), "type": "slider", "step": 1},
            "gradient_height": {"value": self.gradient_height, "range": (10, 200), "type": "slider", "step": 1},
            "direction": {"value": self.direction, "options": ["up", "down", "symmetric"], "type": "combo"},
        })
        return base_params
    
    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_width":
            self.intensity_width = max(1, int(value))
        elif name == "gradient_height":
            self.gradient_height = max(1, int(value))
        elif name == "direction":
            self.direction = value

class RectangularWidget(TFWidget):
    def __init__(self, center_intensity=128, center_gradient=128,
                 intensity_width=40, gradient_height=40, 
                 falloff=5.0,  # ← ADD FALLOFF
                 opacity=1.0, color=(1.0, 1.0, 1.0), blend_mode='max'):
        super().__init__(WidgetType.RECTANGULAR, center_intensity, center_gradient, opacity, color, blend_mode)
        self.intensity_width = max(1, intensity_width)
        self.gradient_height = max(1, gradient_height)
        self.falloff = max(0, falloff)  # Falloff distance
        
    def calculate_opacity(self, intensity, gradient):
        half_width = self.intensity_width / 2.0
        half_height = self.gradient_height / 2.0
    
        # Calculate normalized distances (0 = at edge, 1 = at falloff distance)
        dist_x = max(0, abs(intensity - self.center_intensity) - half_width) / max(1, self.falloff)
        dist_y = max(0, abs(gradient - self.center_gradient) - half_height) / max(1, self.falloff)
    
        # Use the maximum distance (falloff works from any edge)
        max_dist = max(dist_x, dist_y)
    
        if max_dist <= 1.0:  # Within falloff range
            return self.opacity * (1 - max_dist)
        else:
            return 0.0
            
    
    def get_parameters(self):
        base_params = super().get_parameters()
        base_params.update({
            "intensity_width": {"value": self.intensity_width, "range": (5, 200), "type": "slider", "step": 1},
            "gradient_height": {"value": self.gradient_height, "range": (5, 200), "type": "slider", "step": 1},
            "falloff": {"value": self.falloff, "range": (0, 50), "type": "slider", "step": 1},  # ← ADD FALLOFF PARAM
        })
        return base_params
    
    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_width":
            self.intensity_width = max(1, int(value))
        elif name == "gradient_height":
            self.gradient_height = max(1, int(value))
        elif name == "falloff":
            self.falloff = max(0, float(value))

class EllipsoidWidget(TFWidget):
    def __init__(self, center_intensity=128, center_gradient=128,
                 intensity_radius=30, gradient_radius=30,
                 falloff_power=1.0, opacity=1.0, color=(1.0, 1.0, 1.0), blend_mode='max'):
        super().__init__(WidgetType.ELLIPSOID, center_intensity, center_gradient, opacity, color, blend_mode)
        self.intensity_radius = max(1, intensity_radius)
        self.gradient_radius = max(1, gradient_radius)
        self.falloff_power = falloff_power
        
    def calculate_opacity(self, intensity, gradient):
        # Early exit optimization
        if (abs(intensity - self.center_intensity) > self.intensity_radius or
            abs(gradient - self.center_gradient) > self.gradient_radius):
            return 0.0
            
        dx = (intensity - self.center_intensity) / self.intensity_radius
        dy = (gradient - self.center_gradient) / self.gradient_radius
        distance = (dx**2 + dy**2) ** 0.5
        
        if distance > 1:
            return 0.0
        return self.opacity * max(0, 1 - distance ** self.falloff_power)
    
    def get_parameters(self):
        base_params = super().get_parameters()
        base_params.update({
            "intensity_radius": {"value": self.intensity_radius, "range": (5, 100), "type": "slider", "step": 1},
            "gradient_radius": {"value": self.gradient_radius, "range": (5, 100), "type": "slider", "step": 1},
            "falloff_power": {"value": self.falloff_power, "range": (0.5, 3.0), "type": "slider", "step": 0.1},
        })
        return base_params
    
    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_radius":
            self.intensity_radius = max(1, int(value))
        elif name == "gradient_radius":
            self.gradient_radius = max(1, int(value))
        elif name == "falloff_power":
            self.falloff_power = max(0.1, float(value))

class DiamondWidget(TFWidget):
    def __init__(self, center_intensity=128, center_gradient=128,
                 intensity_width=50, gradient_height=50,
                 opacity=1.0, color=(1.0, 1.0, 1.0), blend_mode='max'):
        super().__init__(WidgetType.DIAMOND, center_intensity, center_gradient, opacity, color, blend_mode)
        self.intensity_width = max(1, intensity_width)
        self.gradient_height = max(1, gradient_height)
        
    def calculate_opacity(self, intensity, gradient):
        # Early exit optimization
        if (abs(intensity - self.center_intensity) > self.intensity_width / 2 or
            abs(gradient - self.center_gradient) > self.gradient_height / 2):
            return 0.0
            
        dx = abs(intensity - self.center_intensity) / (self.intensity_width / 2)
        dy = abs(gradient - self.center_gradient) / (self.gradient_height / 2)
        
        if dx + dy > 1:
            return 0.0
        return self.opacity * max(0, 1 - (dx + dy))
    
    def get_parameters(self):
        base_params = super().get_parameters()
        base_params.update({
            "intensity_width": {"value": self.intensity_width, "range": (10, 200), "type": "slider", "step": 1},
            "gradient_height": {"value": self.gradient_height, "range": (10, 200), "type": "slider", "step": 1},
        })
        return base_params
    
    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_width":
            self.intensity_width = max(1, int(value))
        elif name == "gradient_height":
            self.gradient_height = max(1, int(value))

class WidgetFactory:
    @staticmethod
    def create_widget(widget_type, **kwargs):
        preset_name = kwargs.pop('preset', None)
        preset_config = WidgetFactory.get_preset(widget_type, preset_name)
        config = {**preset_config, **kwargs}
        
        if widget_type == WidgetType.GAUSSIAN:
            return GaussianWidget(**config)
        elif widget_type == WidgetType.TRIANGULAR:
            return TriangularWidget(**config)
        elif widget_type == WidgetType.RECTANGULAR:
            return RectangularWidget(**config)
        elif widget_type == WidgetType.ELLIPSOID:
            return EllipsoidWidget(**config)
        elif widget_type == WidgetType.DIAMOND:
            return DiamondWidget(**config)
        else:
            raise ValueError(f"Unknown widget type: {widget_type}")
    
    @staticmethod
    def get_preset(widget_type, preset_name):
        if preset_name is None:
            return {}
            
        presets = {
            WidgetType.GAUSSIAN: {
                'soft_tissue': {
                    'center_intensity': 120, 'center_gradient': 80,
                    'intensity_std': 25, 'gradient_std': 30, 'opacity': 0.6,
                    'color': (0.8, 0.8, 1.0)
                },
                'bone': {
                    'center_intensity': 200, 'center_gradient': 150,
                    'intensity_std': 15, 'gradient_std': 20, 'opacity': 0.9,
                    'color': (1.0, 1.0, 0.8)
                },
                'vessels': {
                    'center_intensity': 80, 'center_gradient': 180,
                    'intensity_std': 10, 'gradient_std': 8, 'opacity': 0.7,
                    'color': (1.0, 0.8, 0.8)
                }
            }
        }
        
        return presets.get(widget_type, {}).get(preset_name, {})