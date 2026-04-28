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
    def __init__(
        self,
        widget_type,
        center_intensity=128,
        center_gradient=128,
        opacity=1.0,
        color=(1.0, 1.0, 1.0),
        blend_mode="max",
        falloff_type="linear",
    ):
        self.widget_type = widget_type
        self.center_intensity = center_intensity
        self.center_gradient = center_gradient
        self.opacity = opacity
        self.color = color
        self.blend_mode = blend_mode
        self.falloff_type = falloff_type
        self.selected = False

    def calculate_opacity(self, intensity, gradient):
        raise NotImplementedError

    def apply_falloff(self, distance):
        distance = float(distance)

        if self.falloff_type == "gaussian":
            return float(np.exp(-(distance * distance) / 2.0))

        if self.falloff_type == "linear":
            return max(0.0, 1.0 - distance)

        if self.falloff_type == "constant":
            return 1.0 if distance <= 1.0 else 0.0

        if self.falloff_type == "power2":
            return max(0.0, 1.0 - distance * distance)

        if self.falloff_type == "power3":
            return max(0.0, 1.0 - distance ** 3)

        return max(0.0, 1.0 - distance)

    def get_parameters(self):
        return {
            "center_intensity": {
                "value": self.center_intensity,
                "range": (0, 255),
                "type": "slider",
                "step": 1,
            },
            "center_gradient": {
                "value": self.center_gradient,
                "range": (0, 255),
                "type": "slider",
                "step": 1,
            },
            "opacity": {
                "value": self.opacity,
                "range": (0, 1),
                "type": "slider",
                "step": 0.01,
            },
            "blend_mode": {
                "value": self.blend_mode,
                "options": ["max", "add", "multiply"],
                "type": "combo",
            },
            "falloff_type": {
                "value": self.falloff_type,
                "options": ["gaussian", "linear", "constant", "power2", "power3"],
                "type": "combo",
            },
        }

    def set_parameter(self, name, value):
        if name == "center_intensity":
            self.center_intensity = max(0, min(255, float(value)))
        elif name == "center_gradient":
            self.center_gradient = max(0, min(255, float(value)))
        elif name == "opacity":
            self.opacity = max(0.0, min(1.0, float(value)))
        elif name == "blend_mode":
            self.blend_mode = value
        elif name == "falloff_type":
            self.falloff_type = value

    def sample(self):
        intensities = np.arange(256)
        gradients = np.arange(256)
        xv, yv = np.meshgrid(intensities, gradients, indexing="ij")

        vec_opacity = np.vectorize(self.calculate_opacity)
        opacity_grid = vec_opacity(xv, yv)

        opacity = opacity_grid.max(axis=1)
        colors = np.tile(self.color, (256, 1))

        return intensities, opacity, colors


class GaussianWidget(TFWidget):
    def __init__(
        self,
        center_intensity=128,
        center_gradient=128,
        intensity_std=12,
        gradient_std=12,
        falloff_power=2.0,
        opacity=1.0,
        color=(1.0, 1.0, 1.0),
        blend_mode="max",
        falloff_type="gaussian",
    ):
        super().__init__(
            WidgetType.GAUSSIAN,
            center_intensity,
            center_gradient,
            opacity,
            color,
            blend_mode,
            falloff_type,
        )
        self.intensity_std = max(1.0, float(intensity_std))
        self.gradient_std = max(1.0, float(gradient_std))
        self.falloff_power = max(0.1, float(falloff_power))

    def calculate_opacity(self, intensity, gradient):
        dx = (float(intensity) - self.center_intensity) / self.intensity_std
        dy = (float(gradient) - self.center_gradient) / self.gradient_std
        distance = np.sqrt(dx * dx + dy * dy)

        # Finite support: prevents Gaussian widgets from visually covering the whole TF.
        if distance > 3.0:
            return 0.0

        falloff = self.apply_falloff(distance)

        if self.falloff_power != 2.0:
            falloff = falloff ** (self.falloff_power / 2.0)

        return float(self.opacity * falloff)

    def get_parameters(self):
        params = super().get_parameters()
        params.update(
            {
                "intensity_std": {
                    "value": self.intensity_std,
                    "range": (1, 100),
                    "type": "slider",
                    "step": 1,
                },
                "gradient_std": {
                    "value": self.gradient_std,
                    "range": (1, 100),
                    "type": "slider",
                    "step": 1,
                },
                "falloff_power": {
                    "value": self.falloff_power,
                    "range": (0.5, 5.0),
                    "type": "slider",
                    "step": 0.1,
                },
            }
        )
        return params

    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_std":
            self.intensity_std = max(1.0, float(value))
        elif name == "gradient_std":
            self.gradient_std = max(1.0, float(value))
        elif name == "falloff_power":
            self.falloff_power = max(0.1, float(value))


class TriangularWidget(TFWidget):
    def __init__(
        self,
        center_intensity=128,
        center_gradient=128,
        intensity_width=70,
        gradient_height=70,
        direction="symmetric",
        opacity=1.0,
        color=(1.0, 1.0, 1.0),
        blend_mode="max",
        falloff_type="linear",
    ):
        super().__init__(
            WidgetType.TRIANGULAR,
            center_intensity,
            center_gradient,
            opacity,
            color,
            blend_mode,
            falloff_type,
        )
        self.intensity_width = max(1.0, float(intensity_width))
        self.gradient_height = max(1.0, float(gradient_height))
        self.direction = direction

    def calculate_opacity(self, intensity, gradient):
        half_width = self.intensity_width / 2.0
        half_height = self.gradient_height / 2.0

        dx = abs(float(intensity) - self.center_intensity) / half_width
        dy_signed = (float(gradient) - self.center_gradient) / half_height

        if self.direction == "up":
            if dy_signed < 0.0 or dy_signed > 1.0:
                return 0.0

            width_at_y = 1.0 - dy_signed
            if width_at_y <= 0.0 or dx > width_at_y:
                return 0.0

            distance = max(dx / width_at_y, dy_signed)

        elif self.direction == "down":
            if dy_signed > 0.0 or dy_signed < -1.0:
                return 0.0

            t = -dy_signed
            width_at_y = 1.0 - t
            if width_at_y <= 0.0 or dx > width_at_y:
                return 0.0

            distance = max(dx / width_at_y, t)

        else:
            dy = abs(dy_signed)
            distance = dx + dy
            if distance > 1.0:
                return 0.0

        falloff = self.apply_falloff(distance)
        return float(self.opacity * falloff)

    def get_parameters(self):
        params = super().get_parameters()
        params.update(
            {
                "intensity_width": {
                    "value": self.intensity_width,
                    "range": (1, 255),
                    "type": "slider",
                    "step": 1,
                },
                "gradient_height": {
                    "value": self.gradient_height,
                    "range": (1, 255),
                    "type": "slider",
                    "step": 1,
                },
                "direction": {
                    "value": self.direction,
                    "options": ["up", "down", "symmetric"],
                    "type": "combo",
                },
            }
        )
        return params

    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_width":
            self.intensity_width = max(1.0, float(value))
        elif name == "gradient_height":
            self.gradient_height = max(1.0, float(value))
        elif name == "direction":
            self.direction = value


class RectangularWidget(TFWidget):
    def __init__(
        self,
        center_intensity=128,
        center_gradient=128,
        intensity_width=70,
        gradient_height=70,
        falloff=0.0,
        opacity=1.0,
        color=(1.0, 1.0, 1.0),
        blend_mode="max",
        falloff_type="constant",
    ):
        super().__init__(
            WidgetType.RECTANGULAR,
            center_intensity,
            center_gradient,
            opacity,
            color,
            blend_mode,
            falloff_type,
        )
        self.intensity_width = max(1.0, float(intensity_width))
        self.gradient_height = max(1.0, float(gradient_height))
        self.falloff = max(0.0, float(falloff))

    def calculate_opacity(self, intensity, gradient):
        half_width = self.intensity_width / 2.0
        half_height = self.gradient_height / 2.0

        dx = abs(float(intensity) - self.center_intensity)
        dy = abs(float(gradient) - self.center_gradient)

        inside_x = dx <= half_width
        inside_y = dy <= half_height

        if inside_x and inside_y:
            return float(self.opacity)

        # Optional outer falloff border.
        if self.falloff <= 0.0:
            return 0.0

        outer_x = half_width + self.falloff
        outer_y = half_height + self.falloff

        if dx > outer_x or dy > outer_y:
            return 0.0

        dist_x = max(0.0, dx - half_width) / self.falloff
        dist_y = max(0.0, dy - half_height) / self.falloff
        distance = max(dist_x, dist_y)

        falloff = self.apply_falloff(distance)
        return float(self.opacity * falloff)

    def get_parameters(self):
        params = super().get_parameters()
        params.update(
            {
                "intensity_width": {
                    "value": self.intensity_width,
                    "range": (1, 255),
                    "type": "slider",
                    "step": 1,
                },
                "gradient_height": {
                    "value": self.gradient_height,
                    "range": (1, 255),
                    "type": "slider",
                    "step": 1,
                },
                "falloff": {
                    "value": self.falloff,
                    "range": (0, 80),
                    "type": "slider",
                    "step": 1,
                },
            }
        )
        return params

    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_width":
            self.intensity_width = max(1.0, float(value))
        elif name == "gradient_height":
            self.gradient_height = max(1.0, float(value))
        elif name == "falloff":
            self.falloff = max(0.0, float(value))


class EllipsoidWidget(TFWidget):
    def __init__(
        self,
        center_intensity=128,
        center_gradient=128,
        intensity_radius=35,
        gradient_radius=35,
        falloff_power=1.0,
        opacity=1.0,
        color=(1.0, 1.0, 1.0),
        blend_mode="max",
        falloff_type="linear",
    ):
        super().__init__(
            WidgetType.ELLIPSOID,
            center_intensity,
            center_gradient,
            opacity,
            color,
            blend_mode,
            falloff_type,
        )
        self.intensity_radius = max(1.0, float(intensity_radius))
        self.gradient_radius = max(1.0, float(gradient_radius))
        self.falloff_power = max(0.1, float(falloff_power))

    def calculate_opacity(self, intensity, gradient):
        dx = (float(intensity) - self.center_intensity) / self.intensity_radius
        dy = (float(gradient) - self.center_gradient) / self.gradient_radius
        distance = np.sqrt(dx * dx + dy * dy)

        if distance > 1.0:
            return 0.0

        falloff = self.apply_falloff(distance)

        if self.falloff_power != 1.0:
            falloff = falloff ** self.falloff_power

        return float(self.opacity * falloff)

    def get_parameters(self):
        params = super().get_parameters()
        params.update(
            {
                "intensity_radius": {
                    "value": self.intensity_radius,
                    "range": (1, 150),
                    "type": "slider",
                    "step": 1,
                },
                "gradient_radius": {
                    "value": self.gradient_radius,
                    "range": (1, 150),
                    "type": "slider",
                    "step": 1,
                },
                "falloff_power": {
                    "value": self.falloff_power,
                    "range": (0.1, 5.0),
                    "type": "slider",
                    "step": 0.1,
                },
            }
        )
        return params

    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_radius":
            self.intensity_radius = max(1.0, float(value))
        elif name == "gradient_radius":
            self.gradient_radius = max(1.0, float(value))
        elif name == "falloff_power":
            self.falloff_power = max(0.1, float(value))


class DiamondWidget(TFWidget):
    def __init__(
        self,
        center_intensity=128,
        center_gradient=128,
        intensity_width=70,
        gradient_height=70,
        opacity=1.0,
        color=(1.0, 1.0, 1.0),
        blend_mode="max",
        falloff_type="linear",
    ):
        super().__init__(
            WidgetType.DIAMOND,
            center_intensity,
            center_gradient,
            opacity,
            color,
            blend_mode,
            falloff_type,
        )
        self.intensity_width = max(1.0, float(intensity_width))
        self.gradient_height = max(1.0, float(gradient_height))

    def calculate_opacity(self, intensity, gradient):
        half_width = self.intensity_width / 2.0
        half_height = self.gradient_height / 2.0

        dx = abs(float(intensity) - self.center_intensity) / half_width
        dy = abs(float(gradient) - self.center_gradient) / half_height

        distance = dx + dy

        if distance > 1.0:
            return 0.0

        falloff = self.apply_falloff(distance)
        return float(self.opacity * falloff)

    def get_parameters(self):
        params = super().get_parameters()
        params.update(
            {
                "intensity_width": {
                    "value": self.intensity_width,
                    "range": (1, 255),
                    "type": "slider",
                    "step": 1,
                },
                "gradient_height": {
                    "value": self.gradient_height,
                    "range": (1, 255),
                    "type": "slider",
                    "step": 1,
                },
            }
        )
        return params

    def set_parameter(self, name, value):
        super().set_parameter(name, value)
        if name == "intensity_width":
            self.intensity_width = max(1.0, float(value))
        elif name == "gradient_height":
            self.gradient_height = max(1.0, float(value))


class WidgetFactory:
    @staticmethod
    def create_widget(widget_type, **kwargs):
        preset_name = kwargs.pop("preset", None)
        preset_config = WidgetFactory.get_preset(widget_type, preset_name)
        config = {**preset_config, **kwargs}

        if "falloff_type" not in config:
            if widget_type == WidgetType.GAUSSIAN:
                config["falloff_type"] = "gaussian"
            elif widget_type == WidgetType.RECTANGULAR:
                config["falloff_type"] = "constant"
            else:
                config["falloff_type"] = "linear"

        if widget_type == WidgetType.GAUSSIAN:
            return GaussianWidget(**config)

        if widget_type == WidgetType.TRIANGULAR:
            return TriangularWidget(**config)

        if widget_type == WidgetType.RECTANGULAR:
            return RectangularWidget(**config)

        if widget_type == WidgetType.ELLIPSOID:
            return EllipsoidWidget(**config)

        if widget_type == WidgetType.DIAMOND:
            return DiamondWidget(**config)

        raise ValueError(f"Unknown widget type: {widget_type}")

    @staticmethod
    def get_preset(widget_type, preset_name):
        if preset_name is None:
            return {}

        presets = {
            WidgetType.GAUSSIAN: {
                "soft_tissue": {
                    "center_intensity": 128,
                    "center_gradient": 128,
                    "intensity_std": 22,
                    "gradient_std": 22,
                    "opacity": 0.6,
                    "color": (0.8, 0.8, 1.0),
                },
                "bone": {
                    "center_intensity": 200,
                    "center_gradient": 150,
                    "intensity_std": 8,
                    "gradient_std": 10,
                    "opacity": 0.9,
                    "color": (1.0, 1.0, 0.8),
                },
                "vessels": {
                    "center_intensity": 80,
                    "center_gradient": 180,
                    "intensity_std": 6,
                    "gradient_std": 6,
                    "opacity": 0.7,
                    "color": (1.0, 0.8, 0.8),
                },
            }
        }

        return presets.get(widget_type, {}).get(preset_name, {})