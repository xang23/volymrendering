# multicomponent_renderer.py
import vtk
import numpy as np
from vtk.util import numpy_support

class MultiComponentRenderer:
    def __init__(self, multi_volume, feature_names, renderer_id="default"):
        self.renderer_id = renderer_id
        self.multi_volume = multi_volume
        self.feature_names = feature_names
        
        print(f"Initializing MultiComponentRenderer: {renderer_id}")
        
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.mapper.SetInputData(multi_volume)
        
        # Volume property
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetIndependentComponents(True)
        
        # Create volume
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.volume_property)
        
        # Add to renderer
        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        print(f"MultiComponentRenderer {renderer_id} ready with {len(feature_names)} features")
    
    def set_feature_pair(self, x_feature, y_feature, widget_samples):
        """Set up rendering using x_feature as X-axis, y_feature as Y-axis"""
        
        print(f"Setting feature pair: {x_feature} vs {y_feature}")
        
        x_idx = self.feature_names.index(x_feature)
        y_idx = self.feature_names.index(y_feature)
        
        # Create transfer functions
        color_tf = vtk.vtkColorTransferFunction()
        opacity_tf = vtk.vtkPiecewiseFunction()
        
        # Get data ranges for scaling
        x_data = self.get_feature_data(x_feature)
        x_min, x_max = np.min(x_data), np.max(x_data)
        
        # Add points from widgets
        for intensity, opacity, color in widget_samples:
            scaled_x = x_min + (intensity / 255.0) * (x_max - x_min)
            
            if opacity > 0.01:
                opacity_tf.AddPoint(scaled_x, opacity)
            
            if len(color) == 3:
                r, g, b = color
                color_tf.AddRGBPoint(scaled_x, r, g, b)
        
        # Ensure boundaries
        opacity_tf.AddPoint(x_min, 0.0)
        opacity_tf.AddPoint(x_max, 0.0)
        color_tf.AddRGBPoint(x_min, 0.5, 0.5, 0.5)
        color_tf.AddRGBPoint(x_max, 0.5, 0.5, 0.5)
        
        # Assign to components
        self.volume_property.SetColor(x_idx, color_tf)
        self.volume_property.SetScalarOpacity(y_idx, opacity_tf)
    
    def get_feature_data(self, feature_name):
        """Get numpy array for a specific feature"""
        idx = self.feature_names.index(feature_name)
        vtk_array = self.multi_volume.GetPointData().GetScalars()
        np_array = numpy_support.vtk_to_numpy(vtk_array)
        return np_array[:, idx]
    
    def get_renderer(self):
        return self.renderer
    
    def reset_camera(self):
        self.renderer.ResetCamera()