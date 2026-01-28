import vtk
from vtk.util import numpy_support


class VolumeRenderer:
    def __init__(self, renderer_id="default"):
        self.renderer_id = renderer_id
        
        # Each instance gets its OWN COMPLETE VTK pipeline
        self.renderer = vtk.vtkRenderer()
        self.mapper = vtk.vtkGPUVolumeRayCastMapper()
        self.color_function = vtk.vtkColorTransferFunction()
        self.opacity_function = vtk.vtkPiecewiseFunction()
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume = vtk.vtkVolume()
        
        self.setup_volume()

    # IN volume_renderer.py - ADD TO YOUR VolumeRenderer CLASS
    def setup_volume(self):
        """Initialize volume properties with texture size limits"""
        print(f"Setting up VolumeRenderer: {self.renderer_id}")
    
        # Configure volume propertyf
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
    
        # FIX: Set smaller texture size to avoid OpenGL warnings
        self.mapper.SetMaxMemoryInBytes(512 * 1024 * 1024)  # 512 MB limit
        self.mapper.SetAutoAdjustSampleDistances(0)  # Better control
    
        # Set up volume
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.volume_property)
    
        # Add volume to renderer
        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
    
        print(f"VolumeRenderer {self.renderer_id} setup complete")

    def update_transfer_functions(self, intensities, opacities, colors, intensity_range=None, 
                              gradient_opacities=None, gradient_range=None):
        """Update transfer functions with optional gradient opacity"""
        # Keep your existing scalar opacity and color setup
        scalar_opacity = vtk.vtkPiecewiseFunction()
        color_tf = vtk.vtkColorTransferFunction()
    
        # Add scalar opacity points
        for intensity, opacity in zip(intensities, opacities):
            if opacity > 0:
                scalar_opacity.AddPoint(intensity, opacity)
    
        # Add color points
        for intensity, color in zip(intensities, colors):
            if len(color) == 3:
                r, g, b = color
            else:
                r, g, b = 1.0, 1.0, 1.0
            color_tf.AddRGBPoint(intensity, r, g, b)
    
        # Set scalar opacity and color
        self.volume_property.SetScalarOpacity(scalar_opacity)
        self.volume_property.SetColor(color_tf)
    
        # NEW: Add gradient opacity if provided
        if gradient_opacities is not None:
            gradient_opacity = vtk.vtkPiecewiseFunction()
            for gradient, opacity in gradient_opacities:
                if opacity > 0:
                    gradient_opacity.AddPoint(gradient, opacity)
            self.volume_property.SetGradientOpacity(gradient_opacity)
            self.volume_property.ShadeOn()
        else:
            # Fallback: use a default gradient opacity
            gradient_opacity = vtk.vtkPiecewiseFunction()
            gradient_opacity.AddPoint(0, 0.0)
            gradient_opacity.AddPoint(50, 0.5)
            gradient_opacity.AddPoint(255, 1.0)
            self.volume_property.SetGradientOpacity(gradient_opacity)
    
        # Enable shading for better surface perception
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
    
        # Trigger render
       # self.render_window.Render()

    def set_volume_data(self, image_data, reader=None):
        """Set volume data for THIS instance."""
        if reader is not None:
            self.mapper.SetInputConnection(reader.GetOutputPort())
            print(f"Set volume data from reader for {self.renderer_id}")
        else:
            self.mapper.SetInputData(image_data)
            print(f"Set volume data from image_data for {self.renderer_id}")

    def reset_camera(self):
        """Reset camera for THIS instance."""
        self.renderer.ResetCamera()
        print(f"Reset camera for {self.renderer_id}")

    def get_renderer(self):
        """Get THIS instance's VTK renderer."""
        return self.renderer

    def get_mapper(self):
        """Get THIS instance's volume mapper."""
        return self.mapper

    def render(self):
        """Trigger render for THIS instance."""
        if hasattr(self, 'renderer') and self.renderer:
            render_window = self.renderer.GetRenderWindow()
            if render_window:
                render_window.Render()
                print(f"Rendered {self.renderer_id}")