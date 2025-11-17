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

    def setup_volume(self):
        """Initialize volume properties and setup for THIS instance."""
        print(f"Setting up VolumeRenderer: {self.renderer_id}")
        
        # Configure volume property for THIS instance
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()

        # Set up volume for THIS instance
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.volume_property)
        
        # Add volume to THIS instance's renderer
        self.renderer.AddVolume(self.volume)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        print(f"VolumeRenderer {self.renderer_id} setup complete")

    def update_transfer_functions(self, points_x, points_y, colors, intensity_range):
        """Update THIS instance's transfer functions."""
        raw_int_min, raw_int_max = intensity_range
        
        # Clear old points from THIS instance
        self.color_function.RemoveAllPoints()
        self.opacity_function.RemoveAllPoints()

        # Add new points to THIS instance
        for x, y, c in zip(points_x, points_y, colors):
            abs_val = raw_int_min + (x / 255.0) * (raw_int_max - raw_int_min)
            self.opacity_function.AddPoint(abs_val, y)
            self.color_function.AddRGBPoint(abs_val, *c)
            
        print(f"Updated TF for {self.renderer_id} with {len(points_x)} points")

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