import numpy as np
import traceback
from vtk.util import numpy_support
import vtk

class NDShaderRenderer:
    """
    VTK-baserad renderer för nD feature×feature rendering.
    """
    
    def __init__(self, image_data, feature_names, nd_manager, renderer_id="default"):
        try:
            self.renderer_id = renderer_id
            self.feature_names = feature_names
            self.nd_manager = nd_manager
            self.current_x_feature = None
            self.current_y_feature = None
            self.current_volume = None
            
            # ===== LÄGG TILL DESSA ATTRIBUT =====
            self.feature_normalization = {}  # ← Lades till!
            self.feature_volumes = {}        # ← Lades till!
            # =================================
            
            # Rendering inställningar
            self.sampling_rate = 2.0
            self.display_boost = 1.0
            self.display_gamma = 1.0
            
            print(f"\n🔧 Initializing VTK NDShaderRenderer: {renderer_id}")
            
            # ===== 1. EXTRAHERA FEATURE DATA FRÅN VTK =====
            dims = image_data.GetDimensions()
            point_data = image_data.GetPointData()
            
            self.dims = dims
            self.n_voxels = dims[0] * dims[1] * dims[2]
            
            for name in feature_names:
                array = point_data.GetArray(name)
                if array:
                    data = numpy_support.vtk_to_numpy(array)
                    min_val = float(np.min(data))
                    max_val = float(np.max(data))
                    if max_val > min_val:
                        data_normalized = (data - min_val) / (max_val - min_val)
                    else:
                        data_normalized = np.zeros_like(data)
                    
                    self.feature_volumes[name] = data_normalized
                    self.feature_normalization[name] = (min_val, max_val)
                    print(f"  Feature: {name} -> normalized to [0,1]")
            
            # Skapa renderer
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.1)
            
            # Lägg till bounding box
            self._add_bounding_box(dims)
            
            # Sätt kamera
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(dims[0], dims[1], dims[2])
            camera.SetFocalPoint(dims[0]/2, dims[1]/2, dims[2]/2)
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCameraClippingRange()
            
            print(f"✅ VTK NDShaderRenderer initialized")
        
        except Exception as e:
            print(f"❌ CRASH: {e}")
            traceback.print_exc()
            raise
    
    def _add_bounding_box(self, dims):
        """Lägg till en wireframe bounding box"""
        bounds = [0, dims[0], 0, dims[1], 0, dims[2]]
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetOpacity(0.3)
        actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(actor)
    
    def build_2d_texture(self, widgets):
        tf_size = 256
        tf_texture = np.zeros((tf_size, tf_size, 4), dtype=np.float32)

        for widget in widgets:
            x_pos = int(widget.center_intensity)
            y_pos = int(widget.center_gradient)
            x_pos = max(0, min(tf_size-1, x_pos))
            y_pos = max(0, min(tf_size-1, y_pos))

            # Identifiera widget-typ
            if hasattr(widget, 'intensity_std'):
                sigma_x = widget.intensity_std
                sigma_y = widget.gradient_std
                shape_type = 'gaussian'
            elif hasattr(widget, 'intensity_width'):
                sigma_x = widget.intensity_width / 2
                sigma_y = widget.gradient_height / 2
                if hasattr(widget, 'widget_type'):
                    wt = widget.widget_type
                    if wt.value == 'triangular':
                        shape_type = 'triangular'
                    elif wt.value == 'diamond':
                        shape_type = 'diamond'
                    else:
                        shape_type = 'rectangular'
                else:
                    shape_type = 'rectangular'
            elif hasattr(widget, 'intensity_radius'):
                sigma_x = widget.intensity_radius
                sigma_y = widget.gradient_radius
                shape_type = 'ellipsoid'
            else:
                sigma_x = 30
                sigma_y = 30
                shape_type = 'gaussian'

            sigma_x = max(1, sigma_x)
            sigma_y = max(1, sigma_y)
            radius_x = int(3 * sigma_x)
            radius_y = int(3 * sigma_y)

            print(f"   Widget: {shape_type} pos=({x_pos},{y_pos}), sigma=({sigma_x:.1f},{sigma_y:.1f})")

            # ===== ANVÄND WIDGETENS EGEN calculate_opacity =====
            # Detta är det ENKLARE och SÄKRARE sättet!
            for i in range(tf_size):
                for j in range(tf_size):
                    alpha = widget.calculate_opacity(i, j)
                
                    if alpha > tf_texture[j, i, 3]:
                        tf_texture[j, i, 0] = widget.color[0]
                        tf_texture[j, i, 1] = widget.color[1]
                        tf_texture[j, i, 2] = widget.color[2]
                        tf_texture[j, i, 3] = alpha

            # ===== DEBUG: Kolla alpha-värden för denna widget =====
            print(f"   🔍 Alpha at center ({x_pos},{y_pos}): {tf_texture[y_pos, x_pos, 3]:.3f}")
        
            # Kolla alpha längs en horisontell linje genom centrum
            values = []
            for i in range(max(0, x_pos-30), min(256, x_pos+30), 5):
                values.append(f"{tf_texture[y_pos, i, 3]:.2f}")
            print(f"   🔍 Alpha along x: {', '.join(values)}")
            # =============================================

        max_alpha = np.max(tf_texture[:,:,3])
        print(f"   Texture built: max alpha = {max_alpha:.3f}")
        non_zero = np.sum(tf_texture[:,:,3] > 0.01)
        print(f"   Non-zero texels: {non_zero} / {tf_size*tf_size}")

        return tf_texture
    
    def set_feature_pair(self, x_feature, y_feature):
        """
        HUVUDMETODEN: Sätt feature-par och uppdatera rendering.
        """
        try:
            print(f"\n🎯 Setting feature pair (VTK): {x_feature} vs {y_feature}")
            
            self.current_x_feature = x_feature
            self.current_y_feature = y_feature
            
            # Hämta widgets
            widgets = self.nd_manager.project_to_2d(x_feature, y_feature)
            print(f"   Found {len(widgets)} widgets")
            
            if len(widgets) == 0:
                from widget_factory import GaussianWidget
                default_widget = GaussianWidget()
                default_widget.center_intensity = 128
                default_widget.center_gradient = 128
                default_widget.opacity = 0.8
                default_widget.color = (0.8, 0.2, 0.2)
                widgets = [default_widget]
            
            # ===== 1. HÄMTA DATA =====
            if x_feature not in self.feature_volumes:
                print(f"   ❌ Error: {x_feature} not found in feature_volumes")
                return
            
            x_data_raw = self.feature_volumes[x_feature]
            y_data_raw = self.feature_volumes.get(y_feature, x_data_raw)
            
            print(f"   X ({x_feature}) range: 0 - 1")
            print(f"   Y ({y_feature}) range: 0 - 1")
            
            # ===== 2. BERÄKNA OPACITET FRÅN WIDGETS =====
            scalar_opacity = np.zeros(256, dtype=np.float32)
            gradient_opacity = np.zeros(256, dtype=np.float32)
            color_for_intensity = np.ones((256, 3), dtype=np.float32) * 0.5
            
            for widget in widgets:
                center_x = widget.center_intensity
                center_y = widget.center_gradient
                
                # Bestäm påverkansområde
                if hasattr(widget, 'intensity_std'):
                    sigma_x = widget.intensity_std
                    sigma_y = widget.gradient_std
                    radius_x = int(3 * sigma_x)
                    radius_y = int(3 * sigma_y)
                else:
                    radius_x = 90
                    radius_y = 90
                    sigma_x = 30
                    sigma_y = 30
                
                intensity_min = max(0, int(center_x - radius_x))
                intensity_max = min(255, int(center_x + radius_x))
                gradient_min = max(0, int(center_y - radius_y))
                gradient_max = min(255, int(center_y + radius_y))
                
                print(f"   Widget range: I[{intensity_min}-{intensity_max}], G[{gradient_min}-{gradient_max}]")
                
                # Projektion på intensitetsaxeln
                for intensity in range(intensity_min, intensity_max + 1):
                    max_opacity = 0
                    step = max(1, (gradient_max - gradient_min) // 50)
                    for gradient in range(gradient_min, gradient_max + 1, step):
                        opacity = widget.calculate_opacity(intensity, gradient)
                        if opacity > max_opacity:
                            max_opacity = opacity
                    
                    if max_opacity > scalar_opacity[intensity]:
                        scalar_opacity[intensity] = max_opacity
                        color_for_intensity[intensity] = widget.color
                
                # Projektion på gradientaxeln
                for gradient in range(gradient_min, gradient_max + 1):
                    max_opacity = 0
                    step = max(1, (intensity_max - intensity_min) // 50)
                    for intensity in range(intensity_min, intensity_max + 1, step):
                        opacity = widget.calculate_opacity(intensity, gradient)
                        if opacity > max_opacity:
                            max_opacity = opacity
                    
                    if max_opacity > gradient_opacity[gradient]:
                        gradient_opacity[gradient] = max_opacity
            
            # Applicera boost och gamma
            scalar_opacity = np.clip(scalar_opacity, 0, 1)
            gradient_opacity = np.clip(gradient_opacity, 0, 1)
            
            if self.display_boost != 1.0:
                scalar_opacity = np.clip(scalar_opacity * self.display_boost, 0, 1)
                gradient_opacity = np.clip(gradient_opacity * self.display_boost, 0, 1)
            
            if self.display_gamma != 1.0:
                scalar_opacity = np.power(scalar_opacity, 1.0/self.display_gamma)
                gradient_opacity = np.power(gradient_opacity, 1.0/self.display_gamma)
            
            print(f"   Scalar opacity max: {scalar_opacity.max():.3f}")
            print(f"   Gradient opacity max: {gradient_opacity.max():.3f}")
            
            # ===== 3. SKAPA VOLYM =====
            final_volume = vtk.vtkImageData()
            final_volume.SetDimensions(self.dims)
            final_volume.AllocateScalars(vtk.VTK_FLOAT, 1)
            
            # Använd X-feature som intensitetsdata
            vtk_scalars = numpy_support.numpy_to_vtk(x_data_raw.astype(np.float32))
            final_volume.GetPointData().SetScalars(vtk_scalars)
            
            # ===== 4. SKAPA TRANSFER FUNCTIONS =====
            color_func = vtk.vtkColorTransferFunction()
            for intensity in range(256):
                if scalar_opacity[intensity] > 0.01:
                    r, g, b = color_for_intensity[intensity]
                    norm_intensity = intensity / 255.0
                    color_func.AddRGBPoint(norm_intensity, r, g, b)
            
            # Om inga punkter lades till, lägg till en standard
            if color_func.GetSize() == 0:
                color_func.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
                color_func.AddRGBPoint(1.0, 0.5, 0.5, 0.5)
            
            scalar_opacity_func = vtk.vtkPiecewiseFunction()
            for intensity in range(256):
                if scalar_opacity[intensity] > 0.01:
                    norm_intensity = intensity / 255.0
                    scalar_opacity_func.AddPoint(norm_intensity, scalar_opacity[intensity])
            
            if scalar_opacity_func.GetSize() == 0:
                scalar_opacity_func.AddPoint(0.0, 0.0)
                scalar_opacity_func.AddPoint(1.0, 0.0)
            
            grad_opacity_func = vtk.vtkPiecewiseFunction()
            for gradient in range(256):
                if gradient_opacity[gradient] > 0.01:
                    gradient_norm = gradient / 255.0
                    grad_opacity_func.AddPoint(gradient_norm, gradient_opacity[gradient] * 0.5)
            
            # ===== 5. SKAPA VOLYM PROPERTY =====
            final_prop = vtk.vtkVolumeProperty()
            final_prop.SetColor(color_func)
            final_prop.SetScalarOpacity(scalar_opacity_func)
            final_prop.SetGradientOpacity(grad_opacity_func)
            final_prop.ShadeOn()
            final_prop.SetInterpolationTypeToLinear()
            final_prop.SetAmbient(0.2)
            final_prop.SetDiffuse(0.7)
            final_prop.SetSpecular(0.1)
            
            # ===== 6. SKAPA MAPPER OCH VOLYM =====
            final_mapper = vtk.vtkGPUVolumeRayCastMapper()
            final_mapper.SetInputData(final_volume)
            final_mapper.SetSampleDistance(1.0 / self.sampling_rate)
            final_mapper.SetAutoAdjustSampleDistances(False)
            
            final_volume_obj = vtk.vtkVolume()
            final_volume_obj.SetMapper(final_mapper)
            final_volume_obj.SetProperty(final_prop)
            
            # Ersätt existerande volymer
            for volume in list(self.renderer.GetVolumes()):
                self.renderer.RemoveVolume(volume)
            self.renderer.AddVolume(final_volume_obj)
            
            self.current_volume = final_volume_obj
            
            self.renderer.Render()
            print(f"   ✅ VTK render complete")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            traceback.print_exc()
    
    def get_renderer(self):
        return self.renderer
    
    def force_volume_visible(self):
        """Debug: tvinga volymen att synas"""
        print("\n🔧 FORCING volume to be visible")
        if self.current_volume:
            color_func = vtk.vtkColorTransferFunction()
            color_func.AddRGBPoint(0.0, 1.0, 0.0, 0.0)
            color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
            
            opacity_func = vtk.vtkPiecewiseFunction()
            opacity_func.AddPoint(0.0, 1.0)
            opacity_func.AddPoint(1.0, 1.0)
            
            self.current_volume.GetProperty().SetColor(color_func)
            self.current_volume.GetProperty().SetScalarOpacity(opacity_func)
            self.current_volume.GetProperty().ShadeOff()
            
            self.renderer.Render()
            print("   ✅ Volume forced to red")
    
    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(self.dims[0], self.dims[1], self.dims[2])
        camera.SetFocalPoint(self.dims[0]/2, self.dims[1]/2, self.dims[2]/2)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.renderer.Render()
    
    def set_sampling_rate(self, rate):
        self.sampling_rate = max(0.5, min(8.0, rate))
        if self.current_volume:
            self.current_volume.GetMapper().SetSampleDistance(1.0 / self.sampling_rate)
    
    def update_transfer_functions(self, intensities, opacities, colors,
                                  intensity_range=None, gradient_values=None,
                                  gradient_opacities=None, gradient_range=None):
        """Interface kompatibilitet"""
        if hasattr(self, 'current_x_feature') and self.current_x_feature:
            self.set_feature_pair(self.current_x_feature, self.current_y_feature)