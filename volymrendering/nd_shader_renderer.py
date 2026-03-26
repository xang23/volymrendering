import vtk
import numpy as np
from vtk.util import numpy_support

class NDShaderRenderer:
    def __init__(self, image_data, feature_names, nd_manager, renderer_id="default"):
        try:
            self.renderer_id = renderer_id
            self.feature_names = feature_names
            self.nd_manager = nd_manager
            self.current_x_feature = None
            self.current_y_feature = None

            print(f"\n🔧 Initializing NDShaderRenderer: {renderer_id}")
            print(f"   Features: {feature_names}")

            dims = image_data.GetDimensions()
            n_voxels = dims[0] * dims[1] * dims[2]
            print(f"   Volume dimensions: {dims}, voxels: {n_voxels}")

            point_data = image_data.GetPointData()

            # Collect and normalize feature data
            feature_data = []
            self.feature_normalization = {}
            self.feature_to_volume = {}
            self.volume_n_comps = []
            self.volumes = []
            self.color_functions = {}
            self.opacity_functions = {}
            self.gradient_opacity_functions = {}

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
                
                    feature_data.append(data_normalized)
                    self.feature_normalization[name] = (min_val, max_val)
                    print(f"  Feature: {name} -> normalized to [0,1]")

            # Split into chunks of 4
            chunks = [feature_data[i:i+4] for i in range(0, len(feature_data), 4)]
            print(f"   Splitting into {len(chunks)} GPU volumes")

            global_idx = 0
            for vol_idx, chunk in enumerate(chunks):
                n_comps = len(chunk)
                self.volume_n_comps.append(n_comps)
                print(f"   Volume {vol_idx}: {n_comps} components")
            
                # Create multi-component array
                multi_array = np.zeros((n_voxels, n_comps), dtype=np.float32)
                for comp_idx, data in enumerate(chunk):
                    multi_array[:, comp_idx] = data
                    name = feature_names[global_idx + comp_idx]
                    self.feature_to_volume[name] = (vol_idx, comp_idx)
                    print(f"      Component {comp_idx}: {name}")
            
                global_idx += n_comps

                # Convert to VTK
                vtk_array = numpy_support.numpy_to_vtk(multi_array)
                del multi_array

                vol_data = vtk.vtkImageData()
                vol_data.SetDimensions(dims)
                vol_data.GetPointData().SetScalars(vtk_array)

                # Create mapper
                mapper = vtk.vtkGPUVolumeRayCastMapper()
                mapper.SetInputData(vol_data)

                # Create property
                prop = vtk.vtkVolumeProperty()
                prop.SetIndependentComponents(True)
                prop.ShadeOn()
                prop.SetInterpolationTypeToLinear()

                # Initialize transfer functions
                for comp_idx in range(n_comps):
                    color_func = vtk.vtkColorTransferFunction()
                    opacity_func = vtk.vtkPiecewiseFunction()
                    grad_opacity_func = vtk.vtkPiecewiseFunction()
                
                    # Start invisible
                    color_func.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
                    color_func.AddRGBPoint(1.0, 0.5, 0.5, 0.5)
                    opacity_func.AddPoint(0.0, 0.0)
                    opacity_func.AddPoint(1.0, 0.0)
                    grad_opacity_func.AddPoint(0.0, 0.0)
                    grad_opacity_func.AddPoint(1.0, 0.0)
                
                    prop.SetColor(comp_idx, color_func)
                    prop.SetScalarOpacity(comp_idx, opacity_func)
                    prop.SetGradientOpacity(comp_idx, grad_opacity_func)
                
                    self.color_functions[(vol_idx, comp_idx)] = color_func
                    self.opacity_functions[(vol_idx, comp_idx)] = opacity_func
                    self.gradient_opacity_functions[(vol_idx, comp_idx)] = grad_opacity_func

                volume = vtk.vtkVolume()
                volume.SetMapper(mapper)
                volume.SetProperty(prop)
                self.volumes.append(volume)

            # Create renderer
            print(f"   Using volume {0}, component {0} for Intensity")
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.1, 0.1)
            for volume in self.volumes:
                self.renderer.AddVolume(volume)
            
            # Add bounding box
            self.add_bounding_box(dims)
            
            # Set camera
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(400, 400, 400)
            camera.SetFocalPoint(128, 128, 47)
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCameraClippingRange()

            print(f"✅ NDShaderRenderer initialized with {len(self.volumes)} volumes")
        
        except Exception as e:
            print(f"❌ CRASH: {e}")
            import traceback
            traceback.print_exc()
            raise

    def add_bounding_box(self, dims):
        bounds = [0, dims[0], 0, dims[1], 0, dims[2]]
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetOpacity(0.5)
        actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(actor)

    def build_2d_texture(self, widgets):
        """Build 2D texture using each widget's own parameters (matching the 2D canvas)"""
        tf_size = 256
        tf_texture = np.zeros((tf_size, tf_size, 4), dtype=np.float32)
    
        for widget in widgets:
            x_pos = int(widget.center_intensity)
            y_pos = int(widget.center_gradient)
            x_pos = max(0, min(tf_size-1, x_pos))
            y_pos = max(0, min(tf_size-1, y_pos))
        
            # ===== USE WIDGET'S OWN PARAMETERS! =====
            # Get sigma values from widget (matching GaussianWidget)
            if hasattr(widget, 'intensity_std'):
                sigma_x = widget.intensity_std
            else:
                sigma_x = 30  # fallback
            
            if hasattr(widget, 'gradient_std'):
                sigma_y = widget.gradient_std
            else:
                sigma_y = 30  # fallback
        
            # Use the same formula as GaussianWidget.calculate_opacity
            radius_x = int(3 * sigma_x)
            radius_y = int(3 * sigma_y)
        
            print(f"   Widget: pos=({x_pos},{y_pos}), sigma=({sigma_x:.1f},{sigma_y:.1f}), opacity={widget.opacity}")
        
            # Build the Gaussian contribution
            for i in range(max(0, x_pos - radius_x), min(tf_size, x_pos + radius_x)):
                for j in range(max(0, y_pos - radius_y), min(tf_size, y_pos + radius_y)):
                    # Use the SAME formula as GaussianWidget.calculate_opacity
                    dx = (i - x_pos) / sigma_x
                    dy = (j - y_pos) / sigma_y
                    distance_sq = dx*dx + dy*dy
                    alpha = widget.opacity * np.exp(-distance_sq / 2)  # Exactly like widget's method!
                
                    # Blend (max blend mode, matching canvas)
                    if alpha > tf_texture[j, i, 3]:
                        tf_texture[j, i, 0] = widget.color[0]
                        tf_texture[j, i, 1] = widget.color[1]
                        tf_texture[j, i, 2] = widget.color[2]
                        tf_texture[j, i, 3] = alpha
    
        # Debug: print max alpha
        max_alpha = np.max(tf_texture[:,:,3])
        print(f"   Texture built: max alpha = {max_alpha:.3f}")
        non_zero = np.sum(tf_texture[:,:,3] > 0.01)
        print(f"   Non-zero texels: {non_zero} / {tf_size*tf_size}")
    
        return tf_texture

    def set_feature_pair(self, x_feature, y_feature):
        """Use EXACT same logic as UnifiedTFCanvas._sample_2d_for_vtk_dual_functions()"""
        try:
            print(f"\n🎯 Setting feature pair: {x_feature} vs {y_feature}")

            if x_feature not in self.feature_to_volume:
                print(f"❌ Error: {x_feature} not found")
                return
            if y_feature not in self.feature_to_volume:
                print(f"❌ Error: {y_feature} not found")
                return
        
            self.current_x_feature = x_feature
            self.current_y_feature = y_feature
        
            # Get volume indices for both features
            x_vol, x_comp = self.feature_to_volume[x_feature]
            y_vol, y_comp = self.feature_to_volume[y_feature]
        
            # ===== GET FRESH WIDGET PROJECTION =====
            # This should get the CURRENT state, not accumulate
            widgets = self.nd_manager.project_to_2d(x_feature, y_feature)
            print(f"   Found {len(widgets)} widgets")
        
            if len(widgets) == 0:
                from tf_widgets import GaussianWidget
                default_widget = GaussianWidget()
                default_widget.center_intensity = 128
                default_widget.center_gradient = 128
                default_widget.opacity = 0.8
                default_widget.color = (1.0, 0.0, 0.0)
                widgets = [default_widget]
        
            # Extract X feature data
            original_volume = self.volumes[x_vol]
            original_data = original_volume.GetMapper().GetInput()
            dims = original_data.GetDimensions()
            n_voxels = dims[0] * dims[1] * dims[2]
        
            scalars = original_data.GetPointData().GetScalars()
            data = numpy_support.vtk_to_numpy(scalars)
        
            if len(data.shape) == 2:
                x_data = data[:, x_comp]
            else:
                x_data = data
        
            print(f"   Extracted {x_feature}: min={x_data.min():.3f}, max={x_data.max():.3f}")
        
            # Extract Y feature data
            if y_vol == x_vol and y_comp == x_comp:
                y_data = x_data
            else:
                y_original_volume = self.volumes[y_vol]
                y_original_data = y_original_volume.GetMapper().GetInput()
                y_scalars = y_original_data.GetPointData().GetScalars()
                y_data_all = numpy_support.vtk_to_numpy(y_scalars)
                if len(y_data_all.shape) == 2:
                    y_data = y_data_all[:, y_comp]
                else:
                    y_data = y_data_all
        
            print(f"   Extracted {y_feature}: min={y_data.min():.3f}, max={y_data.max():.3f}")
        
            # Convert to display space (0-255)
            x_display = x_data * 255.0
            y_display = y_data * 255.0
        
            # ===== RESET TRANSFER FUNCTIONS =====
            # IMPORTANT: Start fresh each time!
            scalar_opacity = np.zeros(256, dtype=np.float32)
            gradient_opacity = np.zeros(256, dtype=np.float32)
            color_for_intensity = np.ones((256, 3), dtype=np.float32) * 0.5  # Default gray
        
            print(f"\n   🔄 Building transfer functions using canvas logic...")
        
            # For each widget, project onto scalar and gradient axes
            for widget in widgets:
                # Get widget parameters
                center_x = widget.center_intensity
                center_y = widget.center_gradient
            
                # Get widget size
                if hasattr(widget, 'intensity_std'):
                    sigma_x = widget.intensity_std
                    sigma_y = widget.gradient_std
                else:
                    sigma_x = 30
                    sigma_y = 30
            
                # Define influence range
                intensity_min = max(0, int(center_x - 3 * sigma_x))
                intensity_max = min(255, int(center_x + 3 * sigma_x))
                gradient_min = max(0, int(center_y - 3 * sigma_y))
                gradient_max = min(255, int(center_y + 3 * sigma_y))
            
                print(f"   Widget: center=({center_x:.1f},{center_y:.1f}), sigma=({sigma_x:.1f},{sigma_y:.1f})")
                print(f"      Intensity range: {intensity_min}-{intensity_max}")
                print(f"      Gradient range: {gradient_min}-{gradient_max}")
            
                # Project onto scalar axis
                for intensity in range(intensity_min, intensity_max + 1):
                    max_opacity = 0
                    for gradient in range(gradient_min, gradient_max + 1, 4):
                        dx = (intensity - center_x) / sigma_x
                        dy = (gradient - center_y) / sigma_y
                        dist_sq = dx*dx + dy*dy
                        opacity = widget.opacity * np.exp(-dist_sq / 2)
                        max_opacity = max(max_opacity, opacity)
                
                    # Max blending - OVERWRITE if greater (not accumulate)
                    if max_opacity > scalar_opacity[intensity]:
                        scalar_opacity[intensity] = max_opacity
                        color_for_intensity[intensity] = widget.color
            
                # Project onto gradient axis
                for gradient in range(gradient_min, gradient_max + 1):
                    max_opacity = 0
                    for intensity in range(intensity_min, intensity_max + 1, 4):
                        dx = (intensity - center_x) / sigma_x
                        dy = (gradient - center_y) / sigma_y
                        dist_sq = dx*dx + dy*dy
                        opacity = widget.opacity * np.exp(-dist_sq / 2)
                        max_opacity = max(max_opacity, opacity)
                
                    if max_opacity > gradient_opacity[gradient]:
                        gradient_opacity[gradient] = max_opacity
        
            # Clamp to 0-1 range
            scalar_opacity = np.clip(scalar_opacity, 0, 1)
            gradient_opacity = np.clip(gradient_opacity, 0, 1)
        
            print(f"\n   Scalar opacity range: {scalar_opacity.min():.3f} - {scalar_opacity.max():.3f}")
            print(f"   Non-zero scalar points: {np.sum(scalar_opacity > 0.01)} / 256")
        
            # ===== SAMPLE VOLUME WITH FRESH OPACITY VALUES =====
            print(f"\n   🔄 Sampling volume with pre-computed transfer functions...")
        
            # Create fresh opacity array
            opacity_data = np.zeros(n_voxels, dtype=np.float32)
        
            # Process in chunks
            chunk_size = 1000000
            for start_idx in range(0, n_voxels, chunk_size):
                end_idx = min(start_idx + chunk_size, n_voxels)
                chunk_x = x_display[start_idx:end_idx].astype(np.int32)
                chunk_x = np.clip(chunk_x, 0, 255)
            
                # Look up opacity from scalar_opacity array (FRESH each time)
                chunk_opacity = scalar_opacity[chunk_x]
                opacity_data[start_idx:end_idx] = chunk_opacity
            
                if (start_idx // chunk_size) % 10 == 0:
                    print(f"      Processed {end_idx}/{n_voxels} voxels")
        
            print(f"   ✅ Sampling complete")
            print(f"   Opacity range: {opacity_data.min():.3f} - {opacity_data.max():.3f}")
            non_zero = np.sum(opacity_data > 0.01)
            print(f"   Non-zero opacity: {non_zero} / {n_voxels} voxels ({100*non_zero/n_voxels:.2f}%)")
        
            # ===== CREATE NEW VOLUME (REPLACE OLD ONE) =====
            final_volume = vtk.vtkImageData()
            final_volume.SetDimensions(dims)
            final_volume.AllocateScalars(vtk.VTK_FLOAT, 1)
            vtk_opacity = numpy_support.numpy_to_vtk(opacity_data.astype(np.float32))
            final_volume.GetPointData().SetScalars(vtk_opacity)
        
            # Create color function (fresh each time)
            color_func = vtk.vtkColorTransferFunction()
            opacity_func = vtk.vtkPiecewiseFunction()
        
            # Add points for each intensity where opacity > 0
            for intensity in range(256):
                if scalar_opacity[intensity] > 0.01:
                    r, g, b = color_for_intensity[intensity]
                    color_func.AddRGBPoint(intensity / 255.0, r, g, b)
                    opacity_func.AddPoint(intensity / 255.0, scalar_opacity[intensity])
        
            # Add endpoints
            color_func.AddRGBPoint(0.0, 0.3, 0.3, 0.3)
            color_func.AddRGBPoint(1.0, 0.5, 0.5, 0.5)
            opacity_func.AddPoint(0.0, 0.0)
            opacity_func.AddPoint(1.0, 0.8)
        
            # Create gradient opacity function
            grad_opacity_func = vtk.vtkPiecewiseFunction()
            for gradient in range(256):
                if gradient_opacity[gradient] > 0.01:
                    grad_opacity_func.AddPoint(gradient / 255.0, gradient_opacity[gradient])
        
            # Create property and volume
            final_prop = vtk.vtkVolumeProperty()
            final_prop.SetColor(color_func)
            final_prop.SetScalarOpacity(opacity_func)
            final_prop.SetGradientOpacity(grad_opacity_func)
            final_prop.ShadeOn()
            final_prop.SetInterpolationTypeToLinear()
        
            final_mapper = vtk.vtkGPUVolumeRayCastMapper()
            final_mapper.SetInputData(final_volume)
        
            final_volume_obj = vtk.vtkVolume()
            final_volume_obj.SetMapper(final_mapper)
            final_volume_obj.SetProperty(final_prop)
        
            # ===== REPLACE VOLUME (not accumulate!) =====
            # Remove all existing volumes
            for volume in list(self.renderer.GetVolumes()):  # Use list to avoid modification during iteration
                self.renderer.RemoveVolume(volume)
        
            # Add only the new volume
            self.renderer.AddVolume(final_volume_obj)
        
            self.current_volume = final_volume_obj
        
            # Set camera if needed
            bounds = final_volume.GetBounds()
            center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
            distance = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]) * 1.5
        
            camera = self.renderer.GetActiveCamera()
            camera.SetPosition(center[0] + distance, center[1] + distance * 0.8, center[2] + distance)
            camera.SetFocalPoint(center[0], center[1], center[2])
            camera.SetViewUp(0, 0, 1)
            self.renderer.ResetCameraClippingRange()
        
            self.renderer.Render()
            print(f"   ✅ Applied fresh transfer function (replaced old volume)")
        
            # Cleanup
            del x_data
            del y_data
            del opacity_data
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    def force_volume_visible(self):
        """Force the volume to be bright red and fully opaque"""
        print("\n🔧 FORCING volume to be visible")
        
        # Get Intensity component (volume 0, component 0)
        color_func = self.color_functions[(0, 0)]
        opacity_func = self.opacity_functions[(0, 0)]
        
        # Make it bright red and fully opaque
        color_func.RemoveAllPoints()
        opacity_func.RemoveAllPoints()
        color_func.AddRGBPoint(0.0, 1.0, 0.0, 0.0)
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        opacity_func.AddPoint(0.0, 1.0)
        opacity_func.AddPoint(1.0, 1.0)
        
        # Debug: print the color and opacity functions
        print(f"   Color function has {color_func.GetSize()} points")
        print(f"   Opacity function has {opacity_func.GetSize()} points")
        print(f"   Color at 0.5: {color_func.GetColor(0.5)}")
        print(f"   Opacity at 0.5: {opacity_func.GetValue(0.5)}")
        
        # Turn off gradient opacity
        grad_func = self.gradient_opacity_functions.get((0, 0))
        if grad_func:
            grad_func.RemoveAllPoints()
            grad_func.AddPoint(0.0, 0.0)
            grad_func.AddPoint(1.0, 0.0)
        
        # Hide other components
        n_comps = self.volume_n_comps[0]
        for comp_idx in range(1, n_comps):
            color_other = self.color_functions.get((0, comp_idx))
            opacity_other = self.opacity_functions.get((0, comp_idx))
            if color_other and opacity_other:
                color_other.RemoveAllPoints()
                opacity_other.RemoveAllPoints()
                color_other.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
                color_other.AddRGBPoint(1.0, 0.5, 0.5, 0.5)
                opacity_other.AddPoint(0.0, 0.0)
                opacity_other.AddPoint(1.0, 0.0)
        
        # Force camera to volume bounds
        volume = self.volumes[0]
        bounds = volume.GetBounds()
        center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0] + 400, center[1] + 300, center[2] + 400)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()

           # Add a green cube at the volume center to verify camera aiming
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(center[0], center[1], center[2])
        sphere.SetRadius(10)
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(0.0, 1.0, 0.0)
        self.renderer.AddActor(sphere_actor)
        
        # Add a bright red bounding box to visualize the volume bounds
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube.GetOutputPort())
        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        cube_actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(cube_actor)
        
        # Turn off shading for this volume to avoid darkening
        self.volumes[0].GetProperty().ShadeOff()
        
        self.renderer.Render()
        print("✅ Volume forced to be bright red and fully opaque")

    def test_simple_render(self):
        print("\n🔧 SIMPLE TEST: Red volume")
        for vol_idx in range(len(self.volumes)):
            n_comps = self.volume_n_comps[vol_idx]
            for comp_idx in range(n_comps):
                color_func = self.color_functions.get((vol_idx, comp_idx))
                opacity_func = self.opacity_functions.get((vol_idx, comp_idx))
                if color_func and opacity_func:
                    color_func.RemoveAllPoints()
                    opacity_func.RemoveAllPoints()
                    color_func.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
                    color_func.AddRGBPoint(1.0, 0.5, 0.5, 0.5)
                    opacity_func.AddPoint(0.0, 0.0)
                    opacity_func.AddPoint(1.0, 0.0)
        
        color_func = self.color_functions[(0, 0)]
        opacity_func = self.opacity_functions[(0, 0)]
        color_func.RemoveAllPoints()
        opacity_func.RemoveAllPoints()
        color_func.AddRGBPoint(0.0, 1.0, 0.0, 0.0)
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        opacity_func.AddPoint(0.0, 0.5)
        opacity_func.AddPoint(1.0, 0.5)
        self.renderer.Render()
        print("✅ Simple test complete")

    def get_renderer(self):
        return self.renderer

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(400, 400, 400)
        camera.SetFocalPoint(128, 128, 47)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.renderer.Render()
    
    def debug_widget_coordinates(self, x_feature, y_feature):
        pass
    
    def fix_widget_coordinates(self, x_feature, y_feature):
        pass

    # ----- Additional debug methods (kept for reference) -----
    def debug_gradient_data(self):
        """Check the Gradient data distribution"""
        print("\n🔍 DEBUG: Gradient data analysis")
        volume = self.volumes[0]
        input_data = volume.GetMapper().GetInput()
        scalars = input_data.GetPointData().GetScalars()
        if scalars:
            data = numpy_support.vtk_to_numpy(scalars)
            if len(data.shape) == 2:
                gradient_data = data[:, 1]
            else:
                gradient_data = data
            print(f"\n📊 Gradient data statistics:")
            print(f"   Min value: {gradient_data.min():.6f}")
            print(f"   Max value: {gradient_data.max():.6f}")
            print(f"   Mean value: {gradient_data.mean():.6f}")
            print(f"   Std dev: {gradient_data.std():.6f}")
            print(f"\n   Percentiles:")
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                val = np.percentile(gradient_data, p)
                print(f"      {p}%: {val:.6f}")
                display_pos = val * 255
                print(f"         → display Y = {display_pos:.0f}")

    def debug_real_volume_data(self):
        """Debug what's actually in your real volume data"""
        print("\n🔍 DEBUG: Real volume data analysis")
        volume = self.volumes[0]
        input_data = volume.GetMapper().GetInput()
        if input_data:
            scalars = input_data.GetPointData().GetScalars()
            if scalars:
                data = numpy_support.vtk_to_numpy(scalars)
                if len(data.shape) == 2:
                    intensity_data = data[:, 0]
                else:
                    intensity_data = data
                print(f"\n📊 Intensity data statistics:")
                print(f"   Min value: {intensity_data.min():.6f}")
                print(f"   Max value: {intensity_data.max():.6f}")
                print(f"   Mean value: {intensity_data.mean():.6f}")
                print(f"   Std dev: {intensity_data.std():.6f}")
                print(f"   Non-zero count: {np.sum(intensity_data > 0.001)} out of {len(intensity_data)}")
                print(f"\n   Percentiles:")
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                    val = np.percentile(intensity_data, p)
                    print(f"      {p}%: {val:.6f}")
                dims = input_data.GetDimensions()
                print(f"\n   Volume dimensions: {dims}")
            else:
                print("   No scalars found!")
        else:
            print("   No input data in mapper!")

    # Independent test volumes (kept for reference)
    def test_force_red_independent(self):
        """Create a completely separate test volume that bypasses all widget logic"""
        print("\n🔧 TEST: Creating independent red cube volume")
        self.renderer.RemoveAllViewProps()
        dims = [200, 200, 200]
        cube_data = vtk.vtkImageData()
        cube_data.SetDimensions(dims)
        cube_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                    cube_data.SetScalarComponentFromFloat(x, y, z, 0, 0.5)
        test_mapper = vtk.vtkGPUVolumeRayCastMapper()
        test_mapper.SetInputData(cube_data)
        test_color = vtk.vtkColorTransferFunction()
        test_color.AddRGBPoint(0.0, 1.0, 0.0, 0.0)
        test_color.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        test_opacity = vtk.vtkPiecewiseFunction()
        test_opacity.AddPoint(0.0, 1.0)
        test_opacity.AddPoint(1.0, 1.0)
        test_prop = vtk.vtkVolumeProperty()
        test_prop.SetColor(test_color)
        test_prop.SetScalarOpacity(test_opacity)
        test_volume = vtk.vtkVolume()
        test_volume.SetMapper(test_mapper)
        test_volume.SetProperty(test_prop)
        self.renderer.AddVolume(test_volume)
        bounds = [0, dims[0], 0, dims[1], 0, dims[2]]
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(actor)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(300, 200, 400)
        camera.SetFocalPoint(100, 100, 100)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.renderer.Render()
        print("✅ Independent red cube created - you should see a RED cube")

    def test_red_cube_with_widget(self):
        """Create a red cube and apply the widget-based transfer function to it"""
        print("\n🔧 TEST: Red cube with widget-based TF")
        for volume in self.volumes:
            self.renderer.RemoveVolume(volume)
        dims = [200, 200, 200]
        cube_data = vtk.vtkImageData()
        cube_data.SetDimensions(dims)
        cube_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                    value = x / dims[0]
                    cube_data.SetScalarComponentFromFloat(x, y, z, 0, value)
        test_mapper = vtk.vtkGPUVolumeRayCastMapper()
        test_mapper.SetInputData(cube_data)
        test_prop = vtk.vtkVolumeProperty()
        test_prop.SetIndependentComponents(True)
        color_func = vtk.vtkColorTransferFunction()
        color_func.AddRGBPoint(0.0, 0.5, 0.5, 0.5)
        color_func.AddRGBPoint(1.0, 0.5, 0.5, 0.5)
        opacity_func = vtk.vtkPiecewiseFunction()
        opacity_func.AddPoint(0.0, 0.0)
        opacity_func.AddPoint(1.0, 0.0)
        test_prop.SetColor(color_func)
        test_prop.SetScalarOpacity(opacity_func)
        test_volume = vtk.vtkVolume()
        test_volume.SetMapper(test_mapper)
        test_volume.SetProperty(test_prop)
        self.volumes = [test_volume]
        self.volume_n_comps = [1]
        self.color_functions[(0, 0)] = color_func
        self.opacity_functions[(0, 0)] = opacity_func
        self.feature_to_volume['Intensity'] = (0, 0)
        self.feature_to_volume['Gradient'] = (0, 0)
        self.renderer.RemoveAllViewProps()
        self.renderer.AddVolume(test_volume)
        self.add_bounding_box(dims)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(300, 200, 400)
        camera.SetFocalPoint(100, 100, 100)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.renderer.Render()
        print("✅ Red cube created with gradient values (0 at left, 1 at right)")
        self.set_feature_pair('Intensity', 'Gradient')

    def test_isolated_volume(self):
        """Create a completely isolated test volume that ignores all widget logic"""
        print("\n🔧 TEST: Creating isolated test volume")
        self.renderer.RemoveAllViewProps()
        dims = [200, 200, 200]
        test_data = vtk.vtkImageData()
        test_data.SetDimensions(dims)
        test_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        for x in range(dims[0]):
            for y in range(dims[1]):
                for z in range(dims[2]):
                    value = x / dims[0]
                    test_data.SetScalarComponentFromFloat(x, y, z, 0, value)
        test_mapper = vtk.vtkGPUVolumeRayCastMapper()
        test_mapper.SetInputData(test_data)
        test_color = vtk.vtkColorTransferFunction()
        test_color.AddRGBPoint(0.0, 0.0, 0.0, 1.0)
        test_color.AddRGBPoint(0.33, 0.0, 1.0, 0.0)
        test_color.AddRGBPoint(0.66, 1.0, 1.0, 0.0)
        test_color.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        test_opacity = vtk.vtkPiecewiseFunction()
        test_opacity.AddPoint(0.0, 0.2)
        test_opacity.AddPoint(0.33, 0.5)
        test_opacity.AddPoint(0.66, 0.8)
        test_opacity.AddPoint(1.0, 0.4)
        test_prop = vtk.vtkVolumeProperty()
        test_prop.SetColor(test_color)
        test_prop.SetScalarOpacity(test_opacity)
        test_prop.ShadeOn()
        test_volume = vtk.vtkVolume()
        test_volume.SetMapper(test_mapper)
        test_volume.SetProperty(test_prop)
        self.renderer.AddVolume(test_volume)
        bounds = [0, dims[0], 0, dims[1], 0, dims[2]]
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube.GetOutputPort())
        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        cube_actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(cube_actor)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(300, 200, 400)
        camera.SetFocalPoint(100, 100, 100)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        self.renderer.Render()
        print("✅ Isolated test volume created - you should see a colorful 3D cube")

    def force_intensity_volume(self):
        """Extract and render only the Intensity component"""
        print("\n🔧 FORCING Intensity-only volume")
        
        # Get the original volume and its data
        original_volume = self.volumes[0]
        original_data = original_volume.GetMapper().GetInput()
        dims = original_data.GetDimensions()
        scalars = original_data.GetPointData().GetScalars()
        data = numpy_support.vtk_to_numpy(scalars)
        
        # Extract component 0 (Intensity)
        if len(data.shape) == 2:
            intensity_data = data[:, 0]
        else:
            intensity_data = data
        
        print(f"   Intensity data: min={intensity_data.min():.3f}, max={intensity_data.max():.3f}, mean={intensity_data.mean():.3f}")
        
        # Create a new volume with just the Intensity component
        intensity_volume = vtk.vtkImageData()
        intensity_volume.SetDimensions(dims)
        intensity_volume.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = numpy_support.numpy_to_vtk(intensity_data)
        intensity_volume.GetPointData().SetScalars(vtk_array)
        
        mapper = vtk.vtkGPUVolumeRayCastMapper()
        mapper.SetInputData(intensity_volume)
        
        # Simple red, full opacity transfer functions
        color_func = vtk.vtkColorTransferFunction()
        opacity_func = vtk.vtkPiecewiseFunction()
        color_func.AddRGBPoint(0.0, 1.0, 0.0, 0.0)
        color_func.AddRGBPoint(1.0, 1.0, 0.0, 0.0)
        opacity_func.AddPoint(0.0, 1.0)
        opacity_func.AddPoint(1.0, 1.0)
        
        prop = vtk.vtkVolumeProperty()
        prop.SetColor(color_func)
        prop.SetScalarOpacity(opacity_func)
        prop.ShadeOff()
        
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(prop)
        
        # Remove existing volumes and add this one
        for vol in self.volumes:
            self.renderer.RemoveVolume(vol)
        self.renderer.AddVolume(volume)
        
        # Set camera to volume bounds
        bounds = intensity_volume.GetBounds()
        center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(center[0] + 400, center[1] + 300, center[2] + 400)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCameraClippingRange()
        
        # Add a red bounding box
        cube = vtk.vtkCubeSource()
        cube.SetBounds(bounds)
        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputConnection(cube.GetOutputPort())
        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        cube_actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(cube_actor)
        
        self.renderer.Render()
        print("✅ Intensity-only volume rendered")

    def update_transfer_functions(self, intensities, opacities, colors, 
                            intensity_range=None, gradient_values=None,
                            gradient_opacities=None, gradient_range=None):
        """
        Update the renderer with new widget data.
        This just calls set_feature_pair to rebuild the volume with current widgets.
        """
        print(f"   🎨 NDShaderRenderer.update_transfer_functions called")
    
        # Store the current feature pair (you need to track this)
        if hasattr(self, 'current_x_feature') and hasattr(self, 'current_y_feature'):
            # Rebuild the volume with the current feature pair and new widget data
            self.set_feature_pair(self.current_x_feature, self.current_y_feature)
        else:
            print(f"   ⚠️ No current feature pair stored, can't update")