import vtk
from vtk.util import numpy_support
import numpy as np
import os
from PyQt5 import QtWidgets

class DatasetLoader:
    def __init__(self, parent_window=None):
        self.parent_window = parent_window
        self.LAST_DIR_FILE = ".last_open_dir"

    def load_volume_dialog(self):
        start_dir = ""
        last_file = os.path.join(os.path.dirname(__file__), self.LAST_DIR_FILE)
        if os.path.exists(last_file):
            try:
                with open(last_file, "r") as f:
                    start_dir = f.read().strip()
            except Exception:
                start_dir = ""
        
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.parent_window, "Open Volume Dataset", start_dir or "",
            "VTI Files (*.vti);;VTK Files (*.vtk);;VOL/RAW/IVF Files (*.vol *.raw *.ivf);;MHD Files (*.mhd);;All Files (*)"
        )
        
        if not file_name:
            return None
        
        try:
            with open(last_file, "w") as f:
                f.write(os.path.dirname(file_name))
        except Exception:
            pass
        
        return file_name

    def _ask_raw_settings(self, fname):
        dims_text, ok = QtWidgets.QInputDialog.getText(
            self.parent_window, "Raw / .vol settings",
            "Enter dimensions as width,height,depth (e.g. 256,256,113):"
        )
        if not ok or not dims_text:
            return None
        try:
            parts = [int(p.strip()) for p in dims_text.split(",")]
            if len(parts) != 3:
                raise ValueError("Expected three integers")
            dims = tuple(parts)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self.parent_window, "Invalid input", f"Invalid dimensions: {e}")
            return None
        
        dtype_items = ["uint8", "uint16", "float32"]
        dtype_choice, ok = QtWidgets.QInputDialog.getItem(
            self.parent_window, "Raw / .vol settings", "Data type:", dtype_items, 0, False
        )
        if not ok:
            return None
        dtype = dtype_choice
        
        bo_items = ["little", "big"]
        bo_choice, ok = QtWidgets.QInputDialog.getItem(
            self.parent_window, "Raw / .vol settings", "Byte order:", bo_items, 0, False
        )
        if not ok:
            return None
        byte_order = bo_choice
        
        return dims, dtype, byte_order

    def _get_reader_for_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.vti':
            reader = vtk.vtkXMLImageDataReader()
        elif ext == '.vtk':
            return None
        elif ext == '.mhd':
            reader = vtk.vtkMetaImageReader()
        elif ext in ('.raw', '.vol', '.ivf'):
            reader = None
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return reader

    def _load_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        image_data = None
        reader = None
        np_scalars = None

        if ext in ('.raw', '.vol', '.ivf'):
            settings = self._ask_raw_settings(file_path)
            if settings is None:
                raise RuntimeError("Raw/.vol load cancelled or invalid settings.")
            
            dims, dtype_str, byte_order = settings
            dtype = np.dtype(dtype_str)
            
            with open(file_path, "rb") as f:
                data = f.read()
            
            arr = np.frombuffer(data, dtype=dtype)
            expected = dims[0] * dims[1] * dims[2]
            
            if arr.size != expected:
                if dtype.itemsize > 1 and byte_order == "big":
                    arr = arr.byteswap().newbyteorder()
                if arr.size != expected:
                    raise RuntimeError(f"Data size mismatch")
            
            arr = arr.reshape(dims[::-1])
            
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(dims[0], dims[1], dims[2])
            vtk_data.SetSpacing(1.0, 1.0, 1.0)
            
            vtk_type = {
                'uint8': vtk.VTK_UNSIGNED_CHAR,
                'uint16': vtk.VTK_UNSIGNED_SHORT,
                'float32': vtk.VTK_FLOAT
            }.get(dtype_str, vtk.VTK_UNSIGNED_CHAR)
            
            vtk_data.AllocateScalars(vtk_type, 1)
            flat = np.ascontiguousarray(arr.ravel(order='C'))
            vtk_arr = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type = vtk_type)
            vtk_data.GetPointData().SetScalars(vtk_arr)
            image_data = vtk_data
            reader = None
            
            vtk_scalars = vtk_data.GetPointData().GetScalars()
            np_scalars = numpy_support.vtk_to_numpy(vtk_scalars).astype(np.float32)


        elif ext == '.vtk':
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()
            
            if not image_data or not image_data.GetPointData().GetScalars():
                reader = vtk.vtkStructuredPointsReader()
                reader.SetFileName(file_path)
                reader.Update()
                image_data = reader.GetOutput()
            
            if not image_data or not image_data.GetPointData().GetScalars():
                raise RuntimeError("Failed to read .vtk file")
            
            np_scalars = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars()).astype(np.float32)

        else:
            reader = self._get_reader_for_file(file_path)
            if reader is None:
                raise RuntimeError(f"Could not create reader for {ext}")
            
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()
            
            scalars = image_data.GetPointData().GetScalars()
            if scalars is None:
                raise ValueError("No scalar data found in file")
            
            np_scalars = numpy_support.vtk_to_numpy(scalars).astype(np.float32)

        return image_data, reader, np_scalars

    def compute_all_features(self, image_data, np_scalars, reader):
        all_features = {}
    
        # Get dimensions from the image data
        dims = image_data.GetDimensions()
        width, height, depth = dims
        print(f"Volume dimensions: {dims} (width={width}, height={height}, depth={depth})")
        
        # Get point data
        point_data = image_data.GetPointData()
    
        # Add Intensity (original scalars)
        all_features['Intensity'] = np_scalars
        print(f"Added primary: Intensity ({np_scalars.shape})")
    
        # Auto-discover existing arrays
        if image_data:
            for i in range(point_data.GetNumberOfArrays()):
                array_name = point_data.GetArrayName(i)
                array = point_data.GetArray(i)
            
                if array_name and array_name not in all_features and array is not None:
                    np_array = numpy_support.vtk_to_numpy(array)
                    if len(np_array) == len(np_scalars):
                        all_features[array_name] = np_array.astype(np.float32)
                        print(f"Auto-discovered: {array_name}")
    
        # Compute and add Gradient
        
        width, height, depth = image_data.GetDimensions()
        volume = np_scalars.reshape((depth, height, width))

        # compute gradient in NumPy (aligned with your scalar data)
        gx = np.gradient(volume, axis=2)
        gy = np.gradient(volume, axis=1)
        gz = np.gradient(volume, axis=0)

        gradient = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32).flatten()

        all_features['Gradient'] = gradient
        gradient_vtk = numpy_support.numpy_to_vtk(gradient)
        gradient_vtk.SetName('Gradient')
        point_data.AddArray(gradient_vtk)
        print(f"Computed and added: Gradient (range: [{gradient.min():.1f}, {gradient.max():.1f}])")
    
        # Compute and add Laplacian
        laplacian = self.compute_laplacian(image_data, reader)
        all_features['Laplacian'] = laplacian
        laplacian_vtk = numpy_support.numpy_to_vtk(laplacian)
        laplacian_vtk.SetName('Laplacian')
        point_data.AddArray(laplacian_vtk)
        print(f"Computed and added: Laplacian (range: [{laplacian.min():.1f}, {laplacian.max():.1f}])")
    
        # Compute and add Texture (using actual dimensions, but with downsampling for performance)
        print("Computing Texture (this may take a moment)...")
        texture = self.compute_local_variance(np_scalars, dims, window=3)
        all_features['Texture'] = texture
        texture_vtk = numpy_support.numpy_to_vtk(texture)
        texture_vtk.SetName('Texture')
        point_data.AddArray(texture_vtk)
        print(f"Computed and added: Texture (range: [{texture.min():.3f}, {texture.max():.3f}])")
    
        # Compute and add Curvature
        curvature = self.compute_curvature(gradient, dims)
        all_features['Curvature'] = curvature
        curvature_vtk = numpy_support.numpy_to_vtk(curvature)
        curvature_vtk.SetName('Curvature')
        point_data.AddArray(curvature_vtk)
        print(f"Computed and added: Curvature (range: [{curvature.min():.3f}, {curvature.max():.3f}])")
    
        # TEMPORARILY DISABLE ENTROPY - too slow for 6 million voxels
        # Create dummy entropy array (zeros) for now
        """print("Note: Entropy computation disabled for performance (using zeros)")
        entropy = np.zeros_like(np_scalars)
        all_features['Entropy'] = entropy
        entropy_vtk = numpy_support.numpy_to_vtk(entropy)
        entropy_vtk.SetName('Entropy')
        point_data.AddArray(entropy_vtk)
        print(f"Computed and added: Entropy (range: [{entropy.min():.3f}, {entropy.max():.3f}])")"""
    
        return all_features

    def load_volume(self, file_path):
        print(f"\nLoading: {file_path}")
    
        image_data, reader, np_scalars = self._load_file(file_path)
    
        # Rename the primary scalars to 'Intensity'
        scalars = image_data.GetPointData().GetScalars()
        if scalars:
            scalars.SetName('Intensity')
            print(f"Renamed primary scalars to 'Intensity'")
    
        all_features = self.compute_all_features(image_data, np_scalars, reader)
    
        print(f"\nTotal features: {len(all_features)}")
        return image_data, reader, all_features

    def compute_gradient(self, image_data, reader=None):
        """Compute gradient magnitude from image data"""
        try:
            gradient = vtk.vtkImageGradientMagnitude()
            
            if reader is not None:
                gradient.SetInputConnection(reader.GetOutputPort())
            else:
                gradient.SetInputData(image_data)
            
            gradient.Update()
            grad_array = gradient.GetOutput().GetPointData().GetScalars()
            
            if grad_array is None:
                print("Warning: Gradient computation produced no output")
                return np.zeros_like(self._get_dummy_data(image_data))
            
            return numpy_support.vtk_to_numpy(grad_array).astype(np.float32)
            
        except Exception as e:
            print(f"Error computing gradient: {e}")
            return np.zeros_like(self._get_dummy_data(image_data))
    
    def compute_laplacian(self, image_data, reader=None):
        """Compute Laplacian from image data"""
        try:
            laplacian = vtk.vtkImageLaplacian()
            
            if reader is not None:
                laplacian.SetInputConnection(reader.GetOutputPort())
            else:
                laplacian.SetInputData(image_data)
            
            laplacian.SetDimensionality(3)
            laplacian.Update()
            lap_array = laplacian.GetOutput().GetPointData().GetScalars()
            
            if lap_array is None:
                print("Warning: Laplacian computation produced no output")
                return np.zeros_like(self._get_dummy_data(image_data))
            
            return numpy_support.vtk_to_numpy(lap_array).astype(np.float32)
            
        except Exception as e:
            print(f"Error computing laplacian: {e}")
            return np.zeros_like(self._get_dummy_data(image_data))
    
    def _get_dummy_data(self, image_data):
        try:
            scalars = image_data.GetPointData().GetScalars()
            if scalars:
                return numpy_support.vtk_to_numpy(scalars).astype(np.float32)
        except:
            pass
        return np.zeros(100, dtype=np.float32)
    
    def compute_local_variance(self, np_scalars, dims, window=3):
        """Compute local variance (texture) using actual volume dimensions"""
        try:
            from scipy import ndimage
            
            width, height, depth = dims
            expected_voxels = width * height * depth
            
            if len(np_scalars) != expected_voxels:
                print(f"Warning: Data size mismatch for texture. Expected {expected_voxels}, got {len(np_scalars)}")
                return np.zeros_like(np_scalars)
            
            # Reshape to 3D volume (VTK order: z, y, x)
            volume = np_scalars.reshape((depth, height, width))
            
            # Compute local variance
            mean = ndimage.uniform_filter(volume, size=window)
            sq_mean = ndimage.uniform_filter(volume**2, size=window)
            variance = sq_mean - mean**2
            
            # Flatten back to 1D
            return variance.flatten()
            
        except Exception as e:
            print(f"Error computing local variance: {e}")
            return np.zeros_like(np_scalars)
    
    def compute_curvature(self, gradient_data, dims):
        """Compute curvature from gradient data (approximation of second derivative)"""
        try:
            width, height, depth = dims
            expected_voxels = width * height * depth
            
            if len(gradient_data) != expected_voxels:
                print(f"Warning: Data size mismatch for curvature. Expected {expected_voxels}, got {len(gradient_data)}")
                return np.zeros_like(gradient_data)
            
            # Reshape to 3D
            grad_volume = gradient_data.reshape((depth, height, width))
            
            # Normalize gradient
            grad_max = np.max(grad_volume)
            if grad_max > 1e-6:
                grad_norm = grad_volume / grad_max
            else:
                grad_norm = grad_volume
            
            # Compute gradient of the gradient (curvature approximation)
            grad_x = np.gradient(grad_norm, axis=2)
            grad_y = np.gradient(grad_norm, axis=1)
            grad_z = np.gradient(grad_norm, axis=0)
            
            curvature = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            
            return curvature.flatten()
            
        except Exception as e:
            print(f"Error computing curvature: {e}")
            return np.zeros_like(gradient_data)
    
    def compute_local_entropy(self, np_scalars, dims, window=3):
        """Compute local entropy - DISABLED for performance"""
        # This function is kept but not called to avoid performance issues
        print("Warning: compute_local_entropy is disabled for performance")
        return np.zeros_like(np_scalars)
    
    def normalize_data(self, np_scalars, np_gradient):

        print("\n===== NORMALIZATION DEBUG =====")

        print("Intensity:")
        print("  min/max:", np_scalars.min(), np_scalars.max())
        print(
            "  percentiles:",
            np.percentile(np_scalars, [0, 1, 5, 50, 95, 99, 99.5, 100])
        )

        print("\nGradient:")
        print("  min/max:", np_gradient.min(), np_gradient.max())
        print(
            "  percentiles:",
            np.percentile(np_gradient, [0, 1, 5, 50, 95, 99, 99.5, 100])
        )

        raw_int_min, raw_int_max = np_scalars.min(), np_scalars.max()
        intensity_range = (raw_int_min, raw_int_max)

        if raw_int_max - raw_int_min == 0:
            normalized_scalars = np.zeros_like(np_scalars)
        else:
            normalized_scalars = (
                255.0 * (np_scalars - raw_int_min)
                / (raw_int_max - raw_int_min)
            )

        raw_grad_min, raw_grad_max = np_gradient.min(), np_gradient.max()
        gradient_range = (raw_grad_min, raw_grad_max)

        if raw_grad_max - raw_grad_min == 0:
            gradient_normalized = np.zeros_like(np_gradient)
        else:
            gradient_normalized = (
                255.0 * (np_gradient - raw_grad_min)
                / (raw_grad_max - raw_grad_min)
            )

        print("\nNormalized Gradient:")
        print(
            "  percentiles:",
            np.percentile(
                gradient_normalized,
                [0, 1, 5, 50, 95, 99, 99.5, 100]
            )
        )

        print("================================\n")

        return (
            normalized_scalars,
            gradient_normalized,
            intensity_range,
            gradient_range,
        )

    def normalize_single(self, data_array):
        data_min, data_max = np.min(data_array), np.max(data_array)
        if data_max > data_min:
            norm = 255.0 * (data_array - data_min) / (data_max - data_min)
            return np.clip(norm, 0, 255).astype(np.float32)
        return np.zeros_like(data_array, dtype=np.float32)

    def create_multicomponent_volume(self, image_data, all_features):
        dims = image_data.GetDimensions()
        n_voxels = dims[0] * dims[1] * dims[2]
        n_features = len(all_features)
    
        print(f"Creating multi-component volume: {n_features} features, {n_voxels} voxels")
    
        multi_array = np.zeros((n_voxels, n_features), dtype=np.float32)
    
        for i, (name, data) in enumerate(all_features.items()):
            if len(data) == n_voxels:
                multi_array[:, i] = data
                print(f"Component {i}: {name}")
            else:
                print(f"Warning: {name} size mismatch")
    
        vtk_array = numpy_support.numpy_to_vtk(multi_array)
        volume_data = vtk.vtkImageData()
        volume_data.SetDimensions(dims)
        volume_data.GetPointData().SetScalars(vtk_array)
    
        return volume_data, list(all_features.keys())