# dataset_loader.py
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
        """Open file dialog and load selected dataset. Remembers last folder."""
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
            "VTI Files (*.vti);;VTK Files (*.vtk);;VOL/RAW Files (*.vol *.raw);;MHD Files (*.mhd);;All Files (*)"
        )
        
        if not file_name:
            return None
        
        # store last dir
        try:
            with open(last_file, "w") as f:
                f.write(os.path.dirname(file_name))
        except Exception:
            pass
        
        return file_name

    def _ask_raw_settings(self, fname):
        """
        Ask the user for raw/.vol settings: dims and dtype and byte order.
        Returns tuple (dims, dtype, byte_order) or None if cancelled.
        dtype -> numpy dtype string, byte_order -> 'little'/'big'
        """
        # dims
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
        
        # dtype
        dtype_items = ["uint8", "uint16", "float32"]
        dtype_choice, ok = QtWidgets.QInputDialog.getItem(
            self.parent_window, "Raw / .vol settings", "Data type:", dtype_items, 0, False
        )
        if not ok:
            return None
        dtype = dtype_choice
        
        # byte order
        bo_items = ["little", "big"]
        bo_choice, ok = QtWidgets.QInputDialog.getItem(
            self.parent_window, "Raw / .vol settings", "Byte order:", bo_items, 0, False
        )
        if not ok:
            return None
        byte_order = bo_choice
        
        return dims, dtype, byte_order

    def _get_reader_for_file(self, file_path):
        """Get appropriate reader based on file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.vti':
            reader = vtk.vtkXMLImageDataReader()
        elif ext == '.vtk':
            # For .vtk, we need to try multiple readers
            return None  # Special handling
        elif ext == '.mhd':
            reader = vtk.vtkMetaImageReader()
        elif ext in ('.raw', '.vol'):
            reader = None  # Special handling
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return reader

    def load_volume(self, file_path):
        """
        Load volume and AUTO-DISCOVER all features.
        Returns tuple (image_data, reader, all_features)
        """
        print(f"\n📂 Loading: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        image_data = None
        reader = None
        np_scalars = None

        # --- HANDLE RAW/VOL FILES ---
        if ext in ('.raw', '.vol'):
            settings = self._ask_raw_settings(file_path)
            if settings is None:
                raise RuntimeError("Raw/.vol load cancelled or invalid settings.")
            
            dims, dtype_str, byte_order = settings
            dtype = np.dtype(dtype_str)
            
            # read file
            with open(file_path, "rb") as f:
                data = f.read()
            
            arr = np.frombuffer(data, dtype=dtype)
            expected = dims[0] * dims[1] * dims[2]
            
            if arr.size != expected:
                if dtype.itemsize > 1:
                    if byte_order == "big":
                        arr = arr.byteswap().newbyteorder()
                if arr.size != expected:
                    raise RuntimeError(f"Data size mismatch: expected {expected} elements, got {arr.size}. Check dims/type.")
            
            arr = arr.reshape(dims[::-1])  # VTK expects z-fast
            
            # create vtkImageData
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(dims[0], dims[1], dims[2])
            
            if dtype_str == "uint8":
                vtk_type = vtk.VTK_UNSIGNED_CHAR
            elif dtype_str == "uint16":
                vtk_type = vtk.VTK_UNSIGNED_SHORT
            elif dtype_str == "float32":
                vtk_type = vtk.VTK_FLOAT
            else:
                vtk_type = vtk.VTK_UNSIGNED_CHAR
            
            vtk_data.AllocateScalars(vtk_type, 1)
            flat = np.ascontiguousarray(arr.ravel(order='C'))
            vtk_arr = numpy_support.numpy_to_vtk(num_array=flat, deep=True, array_type=None)
            vtk_data.GetPointData().SetScalars(vtk_arr)
            image_data = vtk_data
            reader = None
            np_scalars = flat.astype(np.float32)

        # --- HANDLE .VTK FILES (YOUR ORIGINAL CODE) ---
        elif ext == '.vtk':
            # Try DataSetReader first
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()
            
            # Check if it worked
            if not image_data or not image_data.GetPointData().GetScalars():
                # Try StructuredPointsReader
                reader = vtk.vtkStructuredPointsReader()
                reader.SetFileName(file_path)
                reader.Update()
                image_data = reader.GetOutput()
            
            if not image_data or not image_data.GetPointData().GetScalars():
                raise RuntimeError("Failed to read .vtk file with any reader")
            
            np_scalars = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars()).astype(np.float32)

        # --- HANDLE OTHER FORMATS (.vti, .mhd) ---
        else:
            reader = self._get_reader_for_file(file_path)
            if reader is None:
                raise RuntimeError(f"Could not create reader for {ext}")
            
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()
            
            # Get primary scalars
            scalars = image_data.GetPointData().GetScalars()
            if scalars is None:
                raise ValueError("No scalar data found in file")
            
            np_scalars = numpy_support.vtk_to_numpy(scalars).astype(np.float32)

        # ===== DEBUG OUTPUT =====
        print(f"\n🔍 DATA RANGE DEBUG:")
        print(f"   Raw data min: {np_scalars.min():.1f}")
        print(f"   Raw data max: {np_scalars.max():.1f}")
        print(f"   Raw data mean: {np_scalars.mean():.1f}")
        print(f"   Raw data std: {np_scalars.std():.1f}")
        
        # Histogram to see distribution
        hist, bins = np.histogram(np_scalars, bins=20)
        print(f"   Histogram (first 10 bins):")
        for i in range(min(10, len(hist))):
            print(f"     {bins[i]:.0f}-{bins[i+1]:.0f}: {hist[i]} voxels")

        # --- STEP 1: Dictionary to hold ALL features ---
        all_features = {}
        
        # --- STEP 2: ADD PRIMARY SCALAR FIELD ---
        all_features['Intensity'] = np_scalars
        print(f"   ✅ Added primary: Intensity ({np_scalars.shape})")
        
        # --- STEP 3: AUTO-DISCOVER OTHER ARRAYS IN THE FILE ---
        if image_data:
            point_data = image_data.GetPointData()
            for i in range(point_data.GetNumberOfArrays()):
                array_name = point_data.GetArrayName(i)
                array = point_data.GetArray(i)
                
                # Skip if it's the same as scalars or already added
                if array_name and array_name not in all_features:
                    if array is not None:
                        np_array = numpy_support.vtk_to_numpy(array)
                        # Only add if it's the same size as primary
                        if len(np_array) == len(np_scalars):
                            all_features[array_name] = np_array.astype(np.float32)
                            print(f"   ✅ Auto-discovered: {array_name} ({np_array.shape})")
        
        # --- STEP 4: COMPUTE COMMON DERIVED FEATURES ---
        
        # Gradient Magnitude
        all_features['Gradient'] = self.compute_gradient(image_data, reader)
        print(f"   ✅ Computed: Gradient")
        
        # Laplacian
        all_features['Laplacian'] = self.compute_laplacian(image_data, reader)
        print(f"   ✅ Computed: Laplacian")
        
        # Local Variance (simple texture)
        all_features['Texture'] = self.compute_local_variance(np_scalars)
        print(f"   ✅ Computed: Texture")
        
        # Curvature (from gradient)
        if 'Gradient' in all_features:
            all_features['Curvature'] = self.compute_curvature(all_features['Gradient'])
            print(f"   ✅ Computed: Curvature")
        
        # Entropy
        all_features['Entropy'] = self.compute_local_entropy(np_scalars)
        print(f"   ✅ Computed: Entropy")
        
        print(f"\n📊 Total features discovered/computed: {len(all_features)}")
        for name in all_features.keys():
            print(f"   - {name}")
        
        return image_data, reader, all_features

    # --- COMPUTATION METHODS ---
    
    def compute_gradient(self, image_data, reader=None):
        """Compute gradient magnitude using VTK"""
        gradient = vtk.vtkImageGradientMagnitude()
        try:
            if reader is not None:
                gradient.SetInputConnection(reader.GetOutputPort())
            else:
                gradient.SetInputData(image_data)
            gradient.Update()
            grad_array = gradient.GetOutput().GetPointData().GetScalars()
            return numpy_support.vtk_to_numpy(grad_array).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Gradient computation failed: {e}")
            return np.zeros_like(self._get_dummy_data(image_data))
    
    def compute_laplacian(self, image_data, reader=None):
        """Compute Laplacian (second derivative)"""
        laplacian = vtk.vtkImageLaplacian()
        try:
            if reader is not None:
                laplacian.SetInputConnection(reader.GetOutputPort())
            else:
                laplacian.SetInputData(image_data)
            laplacian.SetDimensionality(3)
            laplacian.Update()
            lap_array = laplacian.GetOutput().GetPointData().GetScalars()
            return numpy_support.vtk_to_numpy(lap_array).astype(np.float32)
        except Exception as e:
            print(f"⚠️ Laplacian computation failed: {e}")
            return np.zeros_like(self._get_dummy_data(image_data))
    
    def _get_dummy_data(self, image_data):
        """Helper to get dummy data for error cases"""
        try:
            scalars = image_data.GetPointData().GetScalars()
            if scalars:
                return numpy_support.vtk_to_numpy(scalars).astype(np.float32)
        except:
            pass
        return np.zeros(100, dtype=np.float32)
    
    def compute_local_variance(self, np_scalars, window=3):
        """Compute local variance as simple texture measure"""
        try:
            from scipy import ndimage
            # Try to reshape to 3D
            size = int(round(len(np_scalars) ** (1/3)))
            if size**3 == len(np_scalars):
                volume = np_scalars.reshape((size, size, size))
                mean = ndimage.uniform_filter(volume, size=window)
                sq_mean = ndimage.uniform_filter(volume**2, size=window)
                variance = sq_mean - mean**2
                return variance.flatten()
            else:
                return np.zeros_like(np_scalars)
        except ImportError:
            print("⚠️ scipy not available, Texture feature disabled")
            return np.zeros_like(np_scalars)
        except Exception as e:
            print(f"⚠️ Texture computation failed: {e}")
            return np.zeros_like(np_scalars)
    
    def compute_curvature(self, gradient_data):
        """Compute curvature from gradient (simplified)"""
        try:
            grad_norm = gradient_data / (np.max(gradient_data) + 1e-6)
            curvature = np.gradient(grad_norm)
            return np.abs(curvature)
        except Exception as e:
            print(f"⚠️ Curvature computation failed: {e}")
            return np.zeros_like(gradient_data)
    
    def compute_local_entropy(self, np_scalars, window=3):
        """Compute local entropy"""
        try:
            from scipy import ndimage
            from scipy.stats import entropy
            
            size = int(round(len(np_scalars) ** (1/3)))
            if size**3 == len(np_scalars):
                volume = np_scalars.reshape((size, size, size))
                
                # Normalize to 0-255
                v_min, v_max = np.min(volume), np.max(volume)
                if v_max > v_min:
                    volume_norm = ((volume - v_min) / (v_max - v_min) * 255).astype(int)
                else:
                    volume_norm = volume.astype(int)
                
                def local_entropy_func(data):
                    hist = np.histogram(data, bins=32, range=(0, 255))[0]
                    hist = hist[hist > 0]
                    if len(hist) > 1:
                        return entropy(hist)
                    return 0
                
                entropy_map = ndimage.generic_filter(
                    volume_norm, local_entropy_func, size=window
                )
                return entropy_map.flatten()
            else:
                return np.zeros_like(np_scalars)
        except ImportError:
            print("⚠️ scipy not available, Entropy feature disabled")
            return np.zeros_like(np_scalars)
        except Exception as e:
            print(f"⚠️ Entropy computation failed: {e}")
            return np.zeros_like(np_scalars)
    
    def normalize_data(self, np_scalars, np_gradient):
        """Normalize scalar and gradient data to 0-255 range."""
        raw_int_min, raw_int_max = np_scalars.min(), np_scalars.max()
        intensity_range = (raw_int_min, raw_int_max)
        
        if raw_int_max - raw_int_min == 0:
            normalized_scalars = np.zeros_like(np_scalars)
        else:
            normalized_scalars = 255.0 * (np_scalars - raw_int_min) / (raw_int_max - raw_int_min)

        raw_grad_min, raw_grad_max = np_gradient.min(), np_gradient.max()
        gradient_range = (raw_grad_min, raw_grad_max)
        
        if raw_grad_max - raw_grad_min == 0:
            gradient_normalized = np.zeros_like(np_gradient)
        else:
            gradient_normalized = 255.0 * (np_gradient - raw_grad_min) / (raw_grad_max - raw_grad_min)

        return normalized_scalars, gradient_normalized, intensity_range, gradient_range

    def normalize_single(self, data_array):
        """Normalize a single data array to 0-255 range"""
        data_min = np.min(data_array)
        data_max = np.max(data_array)
    
        if data_max > data_min:
            normalized = 255.0 * (data_array - data_min) / (data_max - data_min)
            return normalized.astype(np.float32)
        return np.zeros_like(data_array, dtype=np.float32)