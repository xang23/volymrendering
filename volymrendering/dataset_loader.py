import numpy as np
import vtk
from vtk.util import numpy_support
from PyQt5 import QtWidgets
import os


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
            "VTI Files (*.vti);;VOL/RAW Files (*.vol *.raw);;MHD Files (*.mhd);;All Files (*)"
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

    def load_volume(self, file_path):
        """
        Load .vti, .mhd, .raw, .vol (raw) datasets.
        Returns tuple (image_data, reader, np_scalars, np_gradient) or raises exception.
        """
        ext = os.path.splitext(file_path)[1].lower()
        image_data = None
        reader = None

        if ext == ".vti":
            reader = vtk.vtkXMLImageDataReader()
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()

        elif ext == ".mhd":
            reader = vtk.vtkMetaImageReader()
            reader.SetFileName(file_path)
            reader.Update()
            image_data = reader.GetOutput()

        elif ext in (".raw", ".vol"):
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

        else:
            raise RuntimeError(f"Unsupported extension: {ext}")

        # Extract scalars
        try:
            np_scalars = numpy_support.vtk_to_numpy(image_data.GetPointData().GetScalars()).astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to extract scalars: {e}")

        # Compute gradient
        grad_filter = vtk.vtkImageGradientMagnitude()
        try:
            if reader is not None:
                grad_filter.SetInputConnection(reader.GetOutputPort())
            else:
                grad_filter.SetInputData(image_data)
            grad_filter.Update()
            np_gradient = numpy_support.vtk_to_numpy(grad_filter.GetOutput().GetPointData().GetScalars()).astype(np.float32)
        except Exception:
            np_gradient = np.zeros_like(np_scalars, dtype=np.float32)

        return image_data, reader, np_scalars, np_gradient

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