# nd_popup.py
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from unified_tf_canvas import UnifiedTFCanvas
from tf_canvas_widget import TFCanvasWidget
from widget_manager_ui import WidgetManager
from volume_renderer import VolumeRenderer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np

class NDFeaturePopup(QtWidgets.QMainWindow):
    """Popup window with big TF view, widget manager, and live render"""
    
    def __init__(self, feat_x, feat_y, data_x, data_y, nd_manager, parent=None):
        super().__init__(parent)
        
        self.feat_x = feat_x
        self.feat_y = feat_y
        self.nd_manager = nd_manager
        
        self.setWindowTitle(f"nD Explorer: {feat_x} vs {feat_y}")
        self.setGeometry(200, 200, 1400, 800)
        
        # Window flags
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowCloseButtonHint |
            Qt.WindowMinimizeButtonHint
        )
        
        # Store ranges
        self.x_range = (float(np.min(data_x)), float(np.max(data_x)))
        self.y_range = (float(np.min(data_y)), float(np.max(data_y)))
        
        # Find main app for volume data
        self.main_app = self.find_main_app()

        # Setup UI
        self.setup_ui(data_x, data_y)
        
        # Load widgets
        self.load_projected_widgets()
        
        # Initial render update
        self.update_render_view()

    def find_main_app(self):
        """Find the main VolumeApp instance"""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'volume_renderer_widget'):
                return parent
            parent = parent.parent()
        return None

    def normalize_to_255(self, data):
        """Normalize data to 0-255 range for display"""
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            return 255.0 * (data - data_min) / (data_max - data_min)
        return np.zeros_like(data)
    
    def setup_ui(self, data_x, data_y):
        """Setup the user interface"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        # Main horizontal layout - THREE columns
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # ===== LEFT: Big TF Canvas =====
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        
        # Title
        title = QtWidgets.QLabel(f"<h2>{self.feat_x} vs {self.feat_y}</h2>")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)
        
        # Info
        info = (f"Raw: {self.feat_x}=[{self.x_range[0]:.1f}, {self.x_range[1]:.1f}] | "
                f"{self.feat_y}=[{self.y_range[0]:.1f}, {self.y_range[1]:.1f}]")
        info_label = QtWidgets.QLabel(info)
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(info_label)
        
        # Normalize data for display
        norm_data_x = self.normalize_to_255(data_x)
        norm_data_y = self.normalize_to_255(data_y)
        
        # Create BIG TF canvas
        self.tf_canvas = UnifiedTFCanvas(
            tf_type='2d',
            data=norm_data_x,
            gradient_data=norm_data_y,
        )
        
        # Store RAW ranges for reference
        self.tf_canvas.raw_intensity_range = self.x_range
        self.tf_canvas.raw_gradient_range = self.y_range
        self.tf_canvas.intensity_range = (0, 255)
        self.tf_canvas.gradient_range = (0, 255)
        
        # CRITICAL: Set projection features and callback
        self.tf_canvas.set_projection_features(self.feat_x, self.feat_y)
        self.tf_canvas.set_nd_callback(self.on_widget_moved)
        
        # Add canvas wrapper
        canvas_wrapper = TFCanvasWidget(self.tf_canvas, self, label='Reset View')
        left_layout.addWidget(canvas_wrapper)
        
        main_layout.addWidget(left_widget, 2)  # 2/5 of space
        
        # ===== MIDDLE: Widget Manager =====
        middle_widget = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_widget)
        
        manager_title = QtWidgets.QLabel("<h3>Widget Manager</h3>")
        manager_title.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(manager_title)
        
        # Create widget manager for this popup
        self.widget_manager = WidgetManager(self.tf_canvas)
        middle_layout.addWidget(self.widget_manager)
        
        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        middle_layout.addWidget(close_btn)
        
        middle_layout.addStretch()
        main_layout.addWidget(middle_widget, 1)  # 1/5 of space
        
        # ===== RIGHT: Volume Render View =====
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        
        render_title = QtWidgets.QLabel("<h3>Live Rendering</h3>")
        render_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(render_title)
        
        # Create VTK renderer for this popup
        self.renderer = VolumeRenderer(f"popup_{self.feat_x}_{self.feat_y}")
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer.get_renderer())
        
        # Copy volume data from main app
        if self.main_app and hasattr(self.main_app, 'image_data'):
            self.renderer.set_volume_data(self.main_app.image_data, self.main_app.reader)
            print(f"✅ Volume data copied to popup renderer")
        
        right_layout.addWidget(self.vtk_widget)
        main_layout.addWidget(right_widget, 2)  # 2/5 of space
        
        # Initialize renderer
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
    
    def load_projected_widgets(self):
        """Load widgets projected to this plane"""
        self.tf_canvas.clear_widgets()
        
        # Get projected widgets from nd_manager
        projected = self.nd_manager.project_to_2d(self.feat_x, self.feat_y)
        
        # DEBUG: Check if widgets have nd_ref
        for i, widget in enumerate(projected):
            print(f"🔍 Widget {i}: has nd_ref? {hasattr(widget, 'nd_ref')}")
            if hasattr(widget, 'nd_ref'):
                print(f"   nd_ref type: {type(widget.nd_ref)}")
            self.tf_canvas.add_widget(widget)
        
        # Update widget manager
        if hasattr(self, 'widget_manager'):
            self.widget_manager.update_widget_list()
        
        print(f"📊 Loaded {len(projected)} widgets into popup")
        self.update_render_view()
    
    def on_widget_moved(self, widget_2d, feat_x, feat_y, new_x, new_y):
        """When widget moves in popup, update nd_manager and render"""
        print(f"🟢 WIDGET MOVED CALLED: ({new_x:.1f}, {new_y:.1f})")
        
        # Update nd_manager
        self.nd_manager.update_nd_position(widget_2d, new_x, new_y)
        print(f"   Updated nd_manager")
        
        # Update render
        self.update_render_view()
    
    def update_render_view(self):
        """Update the volume rendering with current TF"""
        print(f"🔵 UPDATE_RENDER_VIEW called")
        
        if not hasattr(self, 'renderer'):
            print(f"   ❌ No renderer")
            return
        if not self.main_app:
            print(f"   ❌ No main app")
            return
        
        print(f"   Getting samples from TF canvas...")
        samples = self.tf_canvas.sample_for_vtk()
        print(f"   Got {len(samples) if samples else 0} samples")
        
        if samples:
            intensities = [s[0] for s in samples]
            opacities = [s[1] for s in samples]
            colors = [s[2] for s in samples]
            
            print(f"   Sample range: intensities {min(intensities)}-{max(intensities)}")
            
            # Scale intensities to actual data range
            int_min, int_max = self.x_range
            scaled_intensities = [int_min + (i/255.0)*(int_max - int_min) for i in intensities]
            
            print(f"   Scaled range: {min(scaled_intensities):.1f} - {max(scaled_intensities):.1f}")
            print(f"   Calling renderer.update_transfer_functions...")
            
            # Update renderer
            self.renderer.update_transfer_functions(
                scaled_intensities, opacities, colors,
                (int_min, int_max)
            )
            
            print(f"   Rendering...")
            self.vtk_widget.GetRenderWindow().Render()
            print(f"✅ Render updated")
        else:
            print(f"   ⚠️ No samples from TF canvas")
    
    def closeEvent(self, event):
        """Clean up when closing"""
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.close()
        event.accept()