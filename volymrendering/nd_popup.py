import traceback
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from unified_tf_canvas import UnifiedTFCanvas
from tf_canvas_widget import TFCanvasWidget
from widget_manager_ui import WidgetManager
from nd_shader_renderer import NDShaderRenderer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import time

class NDFeaturePopup(QtWidgets.QMainWindow):
    
    def __init__(self, feat_x, feat_y, data_x, data_y, nd_manager, 
                 image_data, all_features, parent=None):
        super().__init__(parent)
        
        self.feat_x = feat_x
        self.feat_y = feat_y
        self.nd_manager = nd_manager
        self.image_data = image_data
        self.all_features = all_features
        self.feature_names = list(all_features.keys())

        point_data = image_data.GetPointData()
        self.feature_names = []
        for i in range(point_data.GetNumberOfArrays()):
            name = point_data.GetArrayName(i)
            if name:
                self.feature_names.append(name)
    
        print(f"📊 Actual features in point data: {self.feature_names}")
        
        self.setWindowTitle(f"nD Explorer: {feat_x} vs {feat_y}")
        self.setGeometry(200, 200, 1400, 800)
        
        self.setWindowFlags(
            Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        )
        
        self.x_range = (float(np.min(data_x)), float(np.max(data_x)))
        self.y_range = (float(np.min(data_y)), float(np.max(data_y)))

        self.setup_ui(data_x, data_y)
        self.load_projected_widgets()
        self.update_render_view()

    def normalize_to_255(self, data):
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            return 255.0 * (data - data_min) / (data_max - data_min)
        return np.zeros_like(data)
    
    def setup_ui(self, data_x, data_y):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # LEFT: Big TF Canvas
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        title = QtWidgets.QLabel(f"<h2>{self.feat_x} vs {self.feat_y}</h2>")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        info = (f"Raw: {self.feat_x}=[{self.x_range[0]:.1f}, {self.x_range[1]:.1f}] | "
                f"{self.feat_y}=[{self.y_range[0]:.1f}, {self.y_range[1]:.1f}]")
        info_label = QtWidgets.QLabel(info)
        info_label.setStyleSheet("color: #666; font-size: 10px;")
        info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(info_label)

        norm_data_x = self.normalize_to_255(data_x)
        norm_data_y = self.normalize_to_255(data_y)

        self.tf_canvas = UnifiedTFCanvas(
            tf_type='2d',
            data=norm_data_x,
            gradient_data=norm_data_y,
            x_label=self.feat_x,      # Add this
            y_label=self.feat_y       # Add this
        )

        # Set the axis labels for the canvas
        
        self.tf_canvas.raw_intensity_range = self.x_range
        self.tf_canvas.raw_gradient_range = self.y_range
        self.tf_canvas.intensity_range = (0, 255)
        self.tf_canvas.gradient_range = (0, 255)

        self.tf_canvas.set_projection_features(self.feat_x, self.feat_y)
        self.tf_canvas.set_nd_callback(self.on_widget_moved)


        canvas_wrapper = TFCanvasWidget(self.tf_canvas, self, label='Reset View')
        left_layout.addWidget(canvas_wrapper)

        main_layout.addWidget(left_widget, 2)

        # MIDDLE: Widget Manager
        middle_widget = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_widget)

        manager_title = QtWidgets.QLabel("<h3>Widget Manager</h3>")
        manager_title.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(manager_title)

        self.widget_manager = WidgetManager(self.tf_canvas)
        middle_layout.addWidget(self.widget_manager)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        middle_layout.addWidget(close_btn)

        intensity_only_btn = QtWidgets.QPushButton("Test: Intensity Only")
        intensity_only_btn.clicked.connect(self.test_intensity_only)
        middle_layout.addWidget(intensity_only_btn)
        
        # Test buttons
        indep_btn = QtWidgets.QPushButton("Test: Red Cube")
        indep_btn.clicked.connect(self.test_red_cube)
        middle_layout.addWidget(indep_btn)

        cube_widget_btn = QtWidgets.QPushButton("Test: Cube + Widget")
        cube_widget_btn.clicked.connect(self.test_cube_with_widget)
        middle_layout.addWidget(cube_widget_btn)

        isolated_btn = QtWidgets.QPushButton("Test: Isolated Volume")
        isolated_btn.clicked.connect(self.test_isolated)
        middle_layout.addWidget(isolated_btn)

        # Force real volume red
        force_btn = QtWidgets.QPushButton("Force Real Volume Red")
        force_btn.clicked.connect(self.force_real_red)
        middle_layout.addWidget(force_btn)

        middle_layout.addStretch()
        main_layout.addWidget(middle_widget, 1)

        # RIGHT: ND Shader Render View
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        render_title = QtWidgets.QLabel(f"<h3>Live Rendering: {self.feat_x} (color) vs {self.feat_y} (opacity)</h3>")
        render_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(render_title)

        self.vtk_widget = QVTKRenderWindowInteractor()

        # CREATE RENDERER
        try:
            print(f"\n🔧 Creating NDShaderRenderer for popup...")
            self.mc_renderer = NDShaderRenderer(
                self.image_data,
                self.feature_names,
                self.nd_manager,
                f"popup_{self.feat_x}_{self.feat_y}"
            )
            print(f"✅ NDShaderRenderer created successfully")
            
            print(f"🎯 Adding renderer to VTK widget...")
            ren_win = self.vtk_widget.GetRenderWindow()
            ren_win.AddRenderer(self.mc_renderer.get_renderer())
            print(f"✅ Renderer added to VTK widget")
            
            right_layout.addWidget(self.vtk_widget)
            print(f"✅ VTK widget added to layout")
            
            self.vtk_widget.show()
            self.vtk_widget.Initialize()

            # Test simple render first
            print("\n🔧 Running simple test render...")
            self.mc_renderer.test_simple_render()
            ren_win.Render()
            print("✅ Simple test render complete")
            
            time.sleep(1)
            
            # Apply widget-based transfer function
            print(f"\n🎯 Applying widget-based TF for {self.feat_x} vs {self.feat_y}...")
            self.mc_renderer.set_feature_pair(self.feat_x, self.feat_y)
            ren_win.Render()
            print(f"✅ Widget-based render complete")
            
        except Exception as e:
            print(f"❌ CRASH in renderer creation: {e}")
            traceback.print_exc()
            self.mc_renderer = None
            error_label = QtWidgets.QLabel(f"Render error: {str(e)}")
            error_label.setStyleSheet("color: red; font-size: 14px;")
            right_layout.addWidget(error_label)
            right_layout.addWidget(self.vtk_widget)
            self.vtk_widget.show()

        main_layout.addWidget(right_widget, 2)

    def force_real_red(self):
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.force_volume_visible()

    def load_projected_widgets(self):
        self.tf_canvas.clear_widgets()
        
        projected = self.nd_manager.project_to_2d(self.feat_x, self.feat_y)
        
        for widget in projected:
            self.tf_canvas.add_widget(widget)
        
        if hasattr(self, 'widget_manager'):
            self.widget_manager.update_widget_list()
        
        print(f"Loaded {len(projected)} widgets into popup")
        self.update_render_view()
    
    def on_widget_moved(self, widget_2d, feat_x, feat_y, new_x, new_y):
        print(f"Widget moved: ({new_x:.1f}, {new_y:.1f})")
    
        # Update the nD manager with the new position
        self.nd_manager.update_nd_position(widget_2d, new_x, new_y)
    
        # DEBUG: Print current widgets
        self.nd_manager.debug_widgets()
    
        # Update the widget manager UI if needed
        if hasattr(self, 'widget_manager'):
            self.widget_manager.update_widget_list()
    
        # Refresh the render
        self.update_render_view()
    
    def update_render_view(self):
        print(f"Updating render view for {self.feat_x} vs {self.feat_y}")

        if not hasattr(self, 'mc_renderer') or self.mc_renderer is None:
            print(f"   No renderer available")
            return

        try:
            # Get ALL widgets from the nd_manager (master state)
            all_widgets = self.nd_manager.widgets
            print(f"   Got {len(all_widgets)} widgets from nd_manager")
        
            # Build transfer function arrays for THIS projection
            intensities = []      # Values for feat_x
            opacities = []        # Opacity at each widget
            colors = []           # RGB color for each widget
            gradient_values = []  # Values for feat_y (if 2D TF)
            gradient_opacities = [] # Opacities for feat_y
        
            for widget in all_widgets:
                # Get widget's position in current feature pair projection
                x_val = widget.nd_coords.get(self.feat_x, 128)  # Display coords 0-255
                y_val = widget.nd_coords.get(self.feat_y, 128)  # Display coords 0-255
            
                intensities.append(x_val)
                opacities.append(widget.opacity)
                colors.append(widget.color)
            
                # For 2D transfer function, use second feature as gradient
                gradient_values.append(y_val)
                gradient_opacities.append(widget.opacity)
        
            print(f"   Built TF with {len(intensities)} widgets")
            if intensities:
                print(f"   Sample widget: intensity={intensities[0]:.1f}, opacity={opacities[0]:.2f}, color={colors[0]}")
        
            # Update the renderer with the current widget state
            if intensities:
                self.mc_renderer.update_transfer_functions(
                    intensities,           # Values for feat_x
                    opacities,            # Opacity at each widget
                    colors,               # Color for each widget
                    self.x_range,         # Raw intensity range for feat_x
                    gradient_values,      # Values for feat_y
                    gradient_opacities,   # Opacities for feat_y
                    self.y_range         # Raw gradient range for feat_y
                )
            else:
                # No widgets - use default gray volume
                self.mc_renderer.update_transfer_functions(
                    [0, 255], [0, 0], [(0.5, 0.5, 0.5)],
                    self.x_range
                )
        
            # Set the feature pair for any additional configuration
            self.mc_renderer.set_feature_pair(self.feat_x, self.feat_y)
        
            # Force render
            self.vtk_widget.GetRenderWindow().Render()
            print(f"   ✅ Render updated with {len(intensities)} active widgets")
    
        except Exception as e:
            print(f"   ❌ Error in update_render_view: {e}")
            traceback.print_exc()
    
    def closeEvent(self, event):
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.close()
        event.accept()

    def test_mid_widget(self):
        print("\n🎯 Creating test widget at middle of range")
        if hasattr(self, 'tf_canvas'):
            from tf_widgets import GaussianWidget
            test_widget = GaussianWidget()
            test_widget.center_intensity = 128
            test_widget.center_gradient = 128
            test_widget.opacity = 1.0
            test_widget.color = (1.0, 0.0, 0.0)
            self.tf_canvas.clear_widgets()
            self.tf_canvas.add_widget(test_widget)
            self.nd_manager.project_to_2d(self.feat_x, self.feat_y)
            if hasattr(self, 'widget_manager'):
                self.widget_manager.update_widget_list()
            self.update_render_view()
            print("✅ Test widget added at center")

    def test_red_cube(self):
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.test_force_red_independent()
    def test_cube_with_widget(self):
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.test_red_cube_with_widget()
    def test_isolated(self):
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.test_isolated_volume()

    def test_intensity_only(self):
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.force_intensity_volume()