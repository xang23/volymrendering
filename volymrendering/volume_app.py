import sys
import numpy as np
import vtk
# WidgetTF
from widget_factory import WidgetFactory, WidgetType
from unified_tf_canvas import UnifiedTFCanvas
from widget_manager_ui import WidgetManager

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# Import our modular components
from dataset_loader import DatasetLoader
from tf_manager import TFManager
from transfer_function_plot import TransferFunctionPlot
from transfer_function_2d import TransferFunction2D
from volume_renderer import VolumeRenderer
from tf_canvas_widget import TFCanvasWidget

import os
import glob

# ===== ADD GLOBAL EXCEPTION HANDLER HERE =====
def global_exception_handler(exctype, value, tb):
    print(f"\n❌ GLOBAL CRASH DETECTED!")
    print(f"   Type: {exctype.__name__}")
    print(f"   Error: {value}")
    print("   Traceback:")
    traceback.print_tb(tb)
    print("\n⚠️ Program will attempt to continue...")
    # Call the original exception handler
    sys.__excepthook__(exctype, value, tb)

# Install the handler
sys.excepthook = global_exception_handler
# =============================================

def show_project_structure():
    print("YOUR PROJECT STRUCTURE:")
    for file in sorted(glob.glob("*.py")):
        print(f"  {file}")
    if os.path.exists("data"):
        print("  data/")
        for data_file in glob.glob("data/*"):
            print(f"    {os.path.basename(data_file)}")

show_project_structure()

class VolumeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('VTK Volume + Dual Transfer Function Comparison')
        
        # Initialize components
        self.dataset_loader = DatasetLoader(self)
        self.volume_renderer = VolumeRenderer()
        self._tf_change_source = None
        self._active_tf_system = 'point'
        
        # nD features
        self.feature_browser = None
        self.current_dataset_dir = None
        
        # Widget manager window
        self.widget_manager_window = None
        
        self.setup_ui()
        self.setup_data_components()
        self.setup_dual_transfer_functions()

        # ND Manager
        from nd_widget_manager import NDWidgetManager
        self.nd_manager = NDWidgetManager()
        self.tf_canvas.set_nd_callback(self.on_widget_moved_in_nd)
        self.setup_artifact_analyzer_button()
        self.setup_quick_benchmark()

    def setup_ui(self):
        """Setup the main user interface with dual render views"""
        self.frame = QtWidgets.QFrame()
        self.main_layout = QtWidgets.QVBoxLayout(self.frame)

        # MAIN VERTICAL SPLITTER
        main_splitter = QtWidgets.QSplitter(Qt.Vertical)

        # RENDER CONTAINER
        render_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Point-based renderer
        self.vtkWidget_point = QVTKRenderWindowInteractor()
        self.vtkWidget_point.GetRenderWindow().AddRenderer(self.volume_renderer.get_renderer())

        # Widget-based renderer
        from volume_renderer import VolumeRenderer
        self.volume_renderer_widget = VolumeRenderer()
        self.vtkWidget_widget = QVTKRenderWindowInteractor()
        self.vtkWidget_widget.GetRenderWindow().AddRenderer(self.volume_renderer_widget.get_renderer())

        # Set up interactors
        self.interactor_point = self.vtkWidget_point.GetRenderWindow().GetInteractor()
        self.interactor_widget = self.vtkWidget_widget.GetRenderWindow().GetInteractor()

        # Create labeled containers
        point_render_container = self.create_render_container(self.vtkWidget_point, "Point-based TF Render")
        widget_render_container = self.create_render_container(self.vtkWidget_widget, "Widget-based TF Render")

        # Add to render splitter
        render_splitter.addWidget(point_render_container)
        render_splitter.addWidget(widget_render_container)

        # TF EDITORS
        tf_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        left_panel = self.setup_point_panel()
        right_panel = self.setup_widget_panel()

        tf_splitter.addWidget(left_panel)
        tf_splitter.addWidget(right_panel)

        # Add both splitters to main splitter
        main_splitter.addWidget(render_splitter)
        main_splitter.addWidget(tf_splitter)

        main_splitter.setSizes([600, 400])

        # Toolbar
        self.toolbar = self.create_toolbar()

        # Styling
        main_splitter.setStyleSheet("QSplitter::handle { background-color: #c0c0c0; }")
        render_splitter.setStyleSheet("QSplitter::handle { background-color: #a0a0a0; }")
        tf_splitter.setStyleSheet("QSplitter::handle { background-color: #a0a0a0; }")

        self.main_layout.addLayout(self.toolbar)
        self.main_layout.addWidget(main_splitter)

        self.frame.setLayout(self.main_layout)
        self.setCentralWidget(self.frame)

        self.resize(1800, 1200)

        self.initialize_both_renderers()

    def create_render_container(self, vtk_widget, label_text):
        """Create a labeled container for a renderer"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        
        label = QtWidgets.QLabel(label_text)
        label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(vtk_widget)
        
        return container

    def initialize_both_renderers(self):
        """Initialize both VTK renderers"""
        if hasattr(self, 'image_data') and hasattr(self, 'reader'):
            self.volume_renderer_widget.set_volume_data(self.image_data, self.reader)
        
        self.interactor_point.Initialize()
        self.interactor_widget.Initialize()
        self.interactor_point.Start()
        self.interactor_widget.Start()

    def setup_point_panel(self):
        """Setup point-based panel"""
        panel_widget = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel_widget)
    
        panel_layout.addWidget(QtWidgets.QLabel("Point-based Transfer Function"))
    
        self.point_canvas_container = QtWidgets.QStackedWidget()
        panel_layout.addWidget(self.point_canvas_container)
    
        controls = QtWidgets.QHBoxLayout()
        self.point_view_toggle = QtWidgets.QPushButton('Switch to 2D TF')
        self.point_view_toggle.setCheckable(True)
        self.point_view_toggle.toggled.connect(self.toggle_point_view)
        controls.addWidget(self.point_view_toggle)
    
        self.point_reset_btn = QtWidgets.QPushButton('Reset View')
        self.point_reset_btn.clicked.connect(self.reset_point_view)
        controls.addWidget(self.point_reset_btn)
    
        self.point_active_indicator = QtWidgets.QLabel(" ACTIVE")
        self.point_active_indicator.setStyleSheet("color: green; font-weight: bold;")
        controls.addWidget(self.point_active_indicator)
    
        panel_layout.addLayout(controls)
    
        return panel_widget

    def setup_widget_panel(self):
        """Setup widget-based panel"""
        panel_widget = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel_widget)

        panel_layout.addWidget(QtWidgets.QLabel("Widget-based Transfer Function"))

        self.widget_canvas_container = QtWidgets.QStackedWidget()
        panel_layout.addWidget(self.widget_canvas_container)

        controls = QtWidgets.QHBoxLayout()
        self.widget_view_toggle = QtWidgets.QPushButton('Switch to 1D View')
        self.widget_view_toggle.setCheckable(True)
        self.widget_view_toggle.toggled.connect(self.toggle_widget_view)
        controls.addWidget(self.widget_view_toggle)

        self.widget_reset_btn = QtWidgets.QPushButton('Reset View')
        self.widget_reset_btn.clicked.connect(self.reset_widget_view)
        controls.addWidget(self.widget_reset_btn)

        self.widget_active_indicator = QtWidgets.QLabel(" INACTIVE") 
        self.widget_active_indicator.setStyleSheet("color: gray;")
        controls.addWidget(self.widget_active_indicator)

        self.open_widget_manager_btn = QtWidgets.QPushButton('Open Widget Manager')
        self.open_widget_manager_btn.clicked.connect(self.show_widget_manager)
        controls.addWidget(self.open_widget_manager_btn)

        panel_layout.addLayout(controls)

        return panel_widget

    def create_widget_manager_window(self):
        """Create floating widget manager window"""
        self.widget_manager_window = QtWidgets.QMainWindow(self)
        self.widget_manager_window.setWindowTitle("Widget Manager")
        self.widget_manager_window.setObjectName("WidgetManagerWindow")
    
        self.widget_manager_window.setWindowFlags(
            Qt.Tool | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint
        )
    
        central_widget = QtWidgets.QWidget()
        self.widget_manager_window.setCentralWidget(central_widget)
    
        layout = QtWidgets.QVBoxLayout(central_widget)
    
        self.widget_manager = WidgetManager(self.tf_canvas)
        layout.addWidget(self.widget_manager)
    
        self.widget_manager_window.resize(400, 300)
        self.widget_manager_window.closeEvent = self.widget_manager_window_close_event

    def widget_manager_window_close_event(self, event):
        self.widget_manager_window.hide()
        event.ignore()

    def show_widget_manager(self):
        if not hasattr(self, 'widget_manager_window') or self.widget_manager_window is None:
            self.create_widget_manager_window()
    
        main_window_geometry = self.geometry()
        self.widget_manager_window.move(
            main_window_geometry.right() - 450,
            main_window_geometry.top() + 50
        )
    
        self.widget_manager_window.show()
        self.widget_manager_window.raise_()
        self.widget_manager_window.activateWindow()

    def create_toolbar(self):
        """Create main toolbar"""
        self.toolbar = QtWidgets.QHBoxLayout()
        
        self.log_checkbox = QtWidgets.QCheckBox('Log Histogram')
        self.log_checkbox.stateChanged.connect(self.toggle_log_histogram)
        self.toolbar.addWidget(self.log_checkbox)
        
        self.toolbar.addWidget(QtWidgets.QLabel("Active System:"))
        self.system_selector = QtWidgets.QComboBox()
        self.system_selector.addItem("Point-based TF", 'point')
        self.system_selector.addItem("Widget-based TF", 'widget')
        self.system_selector.addItem("nD Feature TF", 'nd')
        self.system_selector.currentTextChanged.connect(self.switch_active_system)
        self.toolbar.addWidget(self.system_selector)

        self.toolbar.addStretch(1)

        self.widget_manager_btn = QtWidgets.QPushButton("Widget Manager")
        self.widget_manager_btn.clicked.connect(self.show_widget_manager)
        self.toolbar.addWidget(self.widget_manager_btn)

        self.load_data_btn = QtWidgets.QPushButton("Load Dataset")
        self.load_data_btn.setToolTip("Choose a dataset file to load")
        self.load_data_btn.clicked.connect(self.load_volume_dialog)
        self.toolbar.addWidget(self.load_data_btn)

        self.tf_selector = QtWidgets.QComboBox()
        self.tf_selector.currentIndexChanged.connect(self.load_selected_tf)
        self.toolbar.addWidget(self.tf_selector)

        self.save_tf_btn = QtWidgets.QPushButton("Save TF")
        self.save_tf_btn.clicked.connect(self.save_current_tf)
        self.toolbar.addWidget(self.save_tf_btn)

        return self.toolbar

    def setup_data_components(self):
        """Initialize data with default dataset"""
        default_path = r"C:\Users\josde002\source\repos\volymrendering\data\head-binary-zlib.vti"
        try:
            self.load_volume(default_path)
        except Exception as e:
            print("Failed to load default dataset:", e)
            self.setup_fallback_data()

    def setup_fallback_data(self):
        empty = np.zeros((100,), dtype=np.float32)
        self.normalized_scalars = empty
        self.gradient_normalized = empty
        self.intensity_range = (0.0, 1.0)
        self.gradient_range = (0.0, 1.0)

    def setup_dual_transfer_functions(self):
        """Setup both TF systems"""
        print("\n" + "="*50)
        print("SETTING UP DUAL TRANSFER FUNCTIONS")
        print("="*50)

        self.tf_manager = TFManager(self.tf_selector, self)
        points_x, points_y, colors = self.tf_manager.get_initial_tf_data(self.normalized_scalars, data_range=self.intensity_range)

        # Point-based TF
        print("Setting up point-based TF...")
        self.plot_canvas = TransferFunctionPlot(
            self.update_opacity_function_from_1d,
            self.normalized_scalars, 
            self.log_checkbox
        )
        self.plot_canvas.points_x = points_x
        self.plot_canvas.points_y = points_y  
        self.plot_canvas.colors = colors
        self.plot_canvas._sort_points_with_colors()
        self.plot_canvas._draw()

        self.tf1d_widget = TFCanvasWidget(self.plot_canvas, parent=self, label='Reset 1D View')
        self.point_canvas_container.addWidget(self.tf1d_widget)

        # 2D TF canvas
        hist2d, _, _ = np.histogram2d(
            self.normalized_scalars, self.gradient_normalized, 
            bins=(256, 256), range=((0, 255), (0, 255))
        )
        self.tf2d_canvas = TransferFunction2D(
            hist2d, self.intensity_range, self.gradient_range, self.log_checkbox
        )
        self.tf2d_canvas.set_tf_state(points_x, points_y, colors)
        self.tf2d_widget = TFCanvasWidget(self.tf2d_canvas, parent=self, label='Reset 2D View')
        self.point_canvas_container.addWidget(self.tf2d_widget)

        self.point_canvas_container.setCurrentIndex(0)
        self.point_view_toggle.setChecked(False)

        # Widget-based TF
        print("Setting up widget-based TF...")
        self.tf_canvas = UnifiedTFCanvas(
            tf_type='2d',
            data=self.normalized_scalars,
            gradient_data=self.gradient_normalized,
            update_callback=self.update_volume_from_widgets,
            x_label="Intensity",              # ← ADD THIS
            y_label="Gradient Magnitude"      # ← ADD THIS
        )

        self.canvas_widget = TFCanvasWidget(self.tf_canvas, self, label='Reset TF View')
        self.widget_canvas_container.addWidget(self.canvas_widget)

        self.widget_manager = WidgetManager(self.tf_canvas)

        # Create initial widget
        visible_widget = WidgetFactory.create_widget(
            WidgetType.GAUSSIAN,
            center_intensity=30,
            center_gradient=50,
            intensity_std=20,
            gradient_std=30,
            opacity=0.8,
            color=(0.8, 0.2, 0.2),
            blend_mode='max'
        )

        if hasattr(self, 'nd_manager'):
            self.nd_manager.add_widget(visible_widget)
            projected = self.nd_manager.project_to_2d('Intensity', 'Gradient')
            for widget in projected:
                self.tf_canvas.add_widget(widget)
        else:
            self.tf_canvas.add_widget(visible_widget)

        self.widget_manager.update_widget_list()
        print("Created VISIBLE red widget")

        self.update_opacity_function(points_x, points_y, colors)

        self.interactor_point.Initialize()
        self.interactor_point.Start()
        self.interactor_widget.Initialize()
        self.interactor_widget.Start()

        print("Dual transfer functions setup complete")
        print("="*50 + "\n")

    def switch_active_system(self, system_name):
        system_type = self.system_selector.currentData()
        self._active_tf_system = system_type
    
        if system_type == 'point':
            self.point_active_indicator.setText(" ACTIVE")
            self.point_active_indicator.setStyleSheet("color: green; font-weight: bold;")
            self.widget_active_indicator.setText(" INACTIVE")
            self.widget_active_indicator.setStyleSheet("color: gray;")
            self.highlight_active_render('point')
        
        elif system_type == 'widget':
            self.point_active_indicator.setText(" INACTIVE")
            self.point_active_indicator.setStyleSheet("color: gray;")
            self.widget_active_indicator.setText(" ACTIVE")
            self.widget_active_indicator.setStyleSheet("color: green; font-weight: bold;")
            self.highlight_active_render('widget')
            
            if self.feature_browser:
                self.feature_browser.hide()
            
            self.update_volume_from_widgets()
            
        elif system_type == 'nd':
            self.point_active_indicator.setText(" INACTIVE")
            self.point_active_indicator.setStyleSheet("color: gray;")
            self.widget_active_indicator.setText(" INACTIVE") 
            self.widget_active_indicator.setStyleSheet("color: gray;")
            self.safe_activate_nd_mode()

    def highlight_active_render(self, active_system):
        self.reset_render_highlights()
        if active_system == 'point':
            self.vtkWidget_point.setStyleSheet("border: 3px solid green;")
        elif active_system == 'widget':
            self.vtkWidget_widget.setStyleSheet("border: 3px solid green;")

    def reset_render_highlights(self):
        self.vtkWidget_point.setStyleSheet("border: 1px solid gray;")
        self.vtkWidget_widget.setStyleSheet("border: 1px solid gray;")

    def safe_activate_nd_mode(self):
        print("Activating Feature Matrix mode...")
    
        if not hasattr(self, 'all_features') or not self.all_features:
            print("Cannot activate nD mode: no features loaded")
            self.system_selector.setCurrentIndex(1)
            return
    
        try:
            feature_data = self.all_features
        
            if self.feature_browser is None:
                print("Creating feature matrix...")
                from simple_feature_browser import SimpleMatrixBrowser
            
                self.feature_browser = SimpleMatrixBrowser(
                    feature_data_dict=feature_data,
                    update_callback=self.load_projection
                )
            
                if hasattr(self, 'main_layout'):
                    if self.feature_browser.parent():
                        self.feature_browser.setParent(None)
                    self.main_layout.insertWidget(2, self.feature_browser)
            else:
                self.feature_browser.feature_data = feature_data
                self.feature_browser.update_matrix()
        
            self.nd_manager.update_features(list(feature_data.keys()))
        
            if hasattr(self, 'tf_canvas'):
                for widget in self.tf_canvas.widgets:
                    if widget not in self.nd_manager.widgets:
                        self.nd_manager.add_widget(widget)
        
            self.feature_browser.show()
            print(f"Feature Matrix activated with {len(feature_data)} features")
        
            if 'Intensity' in feature_data and 'Gradient' in feature_data:
                self.load_projection('Intensity', 'Gradient')
            else:
                keys = list(feature_data.keys())
                self.load_projection(keys[0], keys[1])
        
        except Exception as e:
            print(f"Failed to activate nD mode: {e}")
            import traceback
            traceback.print_exc()
            self.system_selector.setCurrentIndex(1)

    def on_matrix_cell_clicked(self, feature_x, feature_y):
        print(f"Matrix cell clicked: {feature_x} vs {feature_y}")

        try:
            feature_data = self.feature_browser.feature_data
            data_x = feature_data[feature_x]
            data_y = feature_data[feature_y]
    
            self.tf_canvas.data = data_x
            self.tf_canvas.gradient_data = data_y
    
            self.tf_canvas._update_data_ranges()
            self.tf_canvas._setup_canvas()
            self.tf_canvas._draw()
    
            print(f"Canvas updated with {feature_x} vs {feature_y}")
            self.update_volume_from_widgets()
        
            self.open_feature_popup(feature_x, feature_y)
    
        except Exception as e:
            print(f"Error updating main TF: {e}")
            import traceback
            traceback.print_exc()

    def update_volume_from_widgets(self):
        if self._active_tf_system not in ['widget', 'nd']:
            print(f"Widget system not active ({self._active_tf_system})")
            return

        samples = self.tf_canvas.sample_for_vtk()

        if samples:
            intensities = [s[0] for s in samples]
            opacities = [s[1] for s in samples]
            colors = [s[2] for s in samples]
    
            int_min, int_max = self.intensity_range
            scaled_intensities = [int_min + (i / 255.0) * (int_max - int_min) for i in intensities]
        
            gradient_opacities = None
            if hasattr(self.tf_canvas, '_cached_gradient_opacity'):
                gradient_op = self.tf_canvas._cached_gradient_opacity
                grad_min, grad_max = self.gradient_range
                gradient_opacities = [(grad_min + (g / 255.0) * (grad_max - grad_min), gradient_op[g]) 
                                    for g in range(0, 256, 4) if gradient_op[g] > 0.01]

            self.volume_renderer_widget.update_transfer_functions(
                scaled_intensities, opacities, colors, 
                self.intensity_range, gradient_opacities, self.gradient_range
            )

            self.vtkWidget_widget.GetRenderWindow().Render()
            print("Render complete")

    def update_opacity_function(self, xs, ys, colors):
        if self._active_tf_system != 'point':
            return

        self.volume_renderer.update_transfer_functions(
            xs, ys, colors, self.intensity_range, None, None
        )

        if self._tf_change_source == '1d':
            try:
                self.tf2d_canvas.set_tf_state(xs, ys, colors)
            except Exception as e:
                print(f"Error syncing to 2D canvas: {e}")
        elif self._tf_change_source == '2d':
            pass
        else:
            try:
                if hasattr(self, 'plot_canvas'):
                    self.plot_canvas.points_x = list(xs)
                    self.plot_canvas.points_y = list(ys)
                    self.plot_canvas.colors = [tuple(c) for c in colors]
                    self.plot_canvas._sort_points_with_colors()
                    self.plot_canvas._draw()
                if hasattr(self, 'tf2d_canvas'):
                    self.tf2d_canvas.set_tf_state(xs, ys, colors)
            except Exception as e:
                print(f"Error syncing canvases: {e}")

        self.vtkWidget_point.GetRenderWindow().Render()

    def toggle_point_view(self, show_2d):
        idx = 1 if show_2d else 0
        self.point_canvas_container.setCurrentIndex(idx)
        self.point_view_toggle.setText('Switch to 1D TF' if show_2d else 'Switch to 2D TF')
    
        if hasattr(self, 'plot_canvas') and hasattr(self, 'tf2d_canvas'):
            xs, ys, colors = self.plot_canvas.points_x, self.plot_canvas.points_y, self.plot_canvas.colors
            if show_2d:
                self.tf2d_canvas.set_tf_state(xs, ys, colors)
            else:
                self.plot_canvas.set_tf_state(xs, ys, colors)

    def toggle_widget_view(self, show_1d):
        if hasattr(self, 'tf_canvas'):
            if show_1d:
                self.tf_canvas.set_tf_type('1d')
                self.widget_view_toggle.setText("Switch to 2D View")
            else:
                self.tf_canvas.set_tf_type('2d')
                self.widget_view_toggle.setText("Switch to 1D View")

    def reset_point_view(self):
        try:
            current_widget = self.point_canvas_container.currentWidget()
            if hasattr(current_widget, 'canvas'):
                current_widget.canvas.reset_view()
        except Exception as e:
            print(f"Error resetting point view: {e}")

    def reset_widget_view(self):
        try:
            if hasattr(self, 'tf_canvas'):
                self.tf_canvas.reset_view()
        except Exception as e:
            print(f"Error resetting widget view: {e}")

    def update_opacity_function_from_1d(self, xs, ys, colors):
        print(f"1D TF callback: Received {len(xs)} points in DISPLAY coordinates (0-255)")
    
        int_min, int_max = self.intensity_range
        actual_xs = [int_min + (x / 255.0) * (int_max - int_min) for x in xs]
    
        if self._tf_change_source == '2d':
            return
        self._tf_change_source = '1d'
        self.update_opacity_function(actual_xs, ys, colors)
        self._tf_change_source = None

    def update_opacity_function_from_2d(self, xs, ys, colors):
        print(f"2D TF callback: Received {len(xs)} points")
    
        if max(xs) <= 255.0:
            int_min, int_max = self.intensity_range
            xs = [int_min + (x / 255.0) * (int_max - int_min) for x in xs]
    
        if self._tf_change_source == '1d':
            return
        self._tf_change_source = '2d'
    
        if hasattr(self, 'plot_canvas'):
            int_min, int_max = self.intensity_range
            display_xs = [255.0 * (x - int_min) / (int_max - int_min) for x in xs]
            self.plot_canvas.points_x = display_xs
            self.plot_canvas.points_y = ys
            self.plot_canvas.colors = [tuple(c) for c in colors]
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
    
        self.update_opacity_function(xs, ys, colors)
        self._tf_change_source = None

    def load_volume_dialog(self):
        file_path = self.dataset_loader.load_volume_dialog()
        if file_path:
            try:
                self.load_volume(file_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Failed", f"Failed to load {file_path}:\n{e}")

    def load_volume(self, file_path):
        """Load and process volume data"""
        image_data, reader, all_features = self.dataset_loader.load_volume(file_path)
    
        if 'Intensity' in all_features:
            np_scalars = all_features['Intensity']
        else:
            first_key = list(all_features.keys())[0]
            np_scalars = all_features[first_key]
    
        if 'Gradient' in all_features:
            np_gradient = all_features['Gradient']
        else:
            np_gradient = np.zeros_like(np_scalars, dtype=np.float32)
    
        (self.normalized_scalars, self.gradient_normalized, 
         self.intensity_range, self.gradient_range) = self.dataset_loader.normalize_data(np_scalars, np_gradient)

        self.current_dataset_dir = os.path.dirname(file_path)
        self.all_features = all_features
        self.image_data = image_data
        self.reader = reader

        self.volume_renderer.set_volume_data(image_data, reader)
        self.volume_renderer_widget.set_volume_data(image_data, reader)

        self.update_tf_canvases()
        self.reset_widget_tf_for_new_data()

        self.volume_renderer.reset_camera()
        self.volume_renderer_widget.reset_camera()

        self.vtkWidget_point.GetRenderWindow().Render()
        self.vtkWidget_widget.GetRenderWindow().Render()

        return True

    def update_tf_canvases(self):
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.hist_data = self.normalized_scalars
            self.plot_canvas._draw()
        if hasattr(self, 'tf2d_canvas'):
            hist2d, _, _ = np.histogram2d(
                self.normalized_scalars, self.gradient_normalized, 
                bins=(256, 256), range=((0, 255), (0, 255))
            )
            self.tf2d_canvas.raw = hist2d
            if self.log_checkbox.isChecked():
                self.tf2d_canvas._on_log_toggled(True)
            else:
                self.tf2d_canvas._draw()

        if hasattr(self, 'tf_canvas'):
            self.tf_canvas.data = self.normalized_scalars
            self.tf_canvas.gradient_data = self.gradient_normalized
            self.tf_canvas._setup_canvas()
            self.tf_canvas._draw()
            print("Updated widget-based TF")
        
        print(f"New data ranges - Intensity: {self.intensity_range}, Gradient: {self.gradient_range}")

    def reset_widget_tf_for_new_data(self):
        if hasattr(self, 'tf_canvas'):
            self.tf_canvas.widgets.clear()
        
            intensity_center = (self.intensity_range[0] + self.intensity_range[1]) / 2
            gradient_center = (self.gradient_range[0] + self.gradient_range[1]) / 2
        
            default_widget = WidgetFactory.create_widget(
                WidgetType.GAUSSIAN,
                center_intensity=intensity_center,
                center_gradient=gradient_center,
                intensity_std=(self.intensity_range[1] - self.intensity_range[0]) * 0.1,
                gradient_std=(self.gradient_range[1] - self.gradient_range[0]) * 0.1
            )
            self.tf_canvas.add_widget(default_widget)
        
            if hasattr(self, 'widget_manager'):
                self.widget_manager.update_widget_list()

    def toggle_log_histogram(self, state):
        try:
            self.plot_canvas._draw()
        except Exception: pass
        try:
            self.tf2d_canvas._on_log_toggled(state)
        except Exception: pass

    def save_current_tf(self):
        if self._active_tf_system == 'point' and hasattr(self, 'plot_canvas'):
            self.tf_manager.save_current_tf(
                self.plot_canvas.points_x,
                self.plot_canvas.points_y,
                self.plot_canvas.colors,
                data_range=self.intensity_range
            )
        else:
            print("Widget TF saving not yet implemented")

    def load_selected_tf(self, idx):
        tf_data = self.tf_manager.load_selected_tf(idx, current_data_range=self.intensity_range)
    
        if tf_data is None:
            print("Failed to load TF, keeping current")
            return
        
        xs, ys, colors = tf_data
    
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.points_x = xs
            self.plot_canvas.points_y = ys
            self.plot_canvas.colors = colors
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
    
        self.volume_renderer.update_transfer_functions(xs, ys, colors, self.intensity_range)
        self.vtkWidget_point.GetRenderWindow().Render()
        self.vtkWidget_widget.GetRenderWindow().Render()

    def closeEvent(self, event):
        if hasattr(self, 'widget_manager_window') and self.widget_manager_window:
            self.widget_manager_window.close()
        event.accept()

    def on_widget_moved_in_nd(self, widget_2d, feat_x, feat_y, new_x, new_y):
        self.nd_manager.update_nd_position(widget_2d, new_x, new_y)

    def load_projection(self, feat_x, feat_y):
        print(f"Loading projection: {feat_x} vs {feat_y}")

        try:
            if hasattr(self, 'feature_browser') and self.feature_browser:
                feature_data = self.feature_browser.feature_data
                data_x = feature_data[feat_x]
                data_y = feature_data[feat_y]
        
                self.active_projection = (feat_x, feat_y)
        
                self.tf_canvas.raw_data_x = data_x
                self.tf_canvas.raw_data_y = data_y
        
                self.tf_canvas.data = self.dataset_loader.normalize_single(data_x)
                self.tf_canvas.gradient_data = self.dataset_loader.normalize_single(data_y)
        
                self.tf_canvas.intensity_range = (float(np.min(data_x)), float(np.max(data_x)))
                self.tf_canvas.gradient_range = (float(np.min(data_y)), float(np.max(data_y)))
        
                self.tf_canvas.set_projection_features(feat_x, feat_y)
        
                # Update the axis labels
                self.tf_canvas.update_axis_labels(feat_x, feat_y)
        
                if hasattr(self, 'nd_manager'):
                    self.nd_manager.feature_ranges[feat_x] = self.tf_canvas.intensity_range
                    self.nd_manager.feature_ranges[feat_y] = self.tf_canvas.gradient_range
        
                self.tf_canvas.clear_widgets()
                projected = self.nd_manager.project_to_2d(feat_x, feat_y)
                for widget in projected:
                    self.tf_canvas.add_widget(widget)
        
                self.tf_canvas._setup_canvas()
                self.tf_canvas._draw()
                self.update_volume_from_widgets()
        
        except Exception as e:
            print(f"Error loading projection: {e}")
            import traceback
            traceback.print_exc()

    def open_feature_popup(self, feat_x, feat_y):
        print(f"Opening popup: {feat_x} vs {feat_y}")

        if hasattr(self, 'open_popups'):
            for popup in self.open_popups:
                if hasattr(popup, 'feat_x') and hasattr(popup, 'feat_y'):
                    if popup.feat_x == feat_x and popup.feat_y == feat_y:
                        popup.raise_()
                        popup.activateWindow()
                        return

        try:
            if hasattr(self, 'feature_browser') and self.feature_browser:
                feature_data = self.feature_browser.feature_data
            
                # ===== ADD THIS MAPPING =====
                # Map display names to actual VTK array names
                display_to_array = {
                    'Intensity': 'Intensity',
                    'Scalars_': 'Intensity',
                    'Gradient': 'Gradient',
                    'Laplacian': 'Laplacian',
                    'Texture': 'Texture',
                    'Curvature': 'Curvature',
                    'Entropy': 'Entropy'
                }
            
                # Get the actual array names
                array_x = display_to_array.get(feat_y, feat_y)  # Note: swapped
                array_y = display_to_array.get(feat_x, feat_x)
            
                # Get the data using the display names (for the 2D view)
                data_x = feature_data[feat_y]
                data_y = feature_data[feat_x]
                # ============================

                from nd_popup import NDFeaturePopup
                print(f"🔧 Creating popup for {array_x} vs {array_y}")
                popup = NDFeaturePopup(
                    feat_y,  # Color feature (first in 2D TF)
                    feat_x,  # Opacity feature (second in 2D TF)
                    data_x, data_y,
                    self.nd_manager,
                    self.image_data,
                    self.all_features,
                    parent=self
                )

                print(f"✅ Popup created, positioning...")
                self.position_popup_near_cell(popup, feat_x, feat_y)
    
                print(f"🎯 Showing popup...")
                popup.show()
                popup.raise_()
    
                # Force Qt to process events
                QtWidgets.QApplication.processEvents()
    
                print(f"🔍 Popup should be visible: {popup.isVisible()}")
                print(f"🔍 Popup geometry: {popup.geometry()}")
                print(f"🔍 Popup window flags: {popup.windowFlags()}")

                if not hasattr(self, 'open_popups'):
                    self.open_popups = []
                self.open_popups.append(popup)

                popup.destroyed.connect(lambda: self.open_popups.remove(popup))
                print(f"✅ Popup fully initialized")
    
        except Exception as e:
            print(f"❌ Error opening popup: {e}")
            import traceback
            traceback.print_exc()

    def position_popup_near_cell(self, popup, feat_x, feat_y):
        if not hasattr(self, 'feature_browser'):
            return
    
        matrix = self.feature_browser.matrix_widget
        row = self.feature_browser.feature_names.index(feat_y)
        col = self.feature_browser.feature_names.index(feat_x)
    
        cell = matrix.layout().itemAtPosition(row+1, col+1).widget()
        if cell:
            global_pos = cell.mapToGlobal(cell.rect().topRight())
            popup.move(global_pos.x() + 10, global_pos.y())

    def setup_quick_benchmark(self):
        self.benchmark_btn = QtWidgets.QPushButton("Benchmark Current TF")
        self.benchmark_btn.clicked.connect(self.run_quick_benchmark)
        self.toolbar.addWidget(self.benchmark_btn)
        
        self.benchmark_status = QtWidgets.QLabel("Ready")
        self.toolbar.addWidget(self.benchmark_status)

    def run_quick_benchmark(self):
        import time
        import csv
        from datetime import datetime
        
        self.benchmark_status.setText("Running...")
        QtWidgets.QApplication.processEvents()
        
        widget_info = [f"{w.widget_type.value}({w.center_intensity:.0f},{w.center_gradient:.0f})" 
                      for w in self.nd_manager.widgets]
        
        times = []
        for i in range(100):
            start = time.perf_counter()
            self.vtkWidget_widget.GetRenderWindow().Render()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time
        
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.take_screenshot(filename)
        
        with open('benchmark_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Date', 'Widgets', 'Avg(ms)', 'FPS', 'Screenshot'])
            writer.writerow([datetime.now().isoformat(), '+'.join(widget_info), 
                            f"{avg_time:.2f}", f"{fps:.1f}", filename])
        
        self.benchmark_status.setText(f"FPS: {fps:.1f} | {avg_time:.2f}ms")
        QtWidgets.QMessageBox.information(self, "Benchmark Complete", 
            f"Average: {avg_time:.2f} ms\nFPS: {fps:.1f}\nScreenshot saved")

    def take_screenshot(self, filename):
        w = self.vtkWidget_widget.GetRenderWindow()
        image = vtk.vtkWindowToImageFilter()
        image.SetInput(w)
        image.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(image.GetOutputPort())
        writer.Write()

    def setup_artifact_analyzer_button(self):
        self.analyzer_btn = QtWidgets.QPushButton("🔬 Artifact Analyzer")
        self.analyzer_btn.clicked.connect(self.open_artifact_analyzer)
        self.toolbar.addWidget(self.analyzer_btn)

    def open_artifact_analyzer(self):
        try:
            from artifact_analyzer import ArtifactAnalyzer
            self.analyzer = ArtifactAnalyzer(self)
            self.analyzer.show()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", 
                f"Could not open Artifact Analyzer:\n{str(e)}")

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolumeApp()
    window.show()
    sys.exit(app.exec_())