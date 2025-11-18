import sys
import numpy as np

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

def show_project_structure():
    print("📁 YOUR PROJECT STRUCTURE:")
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
        self._active_tf_system = 'point'  # 'point' or 'widget' - only one active for rendering
        
        # nD features - INITIALIZE BUT DON'T CREATE
        self.feature_browser = None
        self.current_dataset_dir = None
        
        self.setup_ui()
        self.setup_data_components()
        self.setup_dual_transfer_functions()

    def setup_ui(self):
        """Setup the main user interface with dual view"""
        self.frame = QtWidgets.QFrame()
        self.main_layout = QtWidgets.QVBoxLayout(self.frame)  # Store as instance variable

        # VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.volume_renderer.get_renderer())
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.main_layout.addWidget(self.vtkWidget)

        # Toolbar
        toolbar = self.create_toolbar()
        self.main_layout.addLayout(toolbar)

        # DUAL VIEW CONTAINER
        self.dual_view_container = QtWidgets.QHBoxLayout()
        
        # Setup both panels
        left_panel = self.setup_point_panel()
        right_panel = self.setup_widget_panel()
        
        self.dual_view_container.addLayout(left_panel)
        self.dual_view_container.addLayout(right_panel)
        self.main_layout.addLayout(self.dual_view_container)

        self.frame.setLayout(self.main_layout)
        self.setCentralWidget(self.frame)

    def setup_point_panel(self):
        """Setup point-based panel"""
        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(QtWidgets.QLabel("Point-based Transfer Function"))
        
        self.point_canvas_container = QtWidgets.QStackedWidget()
        panel.addWidget(self.point_canvas_container)
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.point_view_toggle = QtWidgets.QPushButton('Switch to 2D TF')
        self.point_view_toggle.setCheckable(True)
        self.point_view_toggle.toggled.connect(self.toggle_point_view)
        controls.addWidget(self.point_view_toggle)
        
        self.point_reset_btn = QtWidgets.QPushButton('Reset View')
        self.point_reset_btn.clicked.connect(self.reset_point_view)
        controls.addWidget(self.point_reset_btn)
        
        # Active system indicator
        self.point_active_indicator = QtWidgets.QLabel("⚫ ACTIVE")
        self.point_active_indicator.setStyleSheet("color: green; font-weight: bold;")
        controls.addWidget(self.point_active_indicator)
        
        panel.addLayout(controls)
        return panel

    def setup_widget_panel(self):
        """Setup widget-based panel"""
        panel = QtWidgets.QVBoxLayout()
        panel.addWidget(QtWidgets.QLabel("Widget-based Transfer Function"))
        
        self.widget_canvas_container = QtWidgets.QStackedWidget()
        panel.addWidget(self.widget_canvas_container)
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        self.widget_view_toggle = QtWidgets.QPushButton('Switch to 1D View')
        self.widget_view_toggle.setCheckable(True)
        self.widget_view_toggle.toggled.connect(self.toggle_widget_view)
        controls.addWidget(self.widget_view_toggle)
        
        self.widget_reset_btn = QtWidgets.QPushButton('Reset View')
        self.widget_reset_btn.clicked.connect(self.reset_widget_view)
        controls.addWidget(self.widget_reset_btn)
        
        # Active system indicator
        self.widget_active_indicator = QtWidgets.QLabel("⚫ INACTIVE") 
        self.widget_active_indicator.setStyleSheet("color: gray;")
        controls.addWidget(self.widget_active_indicator)
        
        panel.addLayout(controls)
        
        # Widget manager placeholder
        self.widget_manager_placeholder = QtWidgets.QWidget()
        self.widget_manager_placeholder.setMinimumHeight(200)
        panel.addWidget(self.widget_manager_placeholder)
        
        return panel

    def create_toolbar(self):
        """Create the main toolbar with controls."""
        toolbar = QtWidgets.QHBoxLayout()
        
        # Log checkbox
        self.log_checkbox = QtWidgets.QCheckBox('Log Histogram')
        self.log_checkbox.stateChanged.connect(self.toggle_log_histogram)
        toolbar.addWidget(self.log_checkbox)
        
        # Active system switcher
        toolbar.addWidget(QtWidgets.QLabel("Active System:"))
        self.system_selector = QtWidgets.QComboBox()
        self.system_selector.addItem("Point-based TF", 'point')
        self.system_selector.addItem("Widget-based TF", 'widget')
        self.system_selector.addItem("nD Feature TF", 'nd')
        self.system_selector.currentTextChanged.connect(self.switch_active_system)
        toolbar.addWidget(self.system_selector)

        toolbar.addStretch(1)

        # Load dataset button
        self.load_data_btn = QtWidgets.QPushButton("Load Dataset")
        self.load_data_btn.setToolTip("Choose a .vti or .vol file to load (remembers last folder).")
        self.load_data_btn.clicked.connect(self.load_volume_dialog)
        toolbar.addWidget(self.load_data_btn)

        # TF selector
        self.tf_selector = QtWidgets.QComboBox()
        self.tf_selector.currentIndexChanged.connect(self.load_selected_tf)
        toolbar.addWidget(self.tf_selector)

        # Save TF button
        self.save_tf_btn = QtWidgets.QPushButton("Save TF")
        self.save_tf_btn.clicked.connect(self.save_current_tf)
        toolbar.addWidget(self.save_tf_btn)

        return toolbar

    def setup_data_components(self):
        """Initialize data components with default or loaded data."""
        # Try to load default dataset
        default_path = r"C:\Users\josde002\source\repos\volymrendering\data\head-binary-zlib.vti"
        try:
            self.load_volume(default_path)
        except Exception as e:
            print("Failed to load default dataset:", e)
            # Create empty data for UI initialization
            self.setup_fallback_data()

    def setup_fallback_data(self):
        """Setup fallback data when no volume is loaded."""
        empty = np.zeros((100,), dtype=np.float32)
        self.normalized_scalars = empty
        self.gradient_normalized = empty
        self.intensity_range = (0.0, 1.0)
        self.gradient_range = (0.0, 1.0)

    def setup_dual_transfer_functions(self):
        """Setup both transfer function systems for dual view - OPTIMIZED"""
        print("\n" + "="*50)
        print("SETTING UP DUAL TRANSFER FUNCTIONS")
        print("="*50)

        # Initialize TF manager
        self.tf_manager = TFManager(self.tf_selector, self)
        points_x, points_y, colors = self.tf_manager.get_initial_tf_data(self.normalized_scalars)

        # SETUP POINT-BASED TF (left panel)
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

        # Start with 1D view
        self.point_canvas_container.setCurrentIndex(0)
        self.point_view_toggle.setChecked(False)

        # SETUP WIDGET-BASED TF (right panel)
        print("Setting up widget-based TF...")
        self.tf_canvas = UnifiedTFCanvas(
            tf_type='2d',
            data=self.normalized_scalars,
            gradient_data=self.gradient_normalized,
            update_callback=self.update_volume_from_widgets
        )
        
        self.canvas_widget = TFCanvasWidget(self.tf_canvas, self, label='Reset TF View')
        self.widget_canvas_container.addWidget(self.canvas_widget)

        # Replace placeholder with actual widget manager
        self.widget_manager = WidgetManager(self.tf_canvas)
        self.widget_manager_placeholder.setParent(None)
        
        # Find the right panel and add widget manager
        for i in range(self.frame.layout().count()):
            item = self.frame.layout().itemAt(i)
            if isinstance(item, QtWidgets.QHBoxLayout):  # dual_view_container
                right_panel = item.itemAt(1)  # Right panel is index 1
                if right_panel and right_panel.layout():
                    right_panel.layout().addWidget(self.widget_manager)
                    break

        # Add initial widget
        test_widget = WidgetFactory.create_widget(WidgetType.GAUSSIAN)
        self.tf_canvas.add_widget(test_widget)
        self.widget_manager.update_widget_list()

        # Initialize with point-based TF (default active system)
        self.update_opacity_function(points_x, points_y, colors)

        # Initialize VTK
        self.interactor.Initialize()
        self.interactor.Start()

        print("Dual transfer functions setup complete")
        print("="*50 + "\n")

    # ===== PERFORMANCE OPTIMIZATIONS =====
    
    def switch_active_system(self, system_name):
        """Switch which TF system is active for rendering - OPTIMIZED"""
        system_type = self.system_selector.currentData()
        self._active_tf_system = system_type
        
        # Update UI indicators
        if system_type == 'point':
            self.point_active_indicator.setText("⚫ ACTIVE")
            self.point_active_indicator.setStyleSheet("color: green; font-weight: bold;")
            self.widget_active_indicator.setText("⚫ INACTIVE")
            self.widget_active_indicator.setStyleSheet("color: gray;")
            
            # Hide feature browser if it exists
            if self.feature_browser:
                self.feature_browser.hide()
            
            # Trigger render with point-based TF
            if hasattr(self, 'plot_canvas'):
                self.update_opacity_function(
                    self.plot_canvas.points_x, 
                    self.plot_canvas.points_y, 
                    self.plot_canvas.colors
                )
                
        elif system_type == 'widget':
            self.point_active_indicator.setText("⚫ INACTIVE")
            self.point_active_indicator.setStyleSheet("color: gray;")
            self.widget_active_indicator.setText("⚫ ACTIVE")
            self.widget_active_indicator.setStyleSheet("color: green; font-weight: bold;")
            
            # Hide feature browser if it exists
            if self.feature_browser:
                self.feature_browser.hide()
            
            # Trigger render with widget-based TF
            self.update_volume_from_widgets()
            
        elif system_type == 'nd':  # nD mode - MATRIX IMPLEMENTATION
            self.point_active_indicator.setText("⚫ INACTIVE")
            self.point_active_indicator.setStyleSheet("color: gray;")
            self.widget_active_indicator.setText("⚫ INACTIVE") 
            self.widget_active_indicator.setStyleSheet("color: gray;")
            
            # ACTIVATE MATRIX MODE
            self.safe_activate_nd_mode()

    def safe_activate_nd_mode(self):
        """Safely activate nD mode with feature matrix"""
        print("🔄 Activating Feature Matrix mode...")
        
        # Check if we have the required components
        if not hasattr(self, 'normalized_scalars') or not hasattr(self, 'tf_canvas'):
            print("❌ Cannot activate nD mode: missing required components")
            self.system_selector.setCurrentIndex(1)  # Fall back to widget mode
            return
            
        try:
            # Import and create MATRIX browser (not simple feature browser)
            from simple_feature_browser import SimpleMatrixBrowser
            
            # Create feature data dictionary from your existing data
            feature_data = {
                'Intensity': self.normalized_scalars,
                'Gradient': self.gradient_normalized
            }
            # Add more features here as needed:
            # feature_data['Texture'] = your_texture_data
            # feature_data['Curvature'] = your_curvature_data
            
            if self.feature_browser is None:
                print("🔧 Creating feature matrix...")
                self.feature_browser = SimpleMatrixBrowser(
                    feature_data_dict=feature_data,
                    update_callback=self.on_matrix_cell_clicked
                )
                
                # Add to main layout (insert after toolbar, before dual view)
                if hasattr(self, 'main_layout'):
                    # Remove existing feature browser if any
                    if self.feature_browser.parent():
                        self.feature_browser.setParent(None)
                    self.main_layout.insertWidget(2, self.feature_browser)
            
            # Show the feature matrix
            if self.feature_browser:
                self.feature_browser.show()
                feature_count = len(feature_data)
                print(f"✅ Feature Matrix activated with {feature_count} features")
                
            # Use widget-based rendering
            self.update_volume_from_widgets()
            
        except Exception as e:
            print(f"❌ Failed to activate Feature Matrix: {e}")
            self.system_selector.setCurrentIndex(1)  # Fall back to widget mode
            QtWidgets.QMessageBox.warning(self, "nD Mode Error", 
                                        f"Could not activate feature matrix:\n{e}")

    def on_matrix_cell_clicked(self, feature_x, feature_y):
        """When user clicks a cell in the matrix - FIXED VERSION"""
        print(f"🎯 Loading into main TF: {feature_x} vs {feature_y}")
    
        try:
            feature_data = self.feature_browser.feature_data
            data_x = feature_data[feature_x]
            data_y = feature_data[feature_y]
        
            # Update the canvas with new data
            self.tf_canvas.data = data_x
            self.tf_canvas.gradient_data = data_y
        
            # Force canvas to update ranges and redraw
            self.tf_canvas._update_data_ranges()
            self.tf_canvas._setup_canvas()
            self.tf_canvas._draw()
        
            print(f"✅ Canvas updated with {feature_x} vs {feature_y}")
            print(f"📊 New ranges: intensity={self.tf_canvas.intensity_range}, gradient={self.tf_canvas.gradient_range}")
        
            # Update volume rendering
            self.update_volume_from_widgets()
        
        except Exception as e:
            print(f"❌ Error updating main TF: {e}")
            import traceback
            traceback.print_exc()

    def update_volume_from_widgets(self):
        """Update volume from widget-based TF - only if active"""
        # Allow both 'widget' AND 'nd' modes to use widget rendering
        if self._active_tf_system not in ['widget', 'nd']:
            return  # Skip if not active
            
        samples = self.tf_canvas.sample_for_vtk()
        if samples:
            intensities = [s[0] for s in samples]
            opacities = [s[1] for s in samples]
            colors = [s[2] for s in samples]
            
            self.volume_renderer.update_transfer_functions(
                intensities, opacities, colors, self.intensity_range
            )
            self.vtkWidget.GetRenderWindow().Render()

    def update_opacity_function(self, xs, ys, colors):
        """Update VTK transfer functions - only if point-based is active"""
        if self._active_tf_system != 'point':
            return  # Skip if not active
            
        self.volume_renderer.update_transfer_functions(xs, ys, colors, self.intensity_range)
        
        # Sync the OTHER canvas (not the source)
        if self._tf_change_source == '1d':
            # 1D was source, sync to 2D
            try:
                self.tf2d_canvas.set_tf_state(xs, ys, colors)
            except Exception as e:
                print(f"Error syncing to 2D canvas: {e}")
        elif self._tf_change_source == '2d':
            # 2D was source, sync already done in update_opacity_function_from_2d
            pass
        else:
            # External source (like loading TF), sync both
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

        # Trigger render
        self.vtkWidget.GetRenderWindow().Render()

    # ===== UI CONTROL METHODS =====
    
    def toggle_point_view(self, show_2d):
        """Toggle between 1D and 2D views for point-based TF"""
        idx = 1 if show_2d else 0
        self.point_canvas_container.setCurrentIndex(idx)
        self.point_view_toggle.setText('Switch to 1D TF' if show_2d else 'Switch to 2D TF')
    
        # Sync TF state between views
        if hasattr(self, 'plot_canvas') and hasattr(self, 'tf2d_canvas'):
            xs, ys, colors = self.plot_canvas.points_x, self.plot_canvas.points_y, self.plot_canvas.colors
            if show_2d:  # Switching to 2D view
                self.tf2d_canvas.set_tf_state(xs, ys, colors)
            else:  # Switching to 1D view
                self.plot_canvas.set_tf_state(xs, ys, colors)

    def toggle_widget_view(self, show_1d):
        """Toggle between 1D and 2D views for widget-based TF"""
        if hasattr(self, 'tf_canvas'):
            if show_1d:
                self.tf_canvas.set_tf_type('1d')
                self.widget_view_toggle.setText("Switch to 2D View")
            else:
                self.tf_canvas.set_tf_type('2d')
                self.widget_view_toggle.setText("Switch to 1D View")

    def reset_point_view(self):
        """Reset point-based TF view"""
        try:
            current_widget = self.point_canvas_container.currentWidget()
            if hasattr(current_widget, 'canvas'):
                current_widget.canvas.reset_view()
        except Exception as e:
            print(f"Error resetting point view: {e}")

    def reset_widget_view(self):
        """Reset widget-based TF view"""
        try:
            if hasattr(self, 'tf_canvas'):
                self.tf_canvas.reset_view()
        except Exception as e:
            print(f"Error resetting widget view: {e}")

    # ===== EXISTING METHODS (keep as-is) =====
    
    def update_opacity_function_from_1d(self, xs, ys, colors):
        """Update from 1D canvas - sync to 2D and VTK."""
        if self._tf_change_source == '2d':
            return
        self._tf_change_source = '1d'
        self.update_opacity_function(xs, ys, colors)
        self._tf_change_source = None

    def update_opacity_function_from_2d(self, xs, ys, colors):
        """Update from 2D canvas - sync to 1D and VTK."""
        if self._tf_change_source == '1d':
            return
        self._tf_change_source = '2d'
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.points_x = list(xs)
            self.plot_canvas.points_y = list(ys)
            self.plot_canvas.colors = [tuple(c) for c in colors]
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
        self.update_opacity_function(xs, ys, colors)
        self._tf_change_source = None

    def load_volume_dialog(self):
        """Load volume through file dialog."""
        file_path = self.dataset_loader.load_volume_dialog()
        if file_path:
            try:
                self.load_volume(file_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load Failed", f"Failed to load {file_path}:\n{e}")

    def load_volume(self, file_path):
        """Load and process volume data."""
        image_data, reader, np_scalars, np_gradient = self.dataset_loader.load_volume(file_path)
        (self.normalized_scalars, self.gradient_normalized, 
         self.intensity_range, self.gradient_range) = self.dataset_loader.normalize_data(np_scalars, np_gradient)

        # Store dataset directory for potential nD use
        self.current_dataset_dir = os.path.dirname(file_path)

        self.volume_renderer.set_volume_data(image_data, reader)
        self.update_tf_canvases()  # ← This only updates point-based TF!
        self.volume_renderer.reset_camera()
        self.vtkWidget.GetRenderWindow().Render()

        # DON'T create feature browser here - wait until nD mode is activated
        # This prevents interference with existing systems

        self.image_data = image_data
        self.reader = reader
        return True

    def update_tf_canvases(self):
        """Update TF canvases with new data."""
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

        # UPDATE WIDGET-BASED TF SYSTEM
        if hasattr(self, 'tf_canvas'):
            self.tf_canvas.data = self.normalized_scalars
            self.tf_canvas.gradient_data = self.gradient_normalized
            self.tf_canvas._setup_canvas()  # Force complete refresh
            self.tf_canvas._draw()
            print("✅ Updated widget-based TF")
        
        print(f"📊 New data ranges - Intensity: {self.intensity_range}, Gradient: {self.gradient_range}")

    def reset_widget_tf_for_new_data(self):
        """Completely reset widget TF system for new dataset"""
        if hasattr(self, 'tf_canvas'):
            # Clear existing widgets
            self.tf_canvas.widgets.clear()
        
            # Add a default widget centered in the new data range
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
        
            # Update widget manager
            if hasattr(self, 'widget_manager'):
                self.widget_manager.update_widget_list()

    def toggle_log_histogram(self, state):
        """Toggle logarithmic histogram display."""
        try:
            self.plot_canvas._draw()
        except Exception: pass
        try:
            self.tf2d_canvas._on_log_toggled(state)
        except Exception: pass

    def save_current_tf(self):
        """Save current transfer function."""
        # Save from active system
        if self._active_tf_system == 'point' and hasattr(self, 'plot_canvas'):
            self.tf_manager.save_current_tf(
                self.plot_canvas.points_x,
                self.plot_canvas.points_y,
                self.plot_canvas.colors
            )
        else:
            # TODO: Implement widget TF saving
            print("Widget TF saving not yet implemented")

    def load_selected_tf(self, idx):
        """Load selected transfer function."""
        tf_data = self.tf_manager.load_selected_tf(idx)
        if tf_data:
            xs, ys, colors = tf_data
            # Load into point-based system (widget loading TODO)
            self.update_opacity_function(xs, ys, colors)


# --------------------------- Main ---------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolumeApp()
    window.show()
    sys.exit(app.exec_())