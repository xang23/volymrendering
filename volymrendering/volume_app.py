import sys
import numpy as np

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


class VolumeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('VTK Volume + Interactive 1D/2D Transfer Function (Modular)')
        
        # Initialize components
        self.dataset_loader = DatasetLoader(self)
        self.volume_renderer = VolumeRenderer()
        
        # Track which canvas is the source of changes to prevent loops
        self._tf_change_source = None  # '1d', '2d', or None
        
        self.setup_ui()
        self.setup_data_components()

        # Start with traditional TF, allow toggle to switch
        self.setup_transfer_functions()  # Start with traditional
        # Remove: self.setup_unified_tf()
        #self.setup_unified_tf() #For widgets?
        #self.setup_transfer_functions()

    def toggle_tf_mode(self, widget_mode):
        """Toggle between point-based and widget-based transfer functions"""
        if widget_mode:
            # Switch to widget mode (always 2D)
            self.tf_mode_toggle.setText("Switch to Point Mode")
            self.view_toggle.setVisible(False)  # Hide 1D/2D toggle for widgets
            self.setup_unified_tf()
        else:
            # Switch to point mode
            self.tf_mode_toggle.setText("Switch to Widget Mode") 
            self.view_toggle.setVisible(True)  # Show 1D/2D toggle for traditional
            self.setup_traditional_tf()

    def setup_unified_tf(self):
        """Replace the old 1D/2D TF setup with unified widget-based approach"""
        print("Setting up unified TF with widgets...")
    
        # Clear existing TF setup first
        self.clear_existing_tf()

        # Create unified canvas (always 2D for widgets)
        self.tf_canvas = UnifiedTFCanvas(
            tf_type='2d',
            data=self.normalized_scalars,
            gradient_data=self.gradient_normalized,
            update_callback=self.update_volume_from_widgets
        )
    
        # Create canvas widget
        self.canvas_widget = TFCanvasWidget(self.tf_canvas, self, label='Reset TF View')
        self.canvas_container.addWidget(self.canvas_widget)
    
        # Add widget manager
        self.widget_manager = WidgetManager(self.tf_canvas)
    
        # Find where to insert the widget manager in your layout
        main_layout = self.centralWidget().layout()
        if main_layout:
            main_layout.insertWidget(3, self.widget_manager)
    
        # Add a test widget
        test_widget = WidgetFactory.create_widget(WidgetType.GAUSSIAN, preset='soft_tissue')
        self.tf_canvas.add_widget(test_widget)
    
        # Initialize VTK
        self.interactor.Initialize()
        self.interactor.Start()

    def update_volume_from_widgets(self):
        """Update volume from widget-based TF"""
        samples = self.tf_canvas.sample_for_vtk()
        if samples:
            # Convert to your existing volume renderer format
            intensities = [s[0] for s in samples]
            opacities = [s[1] for s in samples]
            colors = [s[2] for s in samples]
            
            # USE your existing volume renderer method
            self.volume_renderer.update_transfer_functions(
                intensities, opacities, colors, self.intensity_range
            )
            
        self.vtkWidget.GetRenderWindow().Render()
        
    # In VolumeApp - replace the current canvas setup
    def setup_ui(self):
        """Setup the main user interface with dual view"""
        self.frame = QtWidgets.QFrame()
        vlay = QtWidgets.QVBoxLayout(self.frame)

        # VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget.GetRenderWindow().AddRenderer(self.volume_renderer.get_renderer())
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        vlay.addWidget(self.vtkWidget)

        # Toolbar
        toolbar = self.create_toolbar()
        vlay.addLayout(toolbar)

        # DUAL VIEW CONTAINER
        dual_view_container = QtWidgets.QHBoxLayout()
    
        # Left side: Point-based TF
        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(QtWidgets.QLabel("Point-based Transfer Function"))
    
        self.point_canvas_container = QtWidgets.QStackedWidget()
        left_panel.addWidget(self.point_canvas_container)
    
        point_controls = QtWidgets.QHBoxLayout()
        self.point_view_toggle = QtWidgets.QPushButton('Switch to 2D TF')
        self.point_view_toggle.setCheckable(True)
        self.point_view_toggle.toggled.connect(self.toggle_point_view)
        point_controls.addWidget(self.point_view_toggle)
    
        self.point_reset_btn = QtWidgets.QPushButton('Reset Point TF View')
        self.point_reset_btn.clicked.connect(self.reset_point_view)
        point_controls.addWidget(self.point_reset_btn)
    
        left_panel.addLayout(point_controls)
    
        # Right side: Widget-based TF  
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.addWidget(QtWidgets.QLabel("Widget-based Transfer Function"))
    
        self.widget_canvas_container = QtWidgets.QStackedWidget()
        right_panel.addWidget(self.widget_canvas_container)
    
        widget_controls = QtWidgets.QHBoxLayout()
        self.widget_view_toggle = QtWidgets.QPushButton('Switch to 1D View')
        self.widget_view_toggle.setCheckable(True)
        self.widget_view_toggle.toggled.connect(self.toggle_widget_view)
        widget_controls.addWidget(self.widget_view_toggle)
    
        self.widget_reset_btn = QtWidgets.QPushButton('Reset Widget TF View')
        self.widget_reset_btn.clicked.connect(self.reset_widget_view)
        widget_controls.addWidget(self.widget_reset_btn)
    
        right_panel.addLayout(widget_controls)
    
        # Add widget manager to right panel
        self.widget_manager = None  # Will be created in setup
        right_panel.addWidget(QtWidgets.QLabel("Widget Management"))
    
        dual_view_container.addLayout(left_panel)
        dual_view_container.addLayout(right_panel)
    
        vlay.addLayout(dual_view_container)

        self.frame.setLayout(vlay)
        self.setCentralWidget(self.frame)

    def toggle_tf_mode(self, widget_mode):
        """Toggle between point-based and widget-based transfer functions"""
        if widget_mode:
            # Switch to widget mode
            self.tf_mode_toggle.setText("Switch to Point Mode")
            self.view_toggle.setVisible(True)  # Show 1D/2D toggle
            self.setup_unified_tf()
        else:
            # Switch to point mode
            self.tf_mode_toggle.setText("Switch to Widget Mode") 
            self.view_toggle.setVisible(False)  # Hide 1D/2D toggle
            self.setup_traditional_tf()

    def clear_existing_tf(self):
        """Clear existing TF setup"""
        # Remove existing canvas widgets
        for i in reversed(range(self.canvas_container.count())):
            widget = self.canvas_container.widget(i)
            if widget:
                widget.setParent(None)
    
        # Remove widget manager if it exists
        if hasattr(self, 'widget_manager') and self.widget_manager:
            self.widget_manager.setParent(None)
            self.widget_manager = None
        
        # Clear TF canvas reference
        if hasattr(self, 'tf_canvas'):
            self.tf_canvas = None

    def setup_traditional_tf(self):
        """Setup traditional point-based transfer functions"""
        self.clear_existing_tf()
        self.setup_transfer_functions()

    def toggle_tf_view(self, checked):
        """Switch between 1D and 2D TF views (traditional mode only)"""
        if hasattr(self, 'canvas_container') and self.canvas_container.count() > 1:
            # Traditional mode: switch between stacked widgets
            idx = 1 if checked else 0
            self.canvas_container.setCurrentIndex(idx)
            self.view_toggle.setText('Switch to 1D TF' if checked else 'Switch to 2D TF')
        
            # Ensure both canvases show the same TF state when switching
            if hasattr(self, 'plot_canvas') and hasattr(self, 'tf2d_canvas'):
                xs, ys, colors = self.plot_canvas.points_x, self.plot_canvas.points_y, self.plot_canvas.colors
                if checked:  # Switching to 2D view
                    self.tf2d_canvas.set_tf_state(xs, ys, colors)
                else:  # Switching to 1D view
                    self.plot_canvas.set_tf_state(xs, ys, colors)

    def create_toolbar(self):
        """Create the main toolbar with controls."""
        toolbar = QtWidgets.QHBoxLayout()
        
        # Log checkbox
        self.log_checkbox = QtWidgets.QCheckBox('Log Histogram')
        self.log_checkbox.stateChanged.connect(self.toggle_log_histogram)
        toolbar.addWidget(self.log_checkbox)
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

        # Setup both TF systems
        self.setup_point_based_tf()
        self.setup_widget_based_tf()

    def setup_fallback_data(self):
        """Setup fallback data when no volume is loaded."""
        empty = np.zeros((100,), dtype=np.float32)
        self.normalized_scalars = empty
        self.gradient_normalized = empty
        self.intensity_range = (0.0, 1.0)
        self.gradient_range = (0.0, 1.0)

    def setup_transfer_functions(self):
            """Initialize transfer function components."""
            print("\n" + "="*50)
            print("SETTING UP TRADITIONAL TRANSFER FUNCTIONS")
            print("="*50)
    
            # Initialize TF manager FIRST - this loads existing TFs from disk
            self.tf_manager = TFManager(self.tf_selector, self)
    
            # Get initial TF data from manager
            points_x, points_y, colors = self.tf_manager.get_initial_tf_data(self.normalized_scalars)
    
            print(f"Initial TF points: {len(points_x)}")
            print(f"TF selector count: {self.tf_selector.count()}")
    
            # 1D TF canvas
            self.plot_canvas = TransferFunctionPlot(
                self.update_opacity_function_from_1d,
                self.normalized_scalars, 
                self.log_checkbox
            )
    
            # Set the initial points from TF manager
            self.plot_canvas.points_x = points_x
            self.plot_canvas.points_y = points_y  
            self.plot_canvas.colors = colors
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
    
            # Add to stacked widget
            self.tf1d_widget = TFCanvasWidget(self.plot_canvas, parent=self, label='Reset 1D View')
            self.canvas_container.addWidget(self.tf1d_widget)

            # 2D TF canvas
            hist2d, _, _ = np.histogram2d(
                self.normalized_scalars, self.gradient_normalized, 
                bins=(256, 256), range=((0, 255), (0, 255))
            )
            self.tf2d_canvas = TransferFunction2D(
                hist2d, self.intensity_range, self.gradient_range, self.log_checkbox
            )
            # Set up 2D canvas with its own callback
            self.tf2d_canvas.set_tf_state(points_x, points_y, colors)
            self.tf2d_widget = TFCanvasWidget(self.tf2d_canvas, parent=self, label='Reset 2D View')
            self.canvas_container.addWidget(self.tf2d_widget)

            # Start with 1D view
            self.canvas_container.setCurrentIndex(0)
            self.view_toggle.setChecked(False)
            self.view_toggle.setText('Switch to 2D TF')

            # Initialize with the current TF
            self.update_opacity_function(points_x, points_y, colors)

            # Initialize VTK interactor
            self.interactor.Initialize()
            self.interactor.Start()
    
            print("Traditional transfer functions setup complete")
            print("="*50 + "\n")

    def update_opacity_function_from_1d(self, xs, ys, colors):
        """Update from 1D canvas - sync to 2D and VTK."""
        if self._tf_change_source == '2d':
            return  # Ignore if 2D is the source
            
        self._tf_change_source = '1d'
        self.update_opacity_function(xs, ys, colors)
        self._tf_change_source = None

    def update_opacity_function_from_2d(self, xs, ys, colors):
        """Update from 2D canvas - sync to 1D and VTK."""
        if self._tf_change_source == '1d':
            return  # Ignore if 1D is the source
            
        self._tf_change_source = '2d'
        
        # Update 1D canvas
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.points_x = list(xs)
            self.plot_canvas.points_y = list(ys)
            self.plot_canvas.colors = [tuple(c) for c in colors]
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
        
        self.update_opacity_function(xs, ys, colors)
        self._tf_change_source = None

    def update_opacity_function(self, xs, ys, colors):
        """Update VTK transfer functions - common method for both sources."""
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

    # Main application methods
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
        # Load data
        image_data, reader, np_scalars, np_gradient = self.dataset_loader.load_volume(file_path)
        
        # Normalize data
        (self.normalized_scalars, self.gradient_normalized, 
         self.intensity_range, self.gradient_range) = self.dataset_loader.normalize_data(np_scalars, np_gradient)

        # Update volume renderer
        self.volume_renderer.set_volume_data(image_data, reader)
        
        # Update TF canvases
        self.update_tf_canvases()
        
        # Reset camera and render
        self.volume_renderer.reset_camera()
        self.vtkWidget.GetRenderWindow().Render()

        # Store references
        self.image_data = image_data
        self.reader = reader

        return True

    def update_tf_canvases(self):
        """Update TF canvases with new data."""
        # Update 1D canvas
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.hist_data = self.normalized_scalars
            self.plot_canvas._draw()

        # Update 2D canvas
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

    def toggle_tf_view(self, checked):
        """Switch between 1D and 2D TF views."""
        idx = 1 if checked else 0
        self.canvas_container.setCurrentIndex(idx)
        self.view_toggle.setText('Switch to 1D TF' if checked else 'Switch to 2D TF')
        
        # Ensure both canvases show the same TF state when switching
        if hasattr(self, 'plot_canvas') and hasattr(self, 'tf2d_canvas'):
            xs, ys, colors = self.plot_canvas.points_x, self.plot_canvas.points_y, self.plot_canvas.colors
            if checked:  # Switching to 2D view
                self.tf2d_canvas.set_tf_state(xs, ys, colors)
            else:  # Switching to 1D view
                self.plot_canvas.set_tf_state(xs, ys, colors)

    def toggle_log_histogram(self, state):
        """Toggle logarithmic histogram display."""
        try:
            self.plot_canvas._draw()
        except Exception:
            pass
        try:
            self.tf2d_canvas._on_log_toggled(state)
        except Exception:
            pass

    def reset_all_views(self):
        """Reset all canvas views."""
        try:
            self.plot_canvas.reset_view()
            self.tf2d_canvas.reset_view()
        except Exception:
            pass

    # TF management methods
    def save_current_tf(self):
        """Save current transfer function."""
        self.tf_manager.save_current_tf(
            self.plot_canvas.points_x,
            self.plot_canvas.points_y,
            self.plot_canvas.colors
        )

    def load_selected_tf(self, idx):
        """Load selected transfer function."""
        tf_data = self.tf_manager.load_selected_tf(idx)
        if tf_data:
            xs, ys, colors = tf_data
            # Use external source to update both canvases
            self.update_opacity_function(xs, ys, colors)

    def setup_point_based_tf(self):
        """Setup point-based transfer function"""
        print("Setting up point-based TF...")
    
        # Initialize TF manager
        self.tf_manager = TFManager(self.tf_selector, self)
    
        # Get initial TF data
        points_x, points_y, colors = self.tf_manager.get_initial_tf_data(self.normalized_scalars)
    
        # 1D TF canvas
        self.plot_canvas = TransferFunctionPlot(
            self.update_opacity_function_from_1d,
            self.normalized_scalars, 
            self.log_checkbox
        )
        self.plot_canvas.points_x = points_x
        self.plot_canvas.points_y = points_y  
        self.plot_canvas.colors = colors
        self.plot_canvas._sort_points_with_colors()
    
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

def setup_widget_based_tf(self):
    """Setup widget-based transfer function"""
    print("Setting up widget-based TF...")
    
    # Create unified canvas
    self.tf_canvas = UnifiedTFCanvas(
        tf_type='2d',
        data=self.normalized_scalars,
        gradient_data=self.gradient_normalized,
        update_callback=self.update_volume_from_widgets
    )
    
    # Create canvas widget
    self.canvas_widget = TFCanvasWidget(self.tf_canvas, self, label='Reset TF View')
    self.widget_canvas_container.addWidget(self.canvas_widget)
    
    # Add widget manager
    self.widget_manager = WidgetManager(self.tf_canvas)
    
    # Find the widget manager placeholder and replace it
    for i in range(self.frame.layout().count()):
        item = self.frame.layout().itemAt(i)
        if (isinstance(item, QtWidgets.QVBoxLayout) and 
            item.itemAt(3) and  # Assuming 4th item is the label
            item.itemAt(3).widget() and 
            item.itemAt(3).widget().text() == "Widget Management"):
            # Replace the label with actual widget manager
            old_widget = item.itemAt(3).widget()
            item.removeWidget(old_widget)
            old_widget.deleteLater()
            item.insertWidget(3, self.widget_manager)
            break
    
    # Add initial widget
    test_widget = WidgetFactory.create_widget(WidgetType.GAUSSIAN)
    self.tf_canvas.add_widget(test_widget)
    self.widget_manager.update_widget_list()


# --------------------------- Main ---------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VolumeApp()
    window.show()
    sys.exit(app.exec_())