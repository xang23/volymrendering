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
        self._active_tf_system = 'point'  # 'point' or 'widget' - only one active for rendering
        
        # nD features - INITIALIZE BUT DON'T CREATE
        self.feature_browser = None
        self.current_dataset_dir = None
        
        # Widget manager window
        self.widget_manager_window = None
        
        self.setup_ui()
        self.setup_data_components()
        self.setup_dual_transfer_functions()

        #ND
        from nd_widget_manager import NDWidgetManager
        self.nd_manager = NDWidgetManager()
        self.tf_canvas.set_nd_callback(self.on_widget_moved_in_nd)
        self.setup_artifact_analyzer_button()
        
        self.setup_quick_benchmark()

    def test_colored_widget(self):
        """Test with a clearly visible colored widget"""
        if not hasattr(self, 'tf_canvas'):
            return
        
        # Clear widgets
        self.tf_canvas.widgets.clear()
        
        # Create widget with VISIBLE color
        colored_widget = WidgetFactory.create_widget(
            WidgetType.GAUSSIAN,
            center_intensity=30,  # Where data is
            center_gradient=50,
            intensity_std=20,
            gradient_std=30,
            opacity=0.8,
            color=(0.8, 0.2, 0.2),  # BRIGHT RED
            blend_mode='max'
        )
        
        self.tf_canvas.add_widget(colored_widget)
        
        print(f"Created BRIGHT RED widget at data location")
        print(f"   Color: {colored_widget.color}")
        
        self.update_volume_from_widgets()

    def setup_ui(self):
        """Setup the main user interface with dual render views"""
        self.frame = QtWidgets.QFrame()
        self.main_layout = QtWidgets.QVBoxLayout(self.frame)

        # MAIN VERTICAL SPLITTER: Renders on top, TF panels on bottom
        main_splitter = QtWidgets.QSplitter(Qt.Vertical)

        # DUAL RENDER CONTAINER (top part)
        render_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        # Create two VTK renderers
        self.vtkWidget_point = QVTKRenderWindowInteractor()
        self.vtkWidget_point.GetRenderWindow().AddRenderer(self.volume_renderer.get_renderer())

        # Create second renderer for widget-based TF
        from volume_renderer import VolumeRenderer
        self.volume_renderer_widget = VolumeRenderer()
        self.vtkWidget_widget = QVTKRenderWindowInteractor()
        self.vtkWidget_widget.GetRenderWindow().AddRenderer(self.volume_renderer_widget.get_renderer())

        # Set up interactors
        self.interactor_point = self.vtkWidget_point.GetRenderWindow().GetInteractor()
        self.interactor_widget = self.vtkWidget_widget.GetRenderWindow().GetInteractor()

        # Create labeled containers for each renderer
        point_render_container = self.create_render_container(self.vtkWidget_point, "Point-based TF Render")
        widget_render_container = self.create_render_container(self.vtkWidget_widget, "Widget-based TF Render")

        # Add to render splitter
        render_splitter.addWidget(point_render_container)
        render_splitter.addWidget(widget_render_container)

        # DUAL VIEW CONTAINER for TF editors (bottom part)
        tf_splitter = QtWidgets.QSplitter(Qt.Horizontal)

        left_panel = self.setup_point_panel()
        right_panel = self.setup_widget_panel()

        tf_splitter.addWidget(left_panel)
        tf_splitter.addWidget(right_panel)

        # Add both splitters to main splitter
        main_splitter.addWidget(render_splitter)  # Top: renders
        main_splitter.addWidget(tf_splitter)      # Bottom: TF editors

        # Set initial sizes (60% for renders, 40% for TF panels)
        main_splitter.setSizes([600, 400])

        # Toolbar at the very top (not in splitter)
        self.toolbar = self.create_toolbar()

        # Make splitters more visible
        main_splitter.setStyleSheet("QSplitter::handle { background-color: #c0c0c0; }")
        render_splitter.setStyleSheet("QSplitter::handle { background-color: #a0a0a0; }")
        tf_splitter.setStyleSheet("QSplitter::handle { background-color: #a0a0a0; }")

        self.main_layout.addLayout(self.toolbar)
        self.main_layout.addWidget(main_splitter)

        self.frame.setLayout(self.main_layout)
        self.setCentralWidget(self.frame)

        # Set initial window size
        self.resize(1800, 1200)

        # Initialize both renderers
        self.initialize_both_renderers()

    def create_render_container(self, vtk_widget, label_text):
        """Create a labeled container for a renderer"""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        
        # Add label
        label = QtWidgets.QLabel(label_text)
        label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        
        # Add the VTK widget
        layout.addWidget(vtk_widget)
        
        return container

    def initialize_both_renderers(self):
        """Initialize both VTK renderers with the same data"""
        # Both renderers will use the same initial data
        if hasattr(self, 'image_data') and hasattr(self, 'reader'):
            self.volume_renderer_widget.set_volume_data(self.image_data, self.reader)
        
        # Initialize both interactors
        self.interactor_point.Initialize()
        self.interactor_widget.Initialize()
        self.interactor_point.Start()
        self.interactor_widget.Start()

    def setup_point_panel(self):
        """Setup point-based panel as a widget"""
        panel_widget = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel_widget)
    
        panel_layout.addWidget(QtWidgets.QLabel("Point-based Transfer Function"))
    
        self.point_canvas_container = QtWidgets.QStackedWidget()
        panel_layout.addWidget(self.point_canvas_container)
    
        # Controls
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
        """Setup widget-based panel as a widget"""
        panel_widget = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel_widget)

        panel_layout.addWidget(QtWidgets.QLabel("Widget-based Transfer Function"))

        self.widget_canvas_container = QtWidgets.QStackedWidget()
        panel_layout.addWidget(self.widget_canvas_container)

        # Controls
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

        # ADD BUTTON TO OPEN WIDGET MANAGER WINDOW
        self.open_widget_manager_btn = QtWidgets.QPushButton('Open Widget Manager')
        self.open_widget_manager_btn.clicked.connect(self.show_widget_manager)
        controls.addWidget(self.open_widget_manager_btn)

        panel_layout.addLayout(controls)

        return panel_widget

    def create_widget_manager_window(self):
        """Create the floating widget manager window"""
        # Create a separate window for widget manager
        self.widget_manager_window = QtWidgets.QMainWindow(self)
        self.widget_manager_window.setWindowTitle("Widget Manager")
        self.widget_manager_window.setObjectName("WidgetManagerWindow")
    
        # Set window flags to make it a tool window (stays on top of main window)
        self.widget_manager_window.setWindowFlags(
            Qt.Tool | 
            Qt.WindowCloseButtonHint | 
            Qt.WindowMinimizeButtonHint
        )
    
        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.widget_manager_window.setCentralWidget(central_widget)
    
        # Create layout
        layout = QtWidgets.QVBoxLayout(central_widget)
    
        # Create the actual widget manager
        self.widget_manager = WidgetManager(self.tf_canvas)
        layout.addWidget(self.widget_manager)
    
        # Set reasonable size
        self.widget_manager_window.resize(400, 300)
    
        # Connect close event to handle cleanup
        self.widget_manager_window.closeEvent = self.widget_manager_window_close_event

    def widget_manager_window_close_event(self, event):
        """Handle widget manager window closing"""
        # Just hide instead of close to preserve state
        self.widget_manager_window.hide()
        event.ignore()  # Don't actually close, just hide

    def show_widget_manager(self):
        """Show the widget manager window"""
        if not hasattr(self, 'widget_manager_window') or self.widget_manager_window is None:
            self.create_widget_manager_window()
    
        # Position near the main window
        main_window_geometry = self.geometry()
        self.widget_manager_window.move(
            main_window_geometry.right() - 450,  # Offset from right edge
            main_window_geometry.top() + 50      # Offset from top
        )
    
        self.widget_manager_window.show()
        self.widget_manager_window.raise_()  # Bring to front
        self.widget_manager_window.activateWindow()

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

        # ADD WIDGET MANAGER BUTTON TO TOOLBAR
        self.widget_manager_btn = QtWidgets.QPushButton("Widget Manager")
        self.widget_manager_btn.clicked.connect(self.show_widget_manager)
        toolbar.addWidget(self.widget_manager_btn)

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
        points_x, points_y, colors = self.tf_manager.get_initial_tf_data(self.normalized_scalars, data_range=self.intensity_range)

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

        # CREATE WIDGET MANAGER BUT DON'T SHOW IT YET - it will be in separate window
        self.widget_manager = WidgetManager(self.tf_canvas)

        # ===== CREATE SINGLE VISIBLE WIDGET =====
        visible_widget = WidgetFactory.create_widget(
            WidgetType.GAUSSIAN,
            center_intensity=30,      # Where your data is
            center_gradient=50,
            intensity_std=20,
            gradient_std=30,
            opacity=0.8,
            color=(0.8, 0.2, 0.2),    # BRIGHT RED - VISIBLE!
            blend_mode='max'
        )

        # ADD TO ND MANAGER (if available)
        if hasattr(self, 'nd_manager'):
            self.nd_manager.add_widget(visible_widget)
            # Project to current view
            projected = self.nd_manager.project_to_2d('Intensity', 'Gradient')
            for widget in projected:
                self.tf_canvas.add_widget(widget)
        else:
            # Fallback
            self.tf_canvas.add_widget(visible_widget)

        self.widget_manager.update_widget_list()
        print("Created VISIBLE red widget")
        # ========================================

        # Initialize with point-based TF (default active system)
        self.update_opacity_function(points_x, points_y, colors)

        # Initialize VTK
        self.interactor_point.Initialize()
        self.interactor_point.Start()
        self.interactor_widget.Initialize()
        self.interactor_widget.Start()

        print("Dual transfer functions setup complete")
        print("="*50 + "\n")

    def switch_active_system(self, system_name):
        """Switch which TF system is active for rendering - OPTIMIZED"""
        system_type = self.system_selector.currentData()
        self._active_tf_system = system_type
    
        # Update UI indicators
        if system_type == 'point':
            self.point_active_indicator.setText(" ACTIVE")
            self.point_active_indicator.setStyleSheet("color: green; font-weight: bold;")
            self.widget_active_indicator.setText(" INACTIVE")
            self.widget_active_indicator.setStyleSheet("color: gray;")
        
            # Highlight the active render window
            self.highlight_active_render('point')
        
        elif system_type == 'widget':
            self.point_active_indicator.setText(" INACTIVE")
            self.point_active_indicator.setStyleSheet("color: gray;")
            self.widget_active_indicator.setText(" ACTIVE")
            self.widget_active_indicator.setStyleSheet("color: green; font-weight: bold;")
        
            # Highlight the active render window
            self.highlight_active_render('widget')
            
            # Hide feature browser if it exists
            if self.feature_browser:
                self.feature_browser.hide()
            
            # Trigger render with widget-based TF
            self.update_volume_from_widgets()
            
        elif system_type == 'nd':  # nD mode - MATRIX IMPLEMENTATION
            self.point_active_indicator.setText(" INACTIVE")
            self.point_active_indicator.setStyleSheet("color: gray;")
            self.widget_active_indicator.setText(" INACTIVE") 
            self.widget_active_indicator.setStyleSheet("color: gray;")
            
            # ACTIVATE MATRIX MODE
            self.safe_activate_nd_mode()

    def highlight_active_render(self, active_system):
        """Visual highlight to show which render window is active"""
        # Reset both first
        self.reset_render_highlights()
    
        if active_system == 'point':
            # Highlight point-based render window (e.g., green border)
            self.vtkWidget_point.setStyleSheet("border: 3px solid green;")
        elif active_system == 'widget':
            # Highlight widget-based render window  
            self.vtkWidget_widget.setStyleSheet("border: 3px solid green;")

    def reset_render_highlights(self):
        """Remove highlights from both render windows"""
        self.vtkWidget_point.setStyleSheet("border: 1px solid gray;")
        self.vtkWidget_widget.setStyleSheet("border: 1px solid gray;")

    def safe_activate_nd_mode(self):
        """Activate nD mode with matrix browser"""
        print("Activating Feature Matrix mode...")
    
        if not hasattr(self, 'all_features') or not self.all_features:
            print("Cannot activate nD mode: no features loaded")
            self.system_selector.setCurrentIndex(1)
            return
    
        try:
            # Use already loaded features
            feature_data = self.all_features
        
            if self.feature_browser is None:
                print("Creating feature matrix...")
                from simple_feature_browser import SimpleMatrixBrowser
            
                self.feature_browser = SimpleMatrixBrowser(
                    feature_data_dict=feature_data,
                    update_callback=self.load_projection
                )
            
                # Add to layout
                if hasattr(self, 'main_layout'):
                    if self.feature_browser.parent():
                        self.feature_browser.setParent(None)
                    self.main_layout.insertWidget(2, self.feature_browser)
            else:
                # Just update existing browser
                self.feature_browser.feature_data = feature_data
                self.feature_browser.update_matrix()
        
            # Update nD manager with all features
            self.nd_manager.update_features(list(feature_data.keys()))
        
            # Migrate existing widgets to nD
            if hasattr(self, 'tf_canvas'):
                for widget in self.tf_canvas.widgets:
                    if widget not in self.nd_manager.widgets:
                        self.nd_manager.add_widget(widget)
        
            # Show matrix
            self.feature_browser.show()
            print(f"Feature Matrix activated with {len(feature_data)} features")
        
            # Load initial projection (Intensity/Gradient if available)
            if 'Intensity' in feature_data and 'Gradient' in feature_data:
                self.load_projection('Intensity', 'Gradient')
            else:
                # Load first two features
                keys = list(feature_data.keys())
                self.load_projection(keys[0], keys[1])
        
        except Exception as e:
            print(f"Failed to activate nD mode: {e}")
            import traceback
            traceback.print_exc()
            self.system_selector.setCurrentIndex(1)

    def on_matrix_cell_clicked(self, feature_x, feature_y):
        """When user clicks a cell in the matrix - FIXED VERSION"""
        print(f"Loading into main TF: {feature_x} vs {feature_y}")
    
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
        
            print(f"Canvas updated with {feature_x} vs {feature_y}")
            print(f"New ranges: intensity={self.tf_canvas.intensity_range}, gradient={self.tf_canvas.gradient_range}")
        
            # Update volume rendering
            self.update_volume_from_widgets()
        
        except Exception as e:
            print(f"Error updating main TF: {e}")
            import traceback
            traceback.print_exc()

    def update_volume_from_widgets(self):
        """Update volume from widget-based TF with PROPER SCALING"""
        if self._active_tf_system not in ['widget', 'nd']:
            print(f"Widget system not active ({self._active_tf_system})")
            return

        # Get samples
        samples = self.tf_canvas.sample_for_vtk()

        # DEBUG: Check what we got
        print(f"Got {len(samples) if samples else 0} samples")
        print(f"Has cached gradient? {hasattr(self.tf_canvas, '_cached_gradient_opacity')}")

        if samples:
            intensities = [s[0] for s in samples]  # 0-255 values
            opacities = [s[1] for s in samples]
            colors = [s[2] for s in samples]
        
            # DEBUG: Show what widgets produced
            print(f"Widget samples: intensity={min(intensities)}-{max(intensities)} "
                  f"opacity={min(opacities):.3f}-{max(opacities):.3f}")
    
            # SCALE intensities from 0-255 to actual range
            int_min, int_max = self.intensity_range
            scaled_intensities = []
            for i in intensities:
                scaled = int_min + (i / 255.0) * (int_max - int_min)
                scaled_intensities.append(scaled)
        
            print(f"Scaled intensities: {min(intensities)}-{max(intensities)} → "
                  f"{min(scaled_intensities):.1f}-{max(scaled_intensities):.1f}")
        
            # SCALE gradient opacity too
            gradient_opacities = None
            if hasattr(self.tf_canvas, '_cached_gradient_opacity'):
                gradient_op = self.tf_canvas._cached_gradient_opacity
        
                # Check if it's a numpy array or dict
                if isinstance(gradient_op, np.ndarray):
                    print(f"Gradient opacity is numpy array with shape: {gradient_op.shape}")
                
                    # SCALE gradient opacity!
                    grad_min, grad_max = self.gradient_range
                    gradient_opacities = []
                    non_zero_count = 0
                
                    for g in range(0, 256, 4):  # Sample every 4th
                        if gradient_op[g] > 0.01:
                            # SCALE gradient from 0-255 to actual range
                            scaled_g = grad_min + (g / 255.0) * (grad_max - grad_min)
                            gradient_opacities.append((scaled_g, gradient_op[g]))
                            non_zero_count += 1
                
                    print(f"Extracted {non_zero_count} gradient points (scaled)")
                    if gradient_opacities:
                        print(f"Scaled gradient range: {grad_min:.1f}-{grad_max:.1f}")
            
                elif isinstance(gradient_op, dict):
                    print(f"Gradient opacity is dict with {len(gradient_op)} entries")
                    # Scale dict entries
                    grad_min, grad_max = self.gradient_range
                    gradient_opacities = []
                    for g, opacity in gradient_op.items():
                        scaled_g = grad_min + (g / 255.0) * (grad_max - grad_min)
                        gradient_opacities.append((scaled_g, opacity))
                else:
                    print(f"Unknown gradient opacity type: {type(gradient_op)}")
            else:
                print(f"No cached gradient opacity found!")

            # Update renderer - DEBUG output
            print(f"Updating widget renderer with {len(scaled_intensities)} intensity points")
            if gradient_opacities:
                print(f"And {len(gradient_opacities)} gradient opacity points")
        
            # Check if opacities are reasonable
            if max(opacities) < 0.05:
                print(f"WARNING: Maximum opacity is very low ({max(opacities):.3f})")
                print(f"Widgets may not be visible. Try increasing widget opacity.")
    
            # DEBUG: Show first few scaled points
            print(f"First 3 scaled points to VTK:")
            for i in range(min(3, len(scaled_intensities))):
                print(f"   {i}: Intensity={scaled_intensities[i]:.1f} "
                      f"(was {intensities[i]} 0-255), Opacity={opacities[i]:.3f}")

            # Pass SCALED values
            self.volume_renderer_widget.update_transfer_functions(
                scaled_intensities,  # ← SCALED!
                opacities, 
                colors, 
                self.intensity_range,
                gradient_opacities,  # ← SCALED!
                self.gradient_range
            )

            # Render
            self.vtkWidget_widget.GetRenderWindow().Render()
            print("Render complete")

    def update_opacity_function(self, xs, ys, colors):
        """Update point-based TF - this should still work"""
        # DEBUG: Check what point-based TF is sending
        print(f"Point-based TF sending {len(xs)} points")
        print(f"   First point: {xs[0]:.1f}, color: {colors[0]}")
        print(f"   Last point: {xs[-1]:.1f}, color: {colors[-1]}")
        if self._active_tf_system != 'point':
            return
            """Update point-based TF"""
        print(f"Point-based TF sending {len(xs)} points")
        print(f"   Points:")
        for i in range(len(xs)):
            print(f"     {i}: Intensity={xs[i]:.1f}, Opacity={ys[i]:.3f}, Color={colors[i]}")
    
        # Use the NEW method signature
        self.volume_renderer.update_transfer_functions(
            xs, ys, colors, 
            self.intensity_range,  # Pass intensity_range
            None,  # No gradient opacity for point-based
            None   # No gradient range for point-based
        )

        # Sync the OTHER canvas (not the source)
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

        # ONLY render the point window
        self.vtkWidget_point.GetRenderWindow().Render()

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

    def update_opacity_function_from_1d(self, xs, ys, colors):
        """Update from 1D canvas - CONVERT from display to actual range!"""
        print(f"1D TF callback: Received {len(xs)} points in DISPLAY coordinates (0-255)")
    
        # Convert from 0-255 display to actual data range
        int_min, int_max = self.intensity_range
        actual_xs = []
    
        for display_x in xs:
            # Convert 0-255 → actual data range
            actual_x = int_min + (display_x / 255.0) * (int_max - int_min)
            actual_xs.append(actual_x)
    
        print(f"   Display range: {min(xs):.1f}-{max(xs):.1f}")
        print(f"   Actual range: {min(actual_xs):.1f}-{max(actual_xs):.1f}")
    
        # Now pass ACTUAL values to update_opacity_function
        if self._tf_change_source == '2d':
            return
        self._tf_change_source = '1d'
        self.update_opacity_function(actual_xs, ys, colors)  # ← ACTUAL values!
        self._tf_change_source = None

    def update_opacity_function_from_2d(self, xs, ys, colors):
        """Update from 2D canvas - CONVERT if needed"""
        print(f"2D TF callback: Received {len(xs)} points")
    
        # Check if xs are in display coordinates (0-255)
        if max(xs) <= 255.0:
            # Convert to actual range
            int_min, int_max = self.intensity_range
            actual_xs = []
            for display_x in xs:
                actual_x = int_min + (display_x / 255.0) * (int_max - int_min)
                actual_xs.append(actual_x)
            xs = actual_xs
    
        if self._tf_change_source == '1d':
            return
        self._tf_change_source = '2d'
    
        # Update 1D canvas with DISPLAY coordinates
        if hasattr(self, 'plot_canvas'):
            # Convert actual → display for 1D canvas
            display_xs = []
            for actual_x in xs:
                display_x = (actual_x - int_min) / (int_max - int_min) * 255.0
                display_xs.append(display_x)
        
            self.plot_canvas.points_x = display_xs  # 0-255 for display!
            self.plot_canvas.points_y = ys
            self.plot_canvas.colors = [tuple(c) for c in colors]
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
    
        self.update_opacity_function(xs, ys, colors)  # ACTUAL values for VTK
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
        """Load and process volume data for BOTH renderers"""
        # FIX: Now receives 3 values from new dataset_loader
        image_data, reader, all_features = self.dataset_loader.load_volume(file_path)
    
        # Extract primary data (Intensity) and Gradient
        if 'Intensity' in all_features:
            np_scalars = all_features['Intensity']
        else:
            # Take first feature as primary
            first_key = list(all_features.keys())[0]
            np_scalars = all_features[first_key]
    
        if 'Gradient' in all_features:
            np_gradient = all_features['Gradient']
        else:
            np_gradient = np.zeros_like(np_scalars, dtype=np.float32)
    
        # Normalize data
        (self.normalized_scalars, self.gradient_normalized, 
         self.intensity_range, self.gradient_range) = self.dataset_loader.normalize_data(np_scalars, np_gradient)

        self.current_dataset_dir = os.path.dirname(file_path)
    
        # Store all features for nD mode
        self.all_features = all_features

        # Set volume data for BOTH renderers
        self.volume_renderer.set_volume_data(image_data, reader)
        self.volume_renderer_widget.set_volume_data(image_data, reader)

        # Update ALL TF systems
        self.update_tf_canvases()

        # Reset widget positions for new data range
        self.reset_widget_tf_for_new_data()

        # Reset cameras for BOTH renderers
        self.volume_renderer.reset_camera()
        self.volume_renderer_widget.reset_camera()

        # Render BOTH windows
        self.vtkWidget_point.GetRenderWindow().Render()
        self.vtkWidget_widget.GetRenderWindow().Render()

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
            print("Updated widget-based TF")
        
        print(f"New data ranges - Intensity: {self.intensity_range}, Gradient: {self.gradient_range}")

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
                self.plot_canvas.colors,
                data_range=self.intensity_range  # ← ADD THIS!
            )
        else:
            # TODO: Implement widget TF saving
            print("Widget TF saving not yet implemented")

    def load_selected_tf(self, idx):
        """Load selected transfer function into point-based system only"""
        tf_data = self.tf_manager.load_selected_tf(
            idx, 
            current_data_range=self.intensity_range
        )
    
        # FIX: Check if tf_data is None
        if tf_data is None:
            print("Failed to load TF, keeping current")
            return
        
        # Now safe to unpack
        xs, ys, colors = tf_data
    
        # Load into point-based system
        if hasattr(self, 'plot_canvas'):
            self.plot_canvas.points_x = xs
            self.plot_canvas.points_y = ys
            self.plot_canvas.colors = colors
            self.plot_canvas._sort_points_with_colors()
            self.plot_canvas._draw()
    
        # Update point renderer with the loaded TF
        self.volume_renderer.update_transfer_functions(
            xs, ys, colors, self.intensity_range
        )
    
        # Render both windows
        self.vtkWidget_point.GetRenderWindow().Render()
        self.vtkWidget_widget.GetRenderWindow().Render()

    def closeEvent(self, event):
        """Handle main window closing"""
        if hasattr(self, 'widget_manager_window') and self.widget_manager_window:
            self.widget_manager_window.close()
        event.accept()

    def on_widget_moved_in_nd(self, widget_2d, feat_x, feat_y, new_x, new_y):
        """When widget moves in 2D projection, update nD coordinates"""
        self.nd_manager.update_nd_position(widget_2d, new_x, new_y)

    def load_projection(self, feat_x, feat_y):
        """Load a specific 2D projection into the main view"""
        print(f"Loading projection: {feat_x} vs {feat_y}")
    
        try:
            # Get data for these features
            if hasattr(self, 'feature_browser') and self.feature_browser:
                feature_data = self.feature_browser.feature_data
                data_x = feature_data[feat_x]
                data_y = feature_data[feat_y]
            
                # Store current projection
                self.active_projection = (feat_x, feat_y)
            
                # Update canvas data (these are ACTUAL data values, not normalized!)
                self.tf_canvas.raw_data_x = data_x
                self.tf_canvas.raw_data_y = data_y
            
                # ===== FIXED: Use normalize_single =====
                self.tf_canvas.data = self.dataset_loader.normalize_single(data_x)
                self.tf_canvas.gradient_data = self.dataset_loader.normalize_single(data_y)
                # ======================================
            
                # Store ranges for this projection
                self.tf_canvas.intensity_range = (float(np.min(data_x)), 
                                                 float(np.max(data_x)))
                self.tf_canvas.gradient_range = (float(np.min(data_y)), 
                                                float(np.max(data_y)))
            
                # Set projection info
                self.tf_canvas.set_projection_features(feat_x, feat_y)
            
                # Update nD manager with ranges for these features
                if hasattr(self, 'nd_manager'):
                    self.nd_manager.feature_ranges[feat_x] = self.tf_canvas.intensity_range
                    self.nd_manager.feature_ranges[feat_y] = self.tf_canvas.gradient_range
            
                # Clear and load PROJECTED widgets
                self.tf_canvas.clear_widgets()
                projected = self.nd_manager.project_to_2d(feat_x, feat_y)
                for widget in projected:
                    self.tf_canvas.add_widget(widget)
            
                # Update canvas
                self.tf_canvas._setup_canvas()
                self.tf_canvas._draw()
            
                # Update volume
                self.update_volume_from_widgets()
            
        except Exception as e:
            print(f"Error loading projection: {e}")
            import traceback
            traceback.print_exc()

    # Also make sure you have on_matrix_cell_clicked that calls this:
    def on_matrix_cell_clicked(self, feature_x, feature_y):
        """When user clicks a cell in the matrix"""
        print(f"Matrix cell clicked: {feature_x} vs {feature_y}")
        self.load_projection(feature_x, feature_y)  # ← Calls the new method

   
    def open_feature_popup(self, feat_x, feat_y):
        """Open popup window for feature pair"""
        print(f"Opening popup: {feat_x} vs {feat_y}")
    
        try:
            # Get data for these features
            if hasattr(self, 'feature_browser') and self.feature_browser:
                feature_data = self.feature_browser.feature_data
                data_x = feature_data[feat_y] #Swapped to see better data arches.
                data_y = feature_data[feat_x]
            
                # Create popup
                from nd_popup import NDFeaturePopup
                popup = NDFeaturePopup(
                    feat_y, feat_x, #Swapped for data arches
                    data_x, data_y,
                    self.nd_manager,
                    parent=self
                )
            
                # Position near the clicked cell
                self.position_popup_near_cell(popup, feat_x, feat_y)
            
                # Show popup
                popup.show()
                popup.raise_()
            
                # Store reference (optional, for tracking)
                if not hasattr(self, 'open_popups'):
                    self.open_popups = []
                self.open_popups.append(popup)
            
                # Clean up when closed
                popup.destroyed.connect(lambda: self.open_popups.remove(popup))
    
        except Exception as e:
            print(f"Error opening popup: {e}")
            import traceback
            traceback.print_exc()

    def position_popup_near_cell(self, popup, feat_x, feat_y):
        """Position popup near the clicked matrix cell"""
        if not hasattr(self, 'feature_browser'):
            return
    
        # Get matrix widget
        matrix = self.feature_browser.matrix_widget
    
        # Find the cell position (simplified - you might need to calculate this)
        row = self.feature_browser.feature_names.index(feat_y)
        col = self.feature_browser.feature_names.index(feat_x)
    
        # Get cell widget
        cell = matrix.layout().itemAtPosition(row+1, col+1).widget()
        if cell:
            # Get global position
            global_pos = cell.mapToGlobal(cell.rect().topRight())
            popup.move(global_pos.x() + 10, global_pos.y())

    def setup_quick_benchmark(self):
        """Add a simple benchmark button to your toolbar"""
        
        # Add to existing toolbar (simplest!)
        self.benchmark_btn = QtWidgets.QPushButton("Benchmark Current TF")
        self.benchmark_btn.clicked.connect(self.run_quick_benchmark)
        self.toolbar.addWidget(self.benchmark_btn)  # Add to your existing toolbar
        
        # Small status label
        self.benchmark_status = QtWidgets.QLabel("Ready")
        self.toolbar.addWidget(self.benchmark_status)

    def run_quick_benchmark(self):
        """Run a quick 100-frame benchmark on current widgets"""
        import time
        import csv
        from datetime import datetime
        
        self.benchmark_status.setText("Running...")
        QtWidgets.QApplication.processEvents()
        
        # Get current widget info
        widget_info = []
        for i, w in enumerate(self.nd_manager.widgets):
            widget_info.append(f"{w.widget_type.value}({w.center_intensity:.0f},{w.center_gradient:.0f})")
        
        # Run 100 renders
        times = []
        for i in range(100):
            start = time.perf_counter()
            self.vtkWidget_widget.GetRenderWindow().Render()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        # Calculate stats
        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time
        
        # Save screenshot
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.take_screenshot(filename)
        
        # Save to CSV
        with open('benchmark_results.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['Date', 'Widgets', 'Avg(ms)', 'FPS', 'Screenshot'])
            writer.writerow([datetime.now().isoformat(), '+'.join(widget_info), 
                            f"{avg_time:.2f}", f"{fps:.1f}", filename])
        
        # Show result
        self.benchmark_status.setText(f"FPS: {fps:.1f} | {avg_time:.2f}ms")
        
        # Optional: popup with details
        QtWidgets.QMessageBox.information(self, "Benchmark Complete", 
            f"Average: {avg_time:.2f} ms\nFPS: {fps:.1f}\nScreenshot saved")

    def take_screenshot(self, filename):
        """Save current render view to file"""
        w = self.vtkWidget_widget.GetRenderWindow()
        image = vtk.vtkWindowToImageFilter()
        image.SetInput(w)
        image.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(image.GetOutputPort())
        writer.Write()

    def setup_artifact_analyzer_button(self):
        """Add button to open artifact analyzer"""
        self.analyzer_btn = QtWidgets.QPushButton("🔬 Artifact Analyzer")
        self.analyzer_btn.clicked.connect(self.open_artifact_analyzer)
        self.toolbar.addWidget(self.analyzer_btn)

    def open_artifact_analyzer(self):
        """Open the artifact analyzer as a separate window"""
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