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
            x_label=self.feat_x,
            y_label=self.feat_y
        )

        self.tf_canvas.raw_intensity_range = self.x_range
        self.tf_canvas.raw_gradient_range = self.y_range
        self.tf_canvas.intensity_range = (0, 255)
        self.tf_canvas.gradient_range = (0, 255)

        self.tf_canvas.set_projection_features(self.feat_x, self.feat_y)
        self.tf_canvas.set_nd_callback(self.on_widget_moved)

        canvas_wrapper = TFCanvasWidget(self.tf_canvas, self, label='Reset View')
        left_layout.addWidget(canvas_wrapper)

        main_layout.addWidget(left_widget, 2)

        # MIDDLE: Simple Widget Controls (not the full WidgetManager)
        middle_widget = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_widget)

        controls_title = QtWidgets.QLabel("<h3>Widget Controls</h3>")
        controls_title.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(controls_title)

        # Simple list of widgets
        self.widget_list = QtWidgets.QListWidget()
        self.widget_list.itemSelectionChanged.connect(self.on_widget_selected)
        middle_layout.addWidget(self.widget_list)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
    
        add_btn = QtWidgets.QPushButton("➕ Add Widget")
        add_btn.clicked.connect(self.add_widget)
        button_layout.addWidget(add_btn)
    
        delete_btn = QtWidgets.QPushButton("❌ Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_widget)
        button_layout.addWidget(delete_btn)
    
        clear_btn = QtWidgets.QPushButton("🗑 Clear All")
        clear_btn.clicked.connect(self.clear_all_widgets)
        button_layout.addWidget(clear_btn)
    
        middle_layout.addLayout(button_layout)

        # Parameter controls (simple version)
        self.param_group = QtWidgets.QGroupBox("Widget Parameters")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_group.setLayout(self.param_layout)
        self.param_group.setVisible(False)
        middle_layout.addWidget(self.param_group)

        # Sync buttons
        sync_layout = QtWidgets.QHBoxLayout()
        sync_to_nd_btn = QtWidgets.QPushButton("🔄 Sync to 3D")
        sync_to_nd_btn.clicked.connect(self.sync_to_nd)
        sync_layout.addWidget(sync_to_nd_btn)
    
        load_from_nd_btn = QtWidgets.QPushButton("📥 Load from 3D")
        load_from_nd_btn.clicked.connect(self.load_from_nd)
        sync_layout.addWidget(load_from_nd_btn)
    
        middle_layout.addLayout(sync_layout)

        # Test buttons
        middle_layout.addStretch()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        middle_layout.addWidget(close_btn)

        main_layout.addWidget(middle_widget, 1)
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
        print(f"Widget moved: ({new_x:.1f}, {new_y:.1f}) for features ({feat_x}, {feat_y})")
    
        # Update the widget's display position (for the canvas)
        widget_2d.center_intensity = new_x
        widget_2d.center_gradient = new_y
    
        # CRITICAL: Update BOTH features in nd_coords
        # feat_x is the X-axis feature (Gradient in your example)
        # feat_y is the Y-axis feature (Laplacian in your example)
        widget_2d.nd_coords[feat_x] = new_x
        widget_2d.nd_coords[feat_y] = new_y
    
        # Also update the nd_manager (redundant but safe)
        self.nd_manager.update_nd_position(widget_2d, new_x, new_y, feat_x, feat_y)
    
        # Update the widget list display
        self.update_widget_list()
    
        # AUTO-SYNC to 3D renderer
        self.sync_to_nd()
    
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

    def setup_widget_manager_sync(self):
        """Setup widget manager with nd_manager synchronization"""
        # Store original methods
        original_add_widget = self.widget_manager.add_preset_widget
        original_clear_widgets = self.widget_manager.clear_btn.clicked
        original_duplicate_widget = self.widget_manager.duplicate_widget
        original_delete_widget = self.widget_manager.delete_widget
    
        # Override with sync versions
        self.widget_manager.add_preset_widget = self.sync_add_preset_widget
        self.widget_manager.duplicate_widget = self.sync_duplicate_widget
    
        # Connect clear button to sync version
        self.widget_manager.clear_btn.clicked.disconnect()
        self.widget_manager.clear_btn.clicked.connect(self.sync_clear_widgets)
    
        # Store reference to original delete method
        self.original_delete_widget = self.widget_manager.delete_widget
        self.widget_manager.delete_widget = self.sync_delete_widget

    def sync_add_preset_widget(self):
        """Add widget to both canvas and nd_manager"""
        # Get widget data from preset combo
        widget_data = self.widget_manager.preset_combo.currentData()
        widget_type_str, preset_name = widget_data
    
        from widget_factory import WidgetType, WidgetFactory
        widget_type = WidgetType(widget_type_str)
        new_widget = WidgetFactory.create_widget(widget_type, preset=preset_name)
    
        # Add to nd_manager first (master)
        self.nd_manager.add_widget(new_widget)
    
        # Then add to canvas
        self.tf_canvas.add_widget(new_widget)
    
        # Update widget list
        self.widget_manager.update_widget_list()
    
        # Refresh render
        self.update_render_view()
        print(f"✅ Added preset widget and synced with nd_manager")

    def sync_clear_widgets(self):
        """Clear all widgets from both canvas and nd_manager"""
        # Clear nd_manager
        self.nd_manager.widgets.clear()
    
        # Clear canvas
        self.tf_canvas.clear_widgets()
    
        # Update widget list
        self.widget_manager.update_widget_list()
    
        # Refresh render
        self.update_render_view()
        print(f"✅ Cleared all widgets and synced with nd_manager")

    def sync_duplicate_widget(self):
        """Duplicate widget in both canvas and nd_manager"""
        selected_items = self.widget_manager.widget_list.selectedItems()
        if selected_items and self.widget_manager.current_widget:
            import copy
            new_widget = copy.copy(self.widget_manager.current_widget)
            # Offset slightly so they don't overlap
            new_widget.center_intensity = min(255, new_widget.center_intensity + 10)
            new_widget.center_gradient = min(255, new_widget.center_gradient + 10)
        
            # Add to nd_manager
            self.nd_manager.add_widget(new_widget)
        
            # Add to canvas
            self.tf_canvas.add_widget(new_widget)
        
            # Update widget list
            self.widget_manager.update_widget_list()
        
            # Refresh render
            self.update_render_view()
            print(f"✅ Duplicated widget and synced with nd_manager")

    def sync_delete_widget(self, widget):
        """Delete widget from both canvas and nd_manager"""
        if widget in self.tf_canvas.widgets:
            # Remove from nd_manager
            self.nd_manager.remove_widget(widget)
        
            # Remove from canvas
            self.tf_canvas.remove_widget(widget)
        
            # Update widget list
            self.widget_manager.update_widget_list()
        
            # Hide parameters panel
            self.widget_manager.param_group.setVisible(False)
            self.widget_manager.current_widget = None
        
            # Refresh render
            self.update_render_view()
            print(f"✅ Deleted widget and synced with nd_manager")

    def update_widget_list(self):
        """Update the widget list display"""
        self.widget_list.clear()
        for i, widget in enumerate(self.tf_canvas.widgets):
            item = QtWidgets.QListWidgetItem(f"{i+1}. {widget.widget_type.value} (X:{widget.center_intensity:.0f}, Y:{widget.center_gradient:.0f})")
            item.setData(Qt.UserRole, i)
            self.widget_list.addItem(item)

    def on_widget_selected(self):
        """Handle widget selection"""
        selected = self.widget_list.selectedItems()
        if not selected:
            self.param_group.setVisible(False)
            return
    
        idx = selected[0].data(Qt.UserRole)
        self.current_widget = self.tf_canvas.widgets[idx]
        self.tf_canvas.active_widget = idx
        self.tf_canvas._draw()
    
        # Update parameter controls
        self.update_parameter_controls()

    def update_parameter_controls(self):
        """Update parameter controls for selected widget"""
        # Clear existing
        for i in reversed(range(self.param_layout.count())):
            self.param_layout.itemAt(i).widget().setParent(None)
    
        if not self.current_widget:
            return
    
        # Add position controls
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(0, 255)
        self.x_spin.setValue(self.current_widget.center_intensity)
        self.x_spin.valueChanged.connect(self.on_x_changed)
        self.param_layout.addRow("X Position:", self.x_spin)
    
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(0, 255)
        self.y_spin.setValue(self.current_widget.center_gradient)
        self.y_spin.valueChanged.connect(self.on_y_changed)
        self.param_layout.addRow("Y Position:", self.y_spin)
    
        # Opacity control
        self.opacity_spin = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setRange(0, 1)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self.current_widget.opacity)
        self.opacity_spin.valueChanged.connect(self.on_opacity_changed)
        self.param_layout.addRow("Opacity:", self.opacity_spin)
    
        # Color button
        self.color_btn = QtWidgets.QPushButton("Change Color")
        self.color_btn.clicked.connect(self.change_color)
        self.param_layout.addRow("Color:", self.color_btn)
        self.update_color_button()
    
        self.param_group.setVisible(True)

    def on_x_changed(self, value):
        if self.current_widget:
            self.current_widget.center_intensity = value
            self.tf_canvas._draw()
            # AUTO-SYNC to 3D renderer
            self.sync_to_nd()
            self.update_render_view()

    def on_y_changed(self, value):
        if self.current_widget:
            self.current_widget.center_gradient = value
            self.tf_canvas._draw()
            # AUTO-SYNC to 3D renderer
            self.sync_to_nd()
            self.update_render_view()

    def on_opacity_changed(self, value):
        if self.current_widget:
            self.current_widget.opacity = value
            self.tf_canvas._draw()
            # AUTO-SYNC to 3D renderer
            self.sync_to_nd()
            self.update_render_view()

    def change_color(self):
        if self.current_widget:
            qcolor = QtWidgets.QColorDialog.getColor()
            if qcolor.isValid():
                self.current_widget.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                self.update_color_button()
                self.tf_canvas._draw()
                # AUTO-SYNC to 3D renderer
                self.sync_to_nd()
                self.update_render_view()

    def update_color_button(self):
        if hasattr(self, 'color_btn') and self.current_widget:
            r, g, b = self.current_widget.color
            self.color_btn.setStyleSheet(f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});")

    def add_widget(self):
        """Add a new widget"""
        from widget_factory import WidgetFactory, WidgetType
    
        new_widget = WidgetFactory.create_widget(
            WidgetType.GAUSSIAN,
            center_intensity=128,
            center_gradient=128,
            intensity_std=25,
            gradient_std=25,
            opacity=0.8,
            color=(0.8, 0.2, 0.2)
        )
    
        # AUTO-SYNC to nd_manager and 3D renderer
        self.nd_manager.add_widget(new_widget)
        self.tf_canvas.add_widget(new_widget)
        self.update_widget_list()
        self.update_render_view()

    def delete_selected_widget(self):
        """Delete selected widget"""
        if hasattr(self, 'current_widget') and self.current_widget:
            self.tf_canvas.remove_widget(self.current_widget)
            self.current_widget = None
            self.update_widget_list()
            self.param_group.setVisible(False)
            # AUTO-SYNC to 3D renderer
            self.sync_to_nd()
            self.update_render_view()
            

    def clear_all_widgets(self):
        """Clear all widgets"""
        self.tf_canvas.clear_widgets()
        self.current_widget = None
        self.update_widget_list()
        self.param_group.setVisible(False)
        # AUTO-SYNC to 3D renderer
        self.sync_to_nd()
        self.update_render_view()

    def sync_to_nd(self):
        """Sync current widgets to nd_manager"""
        self.nd_manager.widgets.clear()
        for widget in self.tf_canvas.widgets:
            self.nd_manager.add_widget(widget)
        self.update_render_view()
        print(f"✅ Synced {len(self.tf_canvas.widgets)} widgets to nD renderer")

    def load_from_nd(self):
        """Load widgets from nd_manager"""
        self.tf_canvas.clear_widgets()
        projected = self.nd_manager.project_to_2d(self.feat_x, self.feat_y)
        for widget in projected:
            self.tf_canvas.add_widget(widget)
        self.update_widget_list()
        self.update_render_view()
        print(f"✅ Loaded {len(projected)} widgets from nD manager")