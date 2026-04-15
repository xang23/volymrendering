import traceback
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from unified_tf_canvas import UnifiedTFCanvas
from tf_canvas_widget import TFCanvasWidget
from nd_shader_renderer import NDShaderRenderer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import time
import vtk
import os
from datetime import datetime

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
    
        print(f"Actual features in point data: {self.feature_names}")
        
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

        # MIDDLE: Simple Widget Controls
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
    
        add_btn = QtWidgets.QPushButton("Add Widget")
        add_btn.clicked.connect(self.add_widget)
        button_layout.addWidget(add_btn)
    
        delete_btn = QtWidgets.QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected_widget)
        button_layout.addWidget(delete_btn)
    
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all_widgets)
        button_layout.addWidget(clear_btn)
    
        middle_layout.addLayout(button_layout)

        # Parameter controls
        self.param_group = QtWidgets.QGroupBox("Widget Parameters")
        self.param_layout = QtWidgets.QFormLayout()
        self.param_group.setLayout(self.param_layout)
        self.param_group.setVisible(False)
        middle_layout.addWidget(self.param_group)

        # Sync buttons
        sync_layout = QtWidgets.QHBoxLayout()
        sync_to_nd_btn = QtWidgets.QPushButton("Sync to 3D")
        sync_to_nd_btn.clicked.connect(self.sync_to_nd)
        sync_layout.addWidget(sync_to_nd_btn)
    
        load_from_nd_btn = QtWidgets.QPushButton("Load from 3D")
        load_from_nd_btn.clicked.connect(self.load_from_nd)
        sync_layout.addWidget(load_from_nd_btn)
    
        middle_layout.addLayout(sync_layout)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        middle_layout.addWidget(close_btn)

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
            print(f"\nCreating NDShaderRenderer for popup...")
            self.mc_renderer = NDShaderRenderer(
                self.image_data,
                self.feature_names,
                self.nd_manager,
                f"popup_{self.feat_x}_{self.feat_y}"
            )
            print(f"NDShaderRenderer created successfully")
            
            print(f"Adding renderer to VTK widget...")
            ren_win = self.vtk_widget.GetRenderWindow()
            ren_win.AddRenderer(self.mc_renderer.get_renderer())
            print(f"Renderer added to VTK widget")
            
            right_layout.addWidget(self.vtk_widget)
            print(f"VTK widget added to layout")
            
            self.vtk_widget.show()
            self.vtk_widget.Initialize()

            # Test simple render first
            print(f"\nApplying widget-based TF for {self.feat_x} vs {self.feat_y}...")
            self.mc_renderer.set_feature_pair(self.feat_x, self.feat_y)
            ren_win.Render()
            print(f"Widget-based render complete")
            
        except Exception as e:
            print(f"CRASH in renderer creation: {e}")
            traceback.print_exc()
            self.mc_renderer = None
            error_label = QtWidgets.QLabel(f"Render error: {str(e)}")
            error_label.setStyleSheet("color: red; font-size: 14px;")
            right_layout.addWidget(error_label)
            right_layout.addWidget(self.vtk_widget)
            self.vtk_widget.show()

        main_layout.addWidget(right_widget, 2)
        
        # Setup widget tester
        self.setup_widget_tester()
        self.setup_woodgrain_demo()

    def force_real_red(self):
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.force_volume_visible()

    def load_projected_widgets(self):
        self.tf_canvas.clear_widgets()
        
        projected = self.nd_manager.project_to_2d(self.feat_x, self.feat_y)
        
        for widget in projected:
            self.tf_canvas.add_widget(widget)
        
        print(f"Loaded {len(projected)} widgets into popup")
        self.update_render_view()
    
    def on_widget_moved(self, widget_2d, feat_x, feat_y, new_x, new_y):
        print(f"Widget moved: ({new_x:.1f}, {new_y:.1f}) for features ({feat_x}, {feat_y})")
    
        widget_2d.center_intensity = new_x
        widget_2d.center_gradient = new_y
    
        widget_2d.nd_coords[feat_x] = new_x
        widget_2d.nd_coords[feat_y] = new_y
    
        self.nd_manager.update_nd_position(widget_2d, new_x, new_y, feat_x, feat_y)
        self.update_widget_list()
        self.sync_to_nd()
    
    def update_render_view(self):
        print(f"Updating render view for {self.feat_x} vs {self.feat_y}")

        if not hasattr(self, 'mc_renderer') or self.mc_renderer is None:
            print(f"   No renderer available")
            return

        try:
            all_widgets = self.nd_manager.widgets
            print(f"   Got {len(all_widgets)} widgets from nd_manager")
        
            intensities = []
            opacities = []
            colors = []
            gradient_values = []
            gradient_opacities = []
        
            for widget in all_widgets:
                x_val = widget.nd_coords.get(self.feat_x, 128)
                y_val = widget.nd_coords.get(self.feat_y, 128)
            
                intensities.append(x_val)
                opacities.append(widget.opacity)
                colors.append(widget.color)
                gradient_values.append(y_val)
                gradient_opacities.append(widget.opacity)
        
            print(f"   Built TF with {len(intensities)} widgets")
        
            if intensities:
                self.mc_renderer.update_transfer_functions(
                    intensities, opacities, colors,
                    self.x_range, gradient_values, gradient_opacities, self.y_range
                )
            else:
                self.mc_renderer.update_transfer_functions(
                    [0, 255], [0, 0], [(0.5, 0.5, 0.5)], self.x_range
                )
        
            self.mc_renderer.set_feature_pair(self.feat_x, self.feat_y)
            self.vtk_widget.GetRenderWindow().Render()
            print(f"   Render updated with {len(intensities)} active widgets")
    
        except Exception as e:
            print(f"   Error in update_render_view: {e}")
            traceback.print_exc()
    
    def closeEvent(self, event):
        if hasattr(self, 'vtk_widget'):
            self.vtk_widget.close()
        event.accept()

    def update_widget_list(self):
        self.widget_list.clear()
        for i, widget in enumerate(self.tf_canvas.widgets):
            item = QtWidgets.QListWidgetItem(f"{i+1}. {widget.widget_type.value} (X:{widget.center_intensity:.0f}, Y:{widget.center_gradient:.0f})")
            item.setData(Qt.UserRole, i)
            self.widget_list.addItem(item)

    def on_widget_selected(self):
        selected = self.widget_list.selectedItems()
        if not selected:
            self.param_group.setVisible(False)
            return
    
        idx = selected[0].data(Qt.UserRole)
        self.current_widget = self.tf_canvas.widgets[idx]
        self.tf_canvas.active_widget = idx
        self.tf_canvas._draw()
        
        # SYNC: Uppdatera test panel med denna widgets info
        self.test_x.setValue(int(self.current_widget.center_intensity))
        self.test_y.setValue(int(self.current_widget.center_gradient))
        
        # Identifiera shape
        shape = self.identify_widget_shape(self.current_widget)
        self.test_widget_type.setCurrentText(shape)
        self.browse_label.setText(shape)
        
        self.update_parameter_controls()

    def identify_widget_shape(self, widget):
        """Identifiera vilken shape en widget har"""
        if hasattr(widget, 'intensity_radius'):
            return 'Ellipsoid'
        elif hasattr(widget, 'direction'):
            return 'Triangular'
        elif hasattr(widget, 'intensity_width') and hasattr(widget, 'gradient_height'):
            # Kolla om det är Gaussian (har intensity_std) eller Rectangular/Diamond
            if hasattr(widget, 'intensity_std'):
                return 'Gaussian'
            else:
                return 'Rectangular'
        return 'Gaussian'

    def update_parameter_controls(self):
        for i in reversed(range(self.param_layout.count())):
            self.param_layout.itemAt(i).widget().setParent(None)
    
        if not self.current_widget:
            return
    
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
    
        self.opacity_spin = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setRange(0, 1)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self.current_widget.opacity)
        self.opacity_spin.valueChanged.connect(self.on_opacity_changed)
        self.param_layout.addRow("Opacity:", self.opacity_spin)
    
        self.color_btn = QtWidgets.QPushButton("Change Color")
        self.color_btn.clicked.connect(self.change_color)
        self.param_layout.addRow("Color:", self.color_btn)
        self.update_color_button()
    
        self.param_group.setVisible(True)

    def on_x_changed(self, value):
        if self.current_widget:
            self.current_widget.center_intensity = value
            self.tf_canvas._draw()
            self.sync_to_nd()
            self.update_render_view()

    def on_y_changed(self, value):
        if self.current_widget:
            self.current_widget.center_gradient = value
            self.tf_canvas._draw()
            self.sync_to_nd()
            self.update_render_view()

    def on_opacity_changed(self, value):
        if self.current_widget:
            self.current_widget.opacity = value
            self.tf_canvas._draw()
            self.sync_to_nd()
            self.update_render_view()

    def change_color(self):
        if self.current_widget:
            qcolor = QtWidgets.QColorDialog.getColor()
            if qcolor.isValid():
                self.current_widget.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                self.update_color_button()
                self.tf_canvas._draw()
                self.sync_to_nd()
                self.update_render_view()

    def update_color_button(self):
        if hasattr(self, 'color_btn') and self.current_widget:
            r, g, b = self.current_widget.color
            self.color_btn.setStyleSheet(f"background-color: rgb({int(r*255)},{int(g*255)},{int(b*255)});")

    def add_widget(self):
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
    
        self.nd_manager.add_widget(new_widget)
        self.tf_canvas.add_widget(new_widget)
        self.update_widget_list()
        self.update_render_view()

    def delete_selected_widget(self):
        if hasattr(self, 'current_widget') and self.current_widget:
            self.tf_canvas.remove_widget(self.current_widget)
            self.current_widget = None
            self.update_widget_list()
            self.param_group.setVisible(False)
            self.sync_to_nd()
            self.update_render_view()

    def clear_all_widgets(self):
        self.tf_canvas.clear_widgets()
        self.current_widget = None
        self.update_widget_list()
        self.param_group.setVisible(False)
        self.sync_to_nd()
        self.update_render_view()

    def sync_to_nd(self):
        self.nd_manager.widgets.clear()
        for widget in self.tf_canvas.widgets:
            self.nd_manager.add_widget(widget)
        self.update_render_view()
        print(f"Synced {len(self.tf_canvas.widgets)} widgets to nD renderer")

    def load_from_nd(self):
        self.tf_canvas.clear_widgets()
        projected = self.nd_manager.project_to_2d(self.feat_x, self.feat_y)
        for widget in projected:
            self.tf_canvas.add_widget(widget)
        self.update_widget_list()
        self.update_render_view()
        print(f"Loaded {len(projected)} widgets from nD manager")

    def setup_widget_tester(self):
        """Lägg till widget tester i popupen"""
        # Hitta middle_layout
        middle_widget = None
        for child in self.centralWidget().children():
            if isinstance(child, QtWidgets.QWidget):
                for subchild in child.children():
                    if hasattr(subchild, 'layout') and subchild.layout():
                        if subchild.layout().count() > 2:
                            middle_widget = subchild
                            break
                if middle_widget:
                    break

        if not middle_widget:
            return

        tester_group = QtWidgets.QGroupBox("Widget Performance Test")
        tester_layout = QtWidgets.QVBoxLayout()

        # Widget type selection
        type_layout = QtWidgets.QHBoxLayout()
        type_layout.addWidget(QtWidgets.QLabel("Shape:"))
        self.test_widget_type = QtWidgets.QComboBox()
        self.test_widget_type.addItems(['Gaussian', 'Rectangular', 'Triangular', 'Ellipsoid', 'Diamond'])
        self.test_widget_type.currentTextChanged.connect(self.on_test_shape_changed)
        type_layout.addWidget(self.test_widget_type)
        tester_layout.addLayout(type_layout)

        # Position - SYNC med canvas
        pos_layout = QtWidgets.QHBoxLayout()
        pos_layout.addWidget(QtWidgets.QLabel("X:"))
        self.test_x = QtWidgets.QSpinBox()
        self.test_x.setRange(0, 255)
        self.test_x.setValue(128)
        self.test_x.valueChanged.connect(self.on_test_position_changed)
        pos_layout.addWidget(self.test_x)
        pos_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.test_y = QtWidgets.QSpinBox()
        self.test_y.setRange(0, 255)
        self.test_y.setValue(128)
        self.test_y.valueChanged.connect(self.on_test_position_changed)
        pos_layout.addWidget(self.test_y)
        tester_layout.addLayout(pos_layout)

        # ===== LÄGG TILL RENDER KONTROLLER =====
        render_group = QtWidgets.QGroupBox("Display Adjustment")
        render_layout = QtWidgets.QVBoxLayout()
    
        # Opacity boost
        boost_layout = QtWidgets.QHBoxLayout()
        boost_layout.addWidget(QtWidgets.QLabel("Opacity Boost:"))
        self.opacity_boost = QtWidgets.QDoubleSpinBox()
        self.opacity_boost.setRange(0.5, 2.5)
        self.opacity_boost.setSingleStep(0.1)
        self.opacity_boost.setValue(1.0)
        self.opacity_boost.setToolTip("Multiply overall opacity (1.0 = normal)")
        self.opacity_boost.valueChanged.connect(self.on_render_settings_changed)
        boost_layout.addWidget(self.opacity_boost)
        render_layout.addLayout(boost_layout)
    
        # Gamma correction
        gamma_layout = QtWidgets.QHBoxLayout()
        gamma_layout.addWidget(QtWidgets.QLabel("Gamma (brightness):"))
        self.gamma = QtWidgets.QDoubleSpinBox()
        self.gamma.setRange(0.3, 2.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1.0)
        self.gamma.setToolTip("Lower values brighten dark areas")
        self.gamma.valueChanged.connect(self.on_render_settings_changed)
        gamma_layout.addWidget(self.gamma)
        render_layout.addLayout(gamma_layout)
    
        render_group.setLayout(render_layout)
        tester_layout.addWidget(render_group)
        # =====================================

        # Apply button - uppdaterar nuvarande widget
        apply_btn = QtWidgets.QPushButton("Apply to Current Widget")
        apply_btn.clicked.connect(self.apply_test_to_current_widget)
        tester_layout.addWidget(apply_btn)

        # Test button
        self.test_btn = QtWidgets.QPushButton("Test This Widget")
        self.test_btn.clicked.connect(self.test_current_widget)
        tester_layout.addWidget(self.test_btn)

        # Test all button
        self.test_all_btn = QtWidgets.QPushButton("Test All Widget Shapes")
        self.test_all_btn.clicked.connect(self.test_all_widget_shapes)
        tester_layout.addWidget(self.test_all_btn)

        # Quick browse buttons
        browse_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.prev_browse_widget)
        browse_layout.addWidget(prev_btn)

        self.browse_label = QtWidgets.QLabel("Gaussian")
        self.browse_label.setAlignment(Qt.AlignCenter)
        browse_layout.addWidget(self.browse_label)

        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(self.next_browse_widget)
        browse_layout.addWidget(next_btn)
        tester_layout.addLayout(browse_layout)

        # FPS display
        self.fps_display = QtWidgets.QLabel("FPS: --")
        self.fps_display.setAlignment(Qt.AlignCenter)
        self.fps_display.setStyleSheet("font-weight: bold; color: #00ff00;")
        tester_layout.addWidget(self.fps_display)

        # Screenshot button
        screenshot_btn = QtWidgets.QPushButton("Take Screenshot")
        screenshot_btn.clicked.connect(self.take_screenshot)
        tester_layout.addWidget(screenshot_btn)

        tester_group.setLayout(tester_layout)
        middle_widget.layout().addWidget(tester_group)

        # Initiera browse
        self.browse_shapes = ['Gaussian', 'Rectangular', 'Triangular', 'Ellipsoid', 'Diamond']
        self.browse_index = 0

    def on_render_settings_changed(self):
        """När render inställningar ändras, applicera direkt"""
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.display_boost = self.opacity_boost.value()
            self.mc_renderer.display_gamma = self.gamma.value()
            self.update_render_view()
            self.fps_display.setText(f"Boost: {self.opacity_boost.value():.1f}, Gamma: {self.gamma.value():.1f}")

    def on_test_shape_changed(self, shape):
        """När test shape ändras"""
        self.fps_display.setText(f"Selected: {shape} (click Apply to use)")

    def on_test_position_changed(self):
        """När test position ändras"""
        x = self.test_x.value()
        y = self.test_y.value()
        self.fps_display.setText(f"Position: ({x},{y})")

    def apply_test_to_current_widget(self):
        """Applicera test shape och position på nuvarande vald widget"""
        if hasattr(self, 'current_widget') and self.current_widget:
            # Uppdatera position
            self.current_widget.center_intensity = self.test_x.value()
            self.current_widget.center_gradient = self.test_y.value()
        
            # Uppdatera shape - VIKTIGT: skapa en ny widget istället för att försöka ändra typ
            shape = self.test_widget_type.currentText()
            x = self.test_x.value()
            y = self.test_y.value()
        
            # Skapa ny widget med rätt shape
            new_widget = self.create_test_widget(shape, x, y)
        
            # Behåll färg och opacitet från gamla widgeten
            new_widget.color = self.current_widget.color
            new_widget.opacity = self.current_widget.opacity
        
            # Byt ut widgeten i canvas och nd_manager
            idx = self.tf_canvas.widgets.index(self.current_widget)
            self.tf_canvas.widgets[idx] = new_widget
            self.nd_manager.widgets[idx] = new_widget
        
            # Uppdatera current_widget referens
            self.current_widget = new_widget
            self.tf_canvas.active_widget = idx
        
            # Uppdatera canvas
            self.tf_canvas._draw()
        
            # Synca till render
            self.sync_to_nd()
            self.update_render_view()
        
            # Uppdatera widget list display
            self.update_widget_list()
        
            self.fps_display.setText(f"Updated widget to {shape} at ({x},{y})")
        else:
            QtWidgets.QMessageBox.warning(self, "No Widget", "Please select a widget first")



    def prev_browse_widget(self):
        """Bläddra till föregående widget shape och applicera direkt"""
        self.browse_index = (self.browse_index - 1) % len(self.browse_shapes)
        shape = self.browse_shapes[self.browse_index]
        self.browse_label.setText(shape)
        self.test_widget_type.setCurrentText(shape)
    
        if hasattr(self, 'current_widget') and self.current_widget:
            # Skapa ny widget med ny shape
            x = self.current_widget.center_intensity
            y = self.current_widget.center_gradient
            new_widget = self.create_test_widget(shape, x, y)
        
            # Behåll färg och opacitet
            new_widget.color = self.current_widget.color
            new_widget.opacity = self.current_widget.opacity
        
            # Byt ut
            idx = self.tf_canvas.widgets.index(self.current_widget)
            self.tf_canvas.widgets[idx] = new_widget
            self.nd_manager.widgets[idx] = new_widget
            self.current_widget = new_widget
            self.tf_canvas.active_widget = idx
        
            self.tf_canvas._draw()
            self.sync_to_nd()
            self.update_render_view()
            self.update_widget_list()
            self.fps_display.setText(f"Changed to {shape}")

    def next_browse_widget(self):
        """Bläddra till nästa widget shape och applicera direkt"""
        self.browse_index = (self.browse_index + 1) % len(self.browse_shapes)
        shape = self.browse_shapes[self.browse_index]
        self.browse_label.setText(shape)
        self.test_widget_type.setCurrentText(shape)
    
        if hasattr(self, 'current_widget') and self.current_widget:
            # Skapa ny widget med ny shape
            x = self.current_widget.center_intensity
            y = self.current_widget.center_gradient
            new_widget = self.create_test_widget(shape, x, y)
        
            # Behåll färg och opacitet
            new_widget.color = self.current_widget.color
            new_widget.opacity = self.current_widget.opacity
        
            # Byt ut
            idx = self.tf_canvas.widgets.index(self.current_widget)
            self.tf_canvas.widgets[idx] = new_widget
            self.nd_manager.widgets[idx] = new_widget
            self.current_widget = new_widget
            self.tf_canvas.active_widget = idx
        
            self.tf_canvas._draw()
            self.sync_to_nd()
            self.update_render_view()
            self.update_widget_list()
            self.fps_display.setText(f"Changed to {shape}")

    def create_test_widget(self, shape, x, y):
        from widget_factory import WidgetFactory, WidgetType

        shape_map = {
            'Gaussian': WidgetType.GAUSSIAN,
            'Rectangular': WidgetType.RECTANGULAR,
            'Triangular': WidgetType.TRIANGULAR,
            'Ellipsoid': WidgetType.ELLIPSOID,
            'Diamond': WidgetType.DIAMOND
        }

        base_params = {
            'center_intensity': x,
            'center_gradient': y,
            'opacity': 0.8,
            'color': (1.0, 1.0, 1.0),
            'blend_mode': 'max'
        }

        # Shape-specifika parametrar - ANVÄND RÄTT PARAMETRAR!
        if shape == 'Gaussian':
            # Gaussian använder intensity_std och gradient_std
            params = {**base_params, 'intensity_std': 30, 'gradient_std': 30}
        elif shape == 'Rectangular':
            params = {**base_params, 'intensity_width': 60, 'gradient_height': 60}
        elif shape == 'Triangular':
            params = {**base_params, 'intensity_width': 60, 'gradient_height': 60, 'direction': 'symmetric'}
        elif shape == 'Ellipsoid':
            params = {**base_params, 'intensity_radius': 30, 'gradient_radius': 40}
        elif shape == 'Diamond':
            params = {**base_params, 'intensity_width': 60, 'gradient_height': 60}
        else:
            params = base_params

        return WidgetFactory.create_widget(shape_map[shape], **params)

    def measure_fps(self):
        if not hasattr(self, 'mc_renderer') or self.mc_renderer is None:
            return 0
    
        times = []
        for i in range(30):
            start = time.perf_counter()
            self.vtk_widget.GetRenderWindow().Render()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
        avg_time = sum(times) / len(times)
        fps = 1000 / avg_time
    
        self.fps_display.setText(f"FPS: {fps:.1f} | {avg_time:.2f}ms")
        return fps

    def test_current_widget(self):
        original_widgets = self.nd_manager.widgets.copy()
    
        shape = self.test_widget_type.currentText()
        x = self.test_x.value()
        y = self.test_y.value()
        test_widget = self.create_test_widget(shape, x, y)
    
        self.nd_manager.widgets.clear()
        self.nd_manager.add_widget(test_widget)
        self.load_projected_widgets()
        self.update_render_view()
    
        fps = self.measure_fps()
    
        self.nd_manager.widgets = original_widgets
        self.load_projected_widgets()
        self.update_render_view()
    
        QtWidgets.QMessageBox.information(self, "Test Result", 
            f"{shape} widget at ({x},{y})\nFPS: {fps:.1f}")

    def test_all_widget_shapes(self):
        results = {}
        original_widgets = self.nd_manager.widgets.copy()
    
        shapes = ['Gaussian', 'Rectangular', 'Triangular', 'Ellipsoid', 'Diamond']
        x = self.test_x.value()
        y = self.test_y.value()
    
        for shape in shapes:
            print(f"\nTesting {shape}...")
        
            test_widget = self.create_test_widget(shape, x, y)
        
            self.nd_manager.widgets.clear()
            self.nd_manager.add_widget(test_widget)
            self.load_projected_widgets()
            self.update_render_view()
        
            times = []
            for i in range(30):
                start = time.perf_counter()
                self.vtk_widget.GetRenderWindow().Render()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
            avg_time = sum(times) / len(times)
            fps = 1000 / avg_time
            results[shape] = fps
        
            print(f"   {shape}: {fps:.1f} FPS ({avg_time:.2f}ms)")
    
        self.nd_manager.widgets = original_widgets
        self.load_projected_widgets()
        self.update_render_view()
    
        msg = "Widget Performance Results:\n\n"
        for shape, fps in sorted(results.items(), key=lambda x: x[1], reverse=True):
            msg += f"{shape}: {fps:.1f} FPS\n"
        msg += f"\nTested at position ({x},{y})"
        
        QtWidgets.QMessageBox.information(self, "Performance Results", msg)

    def take_screenshot(self):
        """Take screenshot of current rendering"""
        # Skapa screenshots mapp om den inte finns
        screenshot_dir = "screenshots"
        if not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        
        shape = self.test_widget_type.currentText()
        x = self.test_x.value()
        y = self.test_y.value()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshot_dir, f"screenshot_{self.feat_x}_{self.feat_y}_{shape}_{x}_{y}_{timestamp}.png")
        
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self.vtk_widget.GetRenderWindow())
        w2if.Update()
        
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        
        self.fps_display.setText(f"Screenshot: {filename}")
        print(f"Screenshot saved: {filename}")

    def setup_woodgrain_demo(self):
        """Add woodgrain artifact demonstration controls"""
    
        # Find the tester group
        tester_group = None
        for child in self.centralWidget().children():
            if isinstance(child, QtWidgets.QWidget):
                for subchild in child.children():
                    if isinstance(subchild, QtWidgets.QGroupBox) and "Widget Performance Test" in subchild.title():
                        tester_group = subchild
                        break
                if tester_group:
                    break
    
        if not tester_group:
            return
    
        # Add sampling control section
        sampling_group = QtWidgets.QGroupBox("Woodgrain Artifact Control")
        sampling_layout = QtWidgets.QVBoxLayout()
    
        # Sampling rate slider
        rate_layout = QtWidgets.QHBoxLayout()
        rate_layout.addWidget(QtWidgets.QLabel("Sampling Rate:"))
        self.sampling_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.sampling_slider.setRange(10, 100)
        self.sampling_slider.setValue(20)  # 2x default (20 = 2.0)
        self.sampling_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sampling_slider.valueChanged.connect(self.on_sampling_changed)
        rate_layout.addWidget(self.sampling_slider)
    
        self.sampling_label = QtWidgets.QLabel("2.0x (Nyquist)")
        rate_layout.addWidget(self.sampling_label)
        sampling_layout.addLayout(rate_layout)
    
        # Pre-integration toggle
        self.preint_checkbox = QtWidgets.QCheckBox("Use Pre-Integrated Classification (High Quality)")
        self.preint_checkbox.setChecked(False)
        self.preint_checkbox.stateChanged.connect(self.on_preintegration_toggled)
        sampling_layout.addWidget(self.preint_checkbox)
    
        # Compare button
        compare_btn = QtWidgets.QPushButton("Compare: Low vs High Quality")
        compare_btn.clicked.connect(self.compare_woodgrain_quality)
        sampling_layout.addWidget(compare_btn)
    
        sampling_group.setLayout(sampling_layout)
        tester_group.layout().addWidget(sampling_group)

    def compare_woodgrain_quality(self):
        """Compare low vs high sampling quality"""
        if not hasattr(self, 'mc_renderer') or self.mc_renderer is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No renderer available")
            return
    
        # Spara nuvarande widget och inställningar
        original_shape = self.test_widget_type.currentText()
        original_sampling = self.sampling_slider.value()
    
        # Använd en skarp widget för att maximera woodgrain effekten
        self.test_widget_type.setCurrentText('Rectangular')
        self.apply_test_to_current_widget()
    
        results = []
    
        # Test 1: Låg sampling (visar woodgrain)
        self.sampling_slider.setValue(10)  # 1.0x
        QtWidgets.QApplication.processEvents()
        time.sleep(0.5)
        self.take_screenshot()
        results.append("1.0x sampling (woodgrain visible)")
    
        # Test 2: Nyquist sampling (minimum för anti-aliasing)
        self.sampling_slider.setValue(20)  # 2.0x
        QtWidgets.QApplication.processEvents()
        time.sleep(0.5)
        self.take_screenshot()
        results.append("2.0x sampling (Nyquist)")
    
        # Test 3: Hög sampling (slät)
        self.sampling_slider.setValue(40)  # 4.0x
        QtWidgets.QApplication.processEvents()
        time.sleep(0.5)
        self.take_screenshot()
        results.append("4.0x sampling (smooth)")
    
        # Återställ original inställningar
        self.test_widget_type.setCurrentText(original_shape)
        self.apply_test_to_current_widget()
        self.sampling_slider.setValue(original_sampling)
    
        msg = "Woodgrain Artifact Comparison:\n\n"
        msg += "Lower sampling = more woodgrain artifacts\n"
        msg += "Higher sampling = smoother surfaces but lower FPS\n\n"
        for r in results:
            msg += f"- {r}\n"
        msg += "\nScreenshots saved in 'screenshots' folder"
    
        QtWidgets.QMessageBox.information(self, "Woodgrain Demo", msg)

    def on_sampling_changed(self, value):
        """Change sampling rate to show/hide woodgrain artifacts"""
        rate = value / 10.0
        self.sampling_label.setText(f"{rate:.1f}x")
    
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            # Lower sampling = more woodgrain, Higher sampling = smoother
            self.mc_renderer.set_sampling_rate(rate)
            self.update_render_view()
        
            # Update FPS display
            self.measure_fps()

    def on_preintegration_toggled(self, state):
        """Toggle pre-integrated classification"""
        if hasattr(self, 'mc_renderer') and self.mc_renderer:
            self.mc_renderer.use_preintegration = (state == Qt.Checked)
            self.update_render_view()
            self.measure_fps()