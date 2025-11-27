# widget_manager_ui.py
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from widget_factory import WidgetFactory, WidgetType

class WidgetManager(QtWidgets.QWidget):
    def __init__(self, tf_canvas, parent=None):
        super().__init__(parent)
        try:
            self.tf_canvas = tf_canvas
            self.current_widget = None
            self.param_controls = {}
        
            # Verify we have a valid canvas
            if not hasattr(self.tf_canvas, 'widgets'):
                print("⚠️ Warning: tf_canvas may not be properly initialized")
            
            self.setup_ui()
            self.update_widget_list()
        
            print(f"✅ WidgetManager ready with {len(self.tf_canvas.widgets) if hasattr(self.tf_canvas, 'widgets') else 'unknown'} widgets")
        
        except Exception as e:
            print(f"❌ Error initializing WidgetManager: {e}")
            # Create a fallback UI even if initialization fails
            error_label = QtWidgets.QLabel(f"Widget Manager Error: {str(e)}")
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(error_label)
            self.setLayout(layout)
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Header
        header_label = QtWidgets.QLabel("Widget Management")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        layout.addWidget(header_label)
        
        # Quick preset section
        preset_group = QtWidgets.QGroupBox("Quick Presets")
        preset_layout = QtWidgets.QHBoxLayout()
        
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Soft Tissue", ('gaussian', 'soft_tissue'))
        self.preset_combo.addItem("Bone", ('gaussian', 'bone'))
        self.preset_combo.addItem("Vessels", ('gaussian', 'vessels'))
        self.preset_combo.addItem("Custom Gaussian", ('gaussian', None))
        self.preset_combo.addItem("Custom Triangular", ('triangular', None))
        self.preset_combo.addItem("Custom Rectangular", ('rectangular', None))
        self.preset_combo.addItem("Custom Ellipsoid", ('ellipsoid', None))
        self.preset_combo.addItem("Custom Diamond", ('diamond', None))
        preset_layout.addWidget(self.preset_combo)
        
        self.add_preset_btn = QtWidgets.QPushButton("Add Widget")
        self.add_preset_btn.clicked.connect(self.add_preset_widget)
        preset_layout.addWidget(self.add_preset_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # Active widgets list
        list_group = QtWidgets.QGroupBox("Active Widgets")
        list_layout = QtWidgets.QVBoxLayout()
        
        self.widget_list = QtWidgets.QListWidget()
        self.widget_list.itemSelectionChanged.connect(self.on_widget_selected)
        list_layout.addWidget(self.widget_list)
        
        # List controls
        list_controls = QtWidgets.QHBoxLayout()
        self.clear_btn = QtWidgets.QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.tf_canvas.clear_widgets)
        list_controls.addWidget(self.clear_btn)
        
        self.duplicate_btn = QtWidgets.QPushButton("Duplicate")
        self.duplicate_btn.clicked.connect(self.duplicate_widget)
        list_controls.addWidget(self.duplicate_btn)
        
        list_layout.addLayout(list_controls)
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Enhanced parameter controls with scroll area
        self.param_group = QtWidgets.QGroupBox("Widget Parameters")
        self.param_scroll = QtWidgets.QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setMinimumHeight(200)  # Ensure it's tall enough
        self.param_content = QtWidgets.QWidget()
        self.param_layout = QtWidgets.QGridLayout(self.param_content)
        self.param_scroll.setWidget(self.param_content)
        self.param_group.setLayout(QtWidgets.QVBoxLayout())
        self.param_group.layout().addWidget(self.param_scroll)
        self.param_group.setVisible(False)
        layout.addWidget(self.param_group)
        
        self.setLayout(layout)
        
    def create_parameter_control(self, param_name, param_config, row):
        """Create parameter controls with proper type handling"""
        label = QtWidgets.QLabel(param_name.replace('_', ' ').title() + ":")
        self.param_layout.addWidget(label, row, 0)
        
        if param_config['type'] == 'slider':
            min_val, max_val = param_config['range']
            
            # Determine if this is an integer or float parameter
            is_integer = isinstance(min_val, int) and isinstance(max_val, int)
            
            if is_integer:
                # Integer parameters - use slider + spinbox
                slider = QtWidgets.QSlider(Qt.Horizontal)
                slider.setRange(min_val, max_val)
                slider.setValue(int(param_config['value']))
                slider.valueChanged.connect(
                    lambda value, p=param_name: self.on_slider_changed(p, value)
                )
                self.param_layout.addWidget(slider, row, 1)
                
                spinbox = QtWidgets.QSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setValue(int(param_config['value']))
                spinbox.valueChanged.connect(
                    lambda value, p=param_name, s=slider: self.on_int_spinbox_changed(p, value, s)
                )
                self.param_layout.addWidget(spinbox, row, 2)
                
                value_label = QtWidgets.QLabel(str(int(param_config['value'])))
                self.param_layout.addWidget(value_label, row, 3)
                
                return {'slider': slider, 'spinbox': spinbox, 'label': value_label, 'is_integer': True}
            else:
                # Float parameters - use only spinbox
                spinbox = QtWidgets.QDoubleSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setValue(param_config['value'])
                spinbox.setSingleStep(param_config.get('step', 0.01))
                spinbox.valueChanged.connect(
                    lambda value, p=param_name: self.on_float_spinbox_changed(p, value)
                )
                self.param_layout.addWidget(spinbox, row, 1, 1, 2)
                
                value_label = QtWidgets.QLabel(f"{param_config['value']:.2f}")
                self.param_layout.addWidget(value_label, row, 3)
                
                return {'spinbox': spinbox, 'label': value_label, 'is_integer': False}
            
        elif param_config['type'] == 'combo':
            # Combo box for options
            combo = QtWidgets.QComboBox()
            for option in param_config['options']:
                combo.addItem(option)
            combo.setCurrentText(param_config['value'])
            combo.currentTextChanged.connect(
                lambda value, p=param_name: self.on_combo_changed(p, value)
            )
            self.param_layout.addWidget(combo, row, 1, 1, 2)
            return {'combo': combo}
            
        return None

    # SEPARATE CALLBACK METHODS FOR EACH CONTROL TYPE
    def on_slider_changed(self, param_name, value):
        """Handle slider changes (integer parameters only)"""
        if self.current_widget:
            self.current_widget.set_parameter(param_name, int(value))
            self.update_ui_label(param_name, int(value))
            self.tf_canvas._draw()
            self.tf_canvas._notify_app()

    def on_int_spinbox_changed(self, param_name, value, slider):
        """Handle integer spinbox changes"""
        if self.current_widget:
            self.current_widget.set_parameter(param_name, int(value))
            slider.blockSignals(True)
            slider.setValue(int(value))
            slider.blockSignals(False)
            self.update_ui_label(param_name, int(value))
            self.tf_canvas._draw()
            self.tf_canvas._notify_app()

    def on_float_spinbox_changed(self, param_name, value):
        """Handle float spinbox changes"""
        if self.current_widget:
            self.current_widget.set_parameter(param_name, float(value))
            self.update_ui_label(param_name, float(value))
            self.tf_canvas._draw()
            self.tf_canvas._notify_app()

    def on_combo_changed(self, param_name, value):
        """Handle combo box changes (string parameters)"""
        if self.current_widget:
            self.current_widget.set_parameter(param_name, value)
            self.tf_canvas._draw()
            self.tf_canvas._notify_app()

    def update_ui_label(self, param_name, value):
        """Update the value label in UI"""
        if param_name in self.param_controls and 'label' in self.param_controls[param_name]:
            if isinstance(value, int):
                self.param_controls[param_name]['label'].setText(f"{value}")
            else:
                self.param_controls[param_name]['label'].setText(f"{value:.2f}")
        
    def on_widget_selected(self):
        """Show enhanced parameter controls for selected widget"""
        self.clear_parameter_controls()
        
        selected_items = self.widget_list.selectedItems()
        if not selected_items:
            self.param_group.setVisible(False)
            return
            
        widget_idx = selected_items[0].data(Qt.UserRole)
        self.current_widget = self.tf_canvas.widgets[widget_idx]
        
        # Create enhanced controls for each parameter
        self.param_controls = {}
        params = self.current_widget.get_parameters()
        
        for row, (param_name, param_config) in enumerate(params.items()):
            controls = self.create_parameter_control(param_name, param_config, row)
            if controls:
                self.param_controls[param_name] = controls
        
        # Color control - FIXED VERSION
        if hasattr(self.current_widget, 'color'):
            row = len(params)
            color_label = QtWidgets.QLabel("Color:")
            self.param_layout.addWidget(color_label, row, 0)
            
            self.color_btn = QtWidgets.QPushButton()
            self.color_btn.setFixedSize(60, 25)
            self.update_color_button()
            self.color_btn.clicked.connect(self.change_widget_color)
            self.param_layout.addWidget(self.color_btn, row, 1)
        
        # Action buttons
        row = len(params) + 2
        action_layout = QtWidgets.QHBoxLayout()
        
        delete_btn = QtWidgets.QPushButton("Delete Widget")
        delete_btn.clicked.connect(lambda: self.delete_widget(self.current_widget))
        action_layout.addWidget(delete_btn)
        
        self.param_layout.addLayout(action_layout, row, 0, 1, 4)
        
        self.param_group.setVisible(True)
        
    def update_color_button(self):
        """Update color button using same color format"""
        if hasattr(self, 'color_btn') and self.current_widget:
            r, g, b = self.current_widget.color
            # Convert from 0-1 float to 0-255 integer for display
            self.color_btn.setStyleSheet(
                f"background-color: rgb("
                f"{int(r*255)}, "
                f"{int(g*255)}, "
                f"{int(b*255)});"
                "border: 1px solid black;"
            )
        
    def clear_parameter_controls(self):
        """Clear all parameter controls"""
        for i in reversed(range(self.param_layout.count())):
            item = self.param_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
            elif item.layout():
                for j in reversed(range(item.layout().count())):
                    item.layout().itemAt(j).widget().setParent(None)
        
    def change_widget_color(self):
        """Change widget color using same pattern as point-based TF"""
        if self.current_widget:
            # Use the EXACT same pattern as your point-based TF
            qcolor = QtWidgets.QColorDialog.getColor()
            if qcolor.isValid():
                self.current_widget.color = (qcolor.redF(), qcolor.greenF(), qcolor.blueF())
                self.update_color_button()
                self.tf_canvas._draw()
                self.tf_canvas._notify_app()
                
    def duplicate_widget(self):
        """Duplicate the selected widget"""
        selected_items = self.widget_list.selectedItems()
        if selected_items and self.current_widget:
            import copy
            new_widget = copy.copy(self.current_widget)
            # Offset slightly so they don't overlap
            new_widget.center_intensity = min(255, new_widget.center_intensity + 10)
            new_widget.center_gradient = min(255, new_widget.center_gradient + 10)
            self.tf_canvas.add_widget(new_widget)
            self.update_widget_list()
            
    def add_preset_widget(self):
        widget_data = self.preset_combo.currentData()
        widget_type_str, preset_name = widget_data
        
        widget_type = WidgetType(widget_type_str)
        widget = WidgetFactory.create_widget(widget_type, preset=preset_name)
        self.tf_canvas.add_widget(widget)
        self.update_widget_list()
        
    def update_widget_list(self):
        self.widget_list.clear()
        for i, widget in enumerate(self.tf_canvas.widgets):
            item = QtWidgets.QListWidgetItem(f"{i+1}. {widget.widget_type.value}")
            item.setData(Qt.UserRole, i)
            self.widget_list.addItem(item)
            
    def delete_widget(self, widget):
        """Delete a widget from both canvas and UI"""
        if widget in self.tf_canvas.widgets:
            self.tf_canvas.remove_widget(widget)
            self.update_widget_list()
            self.param_group.setVisible(False)  # Hide parameters panel
            self.current_widget = None  # Clear current selection