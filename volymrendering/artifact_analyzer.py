# artifact_analyzer.py
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import time
from unified_tf_canvas import UnifiedTFCanvas
from tf_canvas_widget import TFCanvasWidget
from nd_shader_renderer import NDShaderRenderer
from nd_widget_manager import NDWidgetManager
from widget_factory import WidgetFactory, WidgetType

class ArtifactAnalyzer(QtWidgets.QMainWindow):
    """Separate window for controlled artifact analysis with shared renderer"""
    
    def __init__(self, main_app, parent=None):
        super().__init__(parent)
        self.main_app = main_app
        self.setWindowTitle("Artifact Analyzer - Side by Side Comparison")
        self.setGeometry(150, 150, 1600, 900)
        
        # Get data from main app
        self.nd_manager = main_app.nd_manager
        self.image_data = main_app.image_data
        self.reader = main_app.reader
        self.all_features = main_app.all_features
        
        # Ensure widgets list exists
        if not hasattr(self.nd_manager, 'widgets'):
            self.nd_manager.widgets = []
        
        self.test_configs = self.create_test_configurations()
        self.current_results = []
        
        # Shared renderer
        self.shared_renderer = None
        self.temp_nd_manager = None
        
        self.setup_ui()
    
    def normalize_to_255(self, data):
        """Normalize data to 0-255 range"""
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            return 255.0 * (data - data_min) / (data_max - data_min)
        return np.zeros_like(data)
    
    def create_test_configurations(self):
        """Create predefined test configurations with consistent colors"""
        NEUTRAL_COLOR = (1.0, 1.0, 1.0)  # White - most neutral
        
        return [
            {
                'name': 'Gaussian Soft',
                'widgets': [{
                    'type': WidgetType.GAUSSIAN,
                    'params': {
                        'center_intensity': 128,
                        'center_gradient': 128,
                        'intensity_std': 30,
                        'gradient_std': 30,
                        'opacity': 0.8,
                        'color': NEUTRAL_COLOR,
                        'blend_mode': 'max'
                    }
                }]
            },
            {
                'name': 'Gaussian Sharp',
                'widgets': [{
                    'type': WidgetType.GAUSSIAN,
                    'params': {
                        'center_intensity': 128,
                        'center_gradient': 128,
                        'intensity_std': 5,
                        'gradient_std': 5,
                        'opacity': 0.8,
                        'color': NEUTRAL_COLOR,
                        'blend_mode': 'max'
                    }
                }]
            },
            {
                'name': 'Rectangular',
                'widgets': [{
                    'type': WidgetType.RECTANGULAR,
                    'params': {
                        'center_intensity': 128,
                        'center_gradient': 128,
                        'intensity_width': 40,
                        'gradient_height': 40,
                        'opacity': 0.8,
                        'color': NEUTRAL_COLOR,
                        'blend_mode': 'max'
                    }
                }]
            },
            {
                'name': 'Triangular',
                'widgets': [{
                    'type': WidgetType.TRIANGULAR,
                    'params': {
                        'center_intensity': 128,
                        'center_gradient': 128,
                        'intensity_width': 50,
                        'gradient_height': 50,
                        'opacity': 0.8,
                        'color': NEUTRAL_COLOR,
                        'blend_mode': 'max'
                    }
                }]
            },
            {
                'name': 'Two Gaussians',
                'widgets': [
                    {
                        'type': WidgetType.GAUSSIAN,
                        'params': {
                            'center_intensity': 80,
                            'center_gradient': 80,
                            'intensity_std': 20,
                            'gradient_std': 20,
                            'opacity': 0.7,
                            'color': NEUTRAL_COLOR,
                            'blend_mode': 'max'
                        }
                    },
                    {
                        'type': WidgetType.GAUSSIAN,
                        'params': {
                            'center_intensity': 180,
                            'center_gradient': 180,
                            'intensity_std': 20,
                            'gradient_std': 20,
                            'opacity': 0.7,
                            'color': NEUTRAL_COLOR,
                            'blend_mode': 'max'
                        }
                    }
                ]
            }
        ]
    
    def setup_ui(self):
        """Setup the user interface"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        
        # ===== Control Panel =====
        control = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(control)
        
        # Feature selection
        control_layout.addWidget(QtWidgets.QLabel("Feature X:"))
        self.feat_x = QtWidgets.QComboBox()
        self.feat_x.addItems(list(self.all_features.keys()))
        self.feat_x.setCurrentText('Intensity')
        control_layout.addWidget(self.feat_x)
        
        control_layout.addWidget(QtWidgets.QLabel("Y:"))
        self.feat_y = QtWidgets.QComboBox()
        self.feat_y.addItems(list(self.all_features.keys()))
        self.feat_y.setCurrentText('Gradient')
        control_layout.addWidget(self.feat_y)
        
        # Run button
        self.run_btn = QtWidgets.QPushButton("Run All Tests")
        self.run_btn.clicked.connect(self.run_all_tests)
        control_layout.addWidget(self.run_btn)
        
        # Clear button
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_results)
        control_layout.addWidget(self.clear_btn)
        
        control_layout.addStretch()
        layout.addWidget(control)
        
        # ===== Results Area (Scrollable) =====
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.results_widget = QtWidgets.QWidget()
        self.results_layout = QtWidgets.QGridLayout(self.results_widget)
        self.results_layout.setSpacing(15)
        self.scroll.setWidget(self.results_widget)
        layout.addWidget(self.scroll)
        
        self.current_row = 0
        self.current_col = 0
        self.max_cols = 2
    
    def create_test_frame(self, test_name, feat_x, feat_y):
        """Create a frame for a single test result"""
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.Box)
        frame.setLineWidth(2)
        frame.setMinimumWidth(550)
        layout = QtWidgets.QVBoxLayout(frame)
        
        # Test name
        title = QtWidgets.QLabel(f"<b>{test_name}</b>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Feature pair
        feat_label = QtWidgets.QLabel(f"{feat_x} vs {feat_y}")
        feat_label.setAlignment(Qt.AlignCenter)
        feat_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(feat_label)
        
        # Placeholder for mini TF canvas
        tf_placeholder = QtWidgets.QFrame()
        tf_placeholder.setMinimumHeight(80)
        tf_placeholder.setStyleSheet("background-color: #222; border: 1px solid #555;")
        layout.addWidget(tf_placeholder)
        frame.tf_placeholder = tf_placeholder
        
        # Stats label
        stats = QtWidgets.QLabel("Waiting...")
        stats.setAlignment(Qt.AlignCenter)
        stats.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(stats)
        
        # VTK placeholder
        vtk_placeholder = QtWidgets.QFrame()
        vtk_placeholder.setMinimumHeight(250)
        vtk_placeholder.setStyleSheet("background-color: black;")
        layout.addWidget(vtk_placeholder)
        frame.vtk_placeholder = vtk_placeholder
        
        # Store references
        frame.stats_label = stats
        frame.vtk_widget = None
        frame.tf_canvas = None
        frame.last_fps = 0
        frame.last_avg = 0
        
        return frame
    
    def add_mini_tf_to_frame(self, frame, test, feat_x, feat_y):
        """Lägg till mini TF canvas i frame"""
        # Get data for this feature pair
        data_x = self.all_features[feat_x]
        data_y = self.all_features[feat_y]
        
        # Normalize for display
        norm_x = self.normalize_to_255(data_x)
        norm_y = self.normalize_to_255(data_y)
        
        # Create mini TF canvas
        mini_tf = UnifiedTFCanvas(
            tf_type='2d',
            data=norm_x,
            gradient_data=norm_y
        )
        mini_tf.setFixedHeight(80)
        mini_tf.set_projection_features(feat_x, feat_y)
        
        # Ladda widgets till mini canvas
        for w_data in test['widgets']:
            widget = WidgetFactory.create_widget(w_data['type'], **w_data['params'])
            mini_tf.add_widget(widget)
        mini_tf._draw()
        
        # Ersätt placeholder med TF canvas
        layout = frame.layout()
        tf_wrapper = TFCanvasWidget(mini_tf, self, label='')
        tf_wrapper.reset_btn.hide()
        
        tf_index = layout.indexOf(frame.tf_placeholder)
        layout.removeWidget(frame.tf_placeholder)
        frame.tf_placeholder.deleteLater()
        layout.insertWidget(tf_index, tf_wrapper)
        frame.tf_canvas = mini_tf
    
    def run_all_tests(self):
        """Run all test configurations - create new renderer for each test"""
        selected_pairs = self.setup_feature_selection()
        if not selected_pairs:
            return
    
        self.clear_results()
        self.current_row = 0
        self.current_col = 0
    
        for feat_x, feat_y in selected_pairs:
            for test in self.test_configs:
                print(f"\n🔬 Testing {test['name']} with {feat_x} vs {feat_y}")
            
                # Skapa frame
                frame = self.create_test_frame(test['name'], feat_x, feat_y)
                self.results_layout.addWidget(frame, self.current_row, self.current_col)
            
                self.current_col += 1
                if self.current_col >= self.max_cols:
                    self.current_col = 0
                    self.current_row += 1
            
                QtWidgets.QApplication.processEvents()
            
                # Lägg till mini TF canvas
                self.add_mini_tf_to_frame(frame, test, feat_x, feat_y)
            
                # Skapa temporär nd_manager för detta test
                temp_nd_manager = NDWidgetManager()
                for w_data in test['widgets']:
                    widget = WidgetFactory.create_widget(w_data['type'], **w_data['params'])
                    temp_nd_manager.add_widget(widget)
            
                # Skapa ny renderer för detta test
                renderer = NDShaderRenderer(
                    self.image_data,
                    list(self.all_features.keys()),
                    temp_nd_manager,
                    f"test_{test['name']}_{feat_x}_{feat_y}"
                )
            
                # ===== VIKTIGT: Använd bara load_only_features, INTE set_feature_pair =====
                # Detta skapar en 2-komponents volym
                renderer.load_only_features(feat_x, feat_y)
            
                # Skapa VTK widget
                vtk_widget = QVTKRenderWindowInteractor()
                vtk_widget.setMinimumHeight(250)
                vtk_widget.GetRenderWindow().AddRenderer(renderer.get_renderer())
            
                layout = frame.layout()
                vtk_index = layout.indexOf(frame.vtk_placeholder)
                layout.removeWidget(frame.vtk_placeholder)
                frame.vtk_placeholder.deleteLater()
                layout.insertWidget(vtk_index, vtk_widget)
            
                frame.vtk_widget = vtk_widget
                frame.renderer = renderer
            
                vtk_widget.Initialize()
                vtk_widget.Start()
            
                # Mät prestanda
                times = []
                for i in range(20):
                    start = time.perf_counter()
                    vtk_widget.GetRenderWindow().Render()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
            
                avg_time = sum(times) / len(times)
                fps = 1000 / avg_time
            
                frame.last_fps = fps
                frame.last_avg = avg_time
                frame.stats_label.setText(f"FPS: {fps:.1f} | {avg_time:.2f}ms")
            
                self.current_results.append({
                    'tf_name': test['name'],
                    'feat_x': feat_x,
                    'feat_y': feat_y,
                    'fps': fps,
                    'avg_ms': avg_time,
                    'frame': frame,
                    'renderer': renderer
                })
            
                QtWidgets.QApplication.processEvents()
    
        print(f"\n✅ Complete - {len(self.current_results)} tests")
    
    def clear_results(self):
        """Clear all test results"""
        # Stäng alla VTK widgets
        for result in self.current_results:
            frame = result.get('frame')
            if frame and hasattr(frame, 'vtk_widget') and frame.vtk_widget:
                try:
                    frame.vtk_widget.close()
                except:
                    pass
            if frame and hasattr(frame, 'tf_canvas') and frame.tf_canvas:
                try:
                    frame.tf_canvas.close()
                except:
                    pass
        
        # Rensa layout
        while self.results_layout.count():
            item = self.results_layout.itemAt(0)
            if item and item.widget():
                item.widget().deleteLater()
            self.results_layout.removeItem(item)
        
        self.current_results = []
        self.current_row = 0
        self.current_col = 0
    
    def setup_feature_selection(self):
        """Dialog for selecting which feature pairs to test"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Feature Pairs")
        layout = QtWidgets.QVBoxLayout(dialog)
        
        layout.addWidget(QtWidgets.QLabel("Choose feature pairs to test:"))
        
        checkboxes = []
        features = list(self.all_features.keys())
        
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                cb = QtWidgets.QCheckBox(f"{features[i]} vs {features[j]}")
                cb.setChecked(True)
                layout.addWidget(cb)
                checkboxes.append((cb, features[i], features[j]))
        
        btn_layout = QtWidgets.QHBoxLayout()
        
        select_all = QtWidgets.QPushButton("Select All")
        select_all.clicked.connect(lambda: [cb.setChecked(True) for cb,_,_ in checkboxes])
        btn_layout.addWidget(select_all)
        
        deselect_all = QtWidgets.QPushButton("Deselect All")
        deselect_all.clicked.connect(lambda: [cb.setChecked(False) for cb,_,_ in checkboxes])
        btn_layout.addWidget(deselect_all)
        
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        result = []
        def on_ok():
            nonlocal result
            result = [(fx, fy) for cb, fx, fy in checkboxes if cb.isChecked()]
            dialog.accept()
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return result
        return []
    
    def closeEvent(self, event):
        """Städa upp alla renderers när fönstret stängs"""
        for result in self.current_results:
            if 'renderer' in result and result['renderer']:
                try:
                    result['renderer'].cleanup()
                except:
                    pass
        self.clear_results()
        event.accept()



    