# artifact_analyzer.py
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import numpy as np
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import time
from volume_renderer import VolumeRenderer
from unified_tf_canvas import UnifiedTFCanvas
from tf_canvas_widget import TFCanvasWidget

class ArtifactAnalyzer(QtWidgets.QMainWindow):
    """Separate window for controlled artifact analysis"""
    
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
        
        self.setup_ui()
    
    def normalize_to_255(self, data):
        """Normalize data to 0-255 range"""
        data_min, data_max = np.min(data), np.max(data)
        if data_max > data_min:
            return 255.0 * (data - data_min) / (data_max - data_min)
        return np.zeros_like(data)
    
    def create_test_configurations(self):
        """Create predefined test configurations with consistent colors"""
        from widget_factory import WidgetFactory, WidgetType
        
        # Use the same neutral color for ALL tests
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
        
        # Sampling control
        control_layout.addWidget(QtWidgets.QLabel("Sampling:"))
        self.sampling = QtWidgets.QComboBox()
        self.sampling.addItems(['1x (fast)', '2x', '4x (normal)', '8x (slow)'])
        control_layout.addWidget(self.sampling)
        
        # Load Saved TF button
        self.load_tf_btn = QtWidgets.QPushButton("Load Saved TF")
        self.load_tf_btn.clicked.connect(self.load_saved_tf)
        control_layout.addWidget(self.load_tf_btn)
        
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
        self.results_layout = QtWidgets.QHBoxLayout(self.results_widget)
        self.results_layout.setSpacing(20)
        self.scroll.setWidget(self.results_widget)
        layout.addWidget(self.scroll)
    
    def create_test_frame(self, test_name, feat_x, feat_y):
        """Create a frame for a single test result"""
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.Box)
        frame.setLineWidth(2)
        frame.setMinimumWidth(400)
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
        
        # Placeholder for mini TF canvas (will be replaced in update)
        tf_placeholder = QtWidgets.QFrame()
        tf_placeholder.setMinimumHeight(80)
        tf_placeholder.setStyleSheet("background-color: #222; border: 1px solid #555;")
        layout.addWidget(tf_placeholder)
        frame.tf_placeholder = tf_placeholder
        
        # Stats label
        stats = QtWidgets.QLabel("Waiting...")
        stats.setAlignment(Qt.AlignCenter)
        layout.addWidget(stats)
        
        # VTK placeholder
        vtk_placeholder = QtWidgets.QFrame()
        vtk_placeholder.setMinimumHeight(200)
        vtk_placeholder.setStyleSheet("background-color: black;")
        layout.addWidget(vtk_placeholder)
        frame.vtk_placeholder = vtk_placeholder
        
        # Store references
        frame.stats_label = stats
        frame.vtk_widget = None
        frame.renderer = None
        frame.tf_canvas = None
        frame.last_fps = 0
        frame.last_avg = 0
        
        return frame
    
    def update_test_frame(self, frame, test_name, feat_x, feat_y, sampling_factor):
        """Update frame with actual rendering and TF visualization"""
        try:
            layout = frame.layout()
            
            # ===== 1. Create/Update Mini TF Canvas =====
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
            
            # Load projected widgets
            projected = self.nd_manager.project_to_2d(feat_x, feat_y)
            for widget in projected:
                mini_tf.add_widget(widget)
            mini_tf._draw()
            
            # Replace placeholder with actual TF canvas
            from tf_canvas_widget import TFCanvasWidget
            tf_wrapper = TFCanvasWidget(mini_tf, self, label='')
            tf_wrapper.reset_btn.hide()
            
            tf_index = layout.indexOf(frame.tf_placeholder)
            layout.removeWidget(frame.tf_placeholder)
            frame.tf_placeholder.deleteLater()
            layout.insertWidget(tf_index, tf_wrapper)
            frame.tf_canvas = mini_tf
            # ===========================================
            
            # ===== 2. Create VTK Render View =====
            # Remove old VTK widget if it exists
            if hasattr(frame, 'vtk_widget') and frame.vtk_widget:
                frame.vtk_widget.close()
                layout.removeWidget(frame.vtk_widget)
            
            # Create new VTK widget
            vtk_widget = QVTKRenderWindowInteractor()
            vtk_widget.setMinimumHeight(200)
            
            # Create renderer
            renderer = vtk.vtkRenderer()
            vtk_widget.GetRenderWindow().AddRenderer(renderer)
            
            # Setup volume rendering
            vol_renderer = VolumeRenderer(f"test_{test_name}")
            vol_renderer.set_volume_data(self.image_data, self.reader)
            renderer.AddVolume(vol_renderer.volume)
            renderer.ResetCamera()
            
            # Add to layout (replace placeholder)
            vtk_index = layout.indexOf(frame.vtk_placeholder)
            layout.removeWidget(frame.vtk_placeholder)
            frame.vtk_placeholder.deleteLater()
            layout.insertWidget(vtk_index, vtk_widget)
            
            # Store references
            frame.vtk_widget = vtk_widget
            frame.renderer = vol_renderer
            
            # Initialize
            vtk_widget.Initialize()
            vtk_widget.Start()
            # ===================================
            
            # ===== 3. Apply TF to Renderer =====
            # Save current widgets from main canvas
            original_widgets = self.main_app.tf_canvas.widgets.copy()
            
            # Temporarily set the test widgets on the main canvas
            self.main_app.tf_canvas.widgets = self.nd_manager.widgets.copy()
            
            # Set the correct feature data for sampling
            self.main_app.tf_canvas.data = norm_x
            self.main_app.tf_canvas.gradient_data = norm_y
            
            # Sample from the canvas
            samples = self.main_app.tf_canvas.sample_for_vtk()
            
            # Restore original widgets
            self.main_app.tf_canvas.widgets = original_widgets
            # ===================================
            
            if samples:
                intensities = [s[0] for s in samples]
                opacities = [s[1] for s in samples]
                colors = [s[2] for s in samples]
                
                # Scale to data range
                x_min, x_max = np.min(data_x), np.max(data_x)
                scaled = [x_min + (i/255.0)*(x_max - x_min) for i in intensities]
                
                vol_renderer.update_transfer_functions(
                    scaled, opacities, colors, (x_min, x_max)
                )
            
            # ===== 4. Measure Performance =====
            times = []
            for i in range(30):
                start = time.perf_counter()
                vtk_widget.GetRenderWindow().Render()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            avg_time = sum(times) / len(times)
            fps = 1000 / avg_time
            min_time = min(times)
            max_time = max(times)
            std_dev = np.std(times)
            
            # Store stats in frame
            frame.last_fps = fps
            frame.last_avg = avg_time
            
            # Update stats label
            frame.stats_label.setText(
                f"FPS: {fps:.1f} | {avg_time:.2f}ms\n"
                f"Min: {min_time:.2f} Max: {max_time:.2f}\n"
                f"Std: {std_dev:.2f}"
            )
            
            # Force layout update
            layout.update()
            
        except Exception as e:
            print(f"Error in update_test_frame: {e}")
            import traceback
            traceback.print_exc()
            frame.stats_label.setText(f"Error: {str(e)[:30]}")
    
    def run_all_tests(self):
        """Run all test configurations"""
        # Let user select which feature pairs to test
        selected_pairs = self.setup_feature_selection()
        if not selected_pairs:
            return
        
        self.clear_results()
        sampling_idx = self.sampling.currentIndex()
        sampling_factors = [1, 2, 4, 8]
        sampling_factor = sampling_factors[sampling_idx]
        
        for feat_x, feat_y in selected_pairs:
            for test in self.test_configs:
                # Create result frame for this test
                frame = self.create_test_frame(test['name'], feat_x, feat_y)
                self.results_layout.addWidget(frame)
                QtWidgets.QApplication.processEvents()
                
                # Apply test configuration
                self.apply_test_configuration(test)
                
                # Update the frame with rendering
                self.update_test_frame(frame, test['name'], feat_x, feat_y, sampling_factor)
                
                # Store result
                self.current_results.append({
                    'tf_name': test['name'],
                    'feat_x': feat_x,
                    'feat_y': feat_y,
                    'fps': frame.last_fps,
                    'avg_ms': frame.last_avg,
                    'frame': frame
                })
    
    def run_all_tests_with_current_widgets(self, tf_name):
        """Run all feature pair tests with current widget configuration"""
        selected_pairs = self.setup_feature_selection()
        if not selected_pairs:
            return
        
        self.clear_results()
        sampling_idx = self.sampling.currentIndex()
        sampling_factors = [1, 2, 4, 8]
        sampling_factor = sampling_factors[sampling_idx]
        
        for feat_x, feat_y in selected_pairs:
            # Create result frame
            frame = self.create_test_frame(tf_name, feat_x, feat_y)
            self.results_layout.addWidget(frame)
            QtWidgets.QApplication.processEvents()
            
            # Update the frame with rendering
            self.update_test_frame(frame, tf_name, feat_x, feat_y, sampling_factor)
            
            # Store result info
            self.current_results.append({
                'tf_name': tf_name,
                'feat_x': feat_x,
                'feat_y': feat_y,
                'fps': frame.last_fps,
                'avg_ms': frame.last_avg,
                'frame': frame
            })
    
    def apply_test_configuration(self, test):
        """Apply test configuration to nd_manager"""
        from widget_factory import WidgetFactory
        
        # Clear existing widgets
        self.nd_manager.widgets.clear()
        
        # Create widgets from test config
        for w_data in test['widgets']:
            widget = WidgetFactory.create_widget(
                w_data['type'],
                **w_data['params']
            )
            self.nd_manager.add_widget(widget)
    
    def clear_results(self):
        """Clear all test results"""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.current_results = []
    
    def load_saved_tf(self):
        """Load a saved transfer function from your TF manager"""
        tf_names = list(self.main_app.tf_manager.saved_tfs.keys())
        
        if not tf_names:
            QtWidgets.QMessageBox.warning(self, "No TFs", "No saved transfer functions found")
            return
        
        name, ok = QtWidgets.QInputDialog.getItem(
            self, "Select TF", "Choose a saved transfer function:", 
            tf_names, 0, False
        )
        
        if not ok:
            return
        
        # Load the TF data
        tf_data = self.main_app.tf_manager.saved_tfs[name]
        
        # Convert to widget format
        self.convert_tf_to_widgets(tf_data)
        
        # Run tests with this TF
        self.run_all_tests_with_current_widgets(name)
    
    def convert_tf_to_widgets(self, tf_data):
        """Convert a point-based TF to widget format - WITH PROPER NORMALIZATION"""
        from widget_factory import WidgetFactory, WidgetType
    
        # Clear existing widgets
        self.nd_manager.widgets.clear()
    
        # Get TF points
        xs = tf_data.get('x_abs', tf_data.get('x_rel', []))
        ys = tf_data.get('y', [])
        colors = tf_data.get('colors', [])
    
        # Get data range from main app
        int_min, int_max = self.main_app.intensity_range
        print(f"Converting TF with data range: [{int_min:.1f}, {int_max:.1f}]")
    
        # Use neutral color for all widgets
        NEUTRAL_COLOR = (1.0, 1.0, 1.0)
    
        # Create a widget for each significant point
        for i, (x, y, color) in enumerate(zip(xs, ys, colors)):
            if y < 0.05:  # Skip very low opacity
                continue
        
            # ===== CRITICAL: Normalize to 0-255 display space =====
            if int_max > int_min:
                x_display = 255.0 * (x - int_min) / (int_max - int_min)
            else:
                x_display = 128  # Default if range is zero
        
            # Clamp to valid range
            x_display = max(0, min(255, x_display))
        
            print(f"Point {i}: Raw {x:.1f} → Display {x_display:.1f}")
            # ======================================================
        
            # Create Gaussian widget at this point (in display space!)
            widget = WidgetFactory.create_widget(
                WidgetType.GAUSSIAN,
                center_intensity=x_display,  # ← Now in 0-255!
                center_gradient=128,  # Default gradient
                intensity_std=15,  # In display space
                gradient_std=30,
                opacity=y,
                color=NEUTRAL_COLOR,
                blend_mode='max'
            )
            self.nd_manager.add_widget(widget)
    
        print(f"Converted {len(self.nd_manager.widgets)} TF points to widgets (all in 0-255 space)")
    
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
        
        # Buttons
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