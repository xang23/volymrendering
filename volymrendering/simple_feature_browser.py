# simple_feature_browser.py
import numpy as np
from pathlib import Path
from PyQt5 import QtWidgets, QtCore

class SimpleFeatureBrowser(QtWidgets.QWidget):
    def __init__(self, dataset_directory, volume_data, tf_canvas):
        super().__init__()
        self.dataset_dir = Path(dataset_directory)
        self.volume_data = volume_data
        self.tf_canvas = tf_canvas  # Your existing UnifiedTFCanvas!
        self.available_features = {}
        
        self.setup_ui()
        self.discover_features()
    
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = QtWidgets.QLabel("Feature Explorer")
        title.setStyleSheet("font-weight: bold; color: darkblue;")
        layout.addWidget(title)
        
        # Feature pair selector
        pair_layout = QtWidgets.QHBoxLayout()
        
        self.feature_x_combo = QtWidgets.QComboBox()
        self.feature_y_combo = QtWidgets.QComboBox()
        
        self.load_btn = QtWidgets.QPushButton("Load Feature Pair")
        self.load_btn.clicked.connect(self.load_feature_pair)
        
        pair_layout.addWidget(QtWidgets.QLabel("X:"))
        pair_layout.addWidget(self.feature_x_combo)
        pair_layout.addWidget(QtWidgets.QLabel("Y:"))
        pair_layout.addWidget(self.feature_y_combo)
        pair_layout.addWidget(self.load_btn)
        
        layout.addLayout(pair_layout)
        
        # Status
        self.status_label = QtWidgets.QLabel("Select features and click Load")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def discover_features(self):
        """Simple feature discovery"""
        self.available_features.clear()
        
        # Look for feature files
        for npy_file in self.dataset_dir.glob("*.npy"):
            feature_name = npy_file.stem
            self.available_features[feature_name] = str(npy_file)
        
        # Add volume-derived features
        self.available_features['Intensity'] = 'volume'
        self.available_features['Gradient'] = 'volume'
        
        # Populate combos
        feature_names = sorted(self.available_features.keys())
        self.feature_x_combo.addItems(feature_names)
        self.feature_y_combo.addItems(feature_names)
        
        # Set defaults
        if 'Intensity' in feature_names and 'Gradient' in feature_names:
            self.feature_x_combo.setCurrentText('Intensity')
            self.feature_y_combo.setCurrentText('Gradient')
            
        self.status_label.setText(f"Found {len(feature_names)} features")
    
    def load_feature_pair(self):
        """Load and display selected feature pair in existing TF canvas"""
        feature_x = self.feature_x_combo.currentText()
        feature_y = self.feature_y_combo.currentText()
        
        if feature_x == feature_y:
            self.status_label.setText("❌ Select different features")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            return
        
        try:
            # Load feature data
            data_x = self.load_feature_data(feature_x)
            data_y = self.load_feature_data(feature_y)
            
            # Update your existing TF canvas!
            if hasattr(self.tf_canvas, 'set_feature_pair'):
                self.tf_canvas.set_feature_pair(feature_x, feature_y, data_x, data_y)
                self.status_label.setText(f"✅ Loaded: {feature_x} vs {feature_y}")
                self.status_label.setStyleSheet("color: green; font-size: 10px;")
            else:
                self.status_label.setText("❌ TF Canvas doesn't support feature pairs")
                self.status_label.setStyleSheet("color: red; font-size: 10px;")
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            print(f"Error loading features: {e}")
    
    def load_feature_data(self, feature_name):
        """Load a single feature"""
        if feature_name == 'Intensity':
            return self.normalize_to_255(self.volume_data.flatten())
        elif feature_name == 'Gradient':
            return self.normalize_to_255(self.compute_gradient())
        else:
            data = np.load(self.available_features[feature_name])
            return self.normalize_to_255(data)
    
    def compute_gradient(self):
        """Simple gradient computation"""
        try:
            # For 3D volume
            if len(self.volume_data.shape) == 3:
                grad = np.gradient(self.volume_data)
                grad_mag = np.sqrt(grad[0]**2 + grad[1]**2 + grad[2]**2)
            else:
                # For 2D or flat data
                grad_mag = np.abs(np.gradient(self.volume_data))
            return grad_mag.flatten()
        except:
            # Fallback
            return np.ones_like(self.volume_data.flatten()) * 128
    
    def normalize_to_255(self, data):
        """Normalize data to 0-255 range"""
        data = data.astype(np.float32)
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max - data_min > 0:
            normalized = (data - data_min) / (data_max - data_min) * 255
            return normalized.astype(np.uint8)
        else:
            return np.zeros_like(data, dtype=np.uint8)