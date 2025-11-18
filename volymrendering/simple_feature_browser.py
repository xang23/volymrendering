# simple_matrix_browser.py
import numpy as np
from PyQt5 import QtWidgets, QtCore
from unified_tf_canvas import UnifiedTFCanvas  # YOUR EXISTING CANVAS!

class SimpleMatrixBrowser(QtWidgets.QWidget):
    def __init__(self, feature_data_dict, update_callback=None):
        """
        feature_data_dict: {feature_name: normalized_data}
        Uses YOUR existing UnifiedTFCanvas for each cell!
        """
        super().__init__()
        self.feature_data = feature_data_dict
        self.update_callback = update_callback
        self.feature_names = list(feature_data_dict.keys())
        self.canvases = {}  # (i,j) -> YOUR UnifiedTFCanvas
        
        self.setup_ui()
        self.build_matrix()
    
    def setup_ui(self):
        """Simple matrix UI using your existing canvas components"""
        self.main_layout = QtWidgets.QVBoxLayout()
        
        # Title
        title = QtWidgets.QLabel(f"Feature Matrix - Click any cell to explore")
        title.setStyleSheet("font-weight: bold; color: darkblue;")
        self.main_layout.addWidget(title)
        
        # Matrix container
        self.matrix_container = QtWidgets.QWidget()
        self.matrix_layout = QtWidgets.QGridLayout()
        self.matrix_container.setLayout(self.matrix_layout)
        
        # Scroll area for large matrices
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.matrix_container)
        self.scroll_area.setMinimumSize(700, 500)
        
        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)
    
    def build_matrix(self):
        """Build matrix using YOUR existing UnifiedTFCanvas for each cell"""
        n = len(self.feature_names)
        print(f"🔄 Building {n}x{n} feature matrix...")
        
        # Column headers
        for j in range(1, n):
            header = QtWidgets.QLabel(self.feature_names[j])
            header.setAlignment(QtCore.Qt.AlignCenter)
            header.setStyleSheet("font-weight: bold; background: #e0e0e0; padding: 2px;")
            self.matrix_layout.addWidget(header, 0, j)
        
        # Row labels and YOUR canvases
        for i in range(n - 1):
            # Row label
            row_label = QtWidgets.QLabel(self.feature_names[i])
            row_label.setAlignment(QtCore.Qt.AlignCenter)
            row_label.setStyleSheet("font-weight: bold; background: #e0e0e0; padding: 2px;")
            self.matrix_layout.addWidget(row_label, i + 1, 0)
            
            # YOUR UnifiedTFCanvas for each feature pair
            for j in range(i + 1, n):
                canvas = self.create_matrix_cell(i, j)
                self.canvases[(i, j)] = canvas
                self.matrix_layout.addWidget(canvas, i + 1, j)
        
        print(f"✅ Matrix built with {len(self.canvases)} feature pairs")
    
    def create_matrix_cell(self, idx_i, idx_j):
        """Create a matrix cell using YOUR UnifiedTFCanvas"""
        feature_x = self.feature_names[idx_i]
        feature_y = self.feature_names[idx_j]
        
        # USE YOUR EXISTING UNIFIEDTFCANVAS!
        canvas = UnifiedTFCanvas(
            tf_type='2d',
            data=self.feature_data[feature_x],
            gradient_data=self.feature_data[feature_y],
            update_callback=lambda: self.on_cell_click(feature_x, feature_y)
        )
        
        # Smaller size for matrix cells
        canvas.setMinimumSize(150, 150)
        canvas.setMaximumSize(200, 200)
        
        # Update labels to show feature names
        canvas.ax.set_xlabel(feature_x, fontsize=6)
        canvas.ax.set_ylabel(feature_y, fontsize=6)
        canvas.ax.set_title(f'{feature_x} vs {feature_y}', fontsize=7)
        canvas.ax.tick_params(labelsize=5)
        canvas.draw()
        
        return canvas
    
    def on_cell_click(self, feature_x, feature_y):
        """When user interacts with a matrix cell"""
        print(f"🎯 Matrix cell clicked: {feature_x} vs {feature_y}")
        if self.update_callback:
            self.update_callback(feature_x, feature_y)