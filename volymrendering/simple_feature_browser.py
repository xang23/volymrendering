# simple_feature_browser.py
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class SimpleMatrixBrowser(QtWidgets.QWidget):
    def __init__(self, feature_data_dict, update_callback, parent=None):
        super().__init__(parent)
        self.feature_data = feature_data_dict
        self.update_callback = update_callback
        self.feature_names = list(feature_data_dict.keys())
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        n = len(self.feature_names)
        total_pairs = n * (n - 1) // 2
        
        # Title
        title = QtWidgets.QLabel(
            f"nD Feature Matrix ({n} features, {total_pairs} unique pairs)"
        )
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2E86AB;")
        layout.addWidget(title)
        
        # Description
        desc = QtWidgets.QLabel(
            f"Click any cell above the diagonal to explore that feature pair. "
            f"Lower triangle is mirror (not shown)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(desc)
        
        # Add SCROLL AREA for large matrices
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create matrix grid
        self.matrix_widget = QtWidgets.QWidget()
        matrix_layout = QtWidgets.QGridLayout(self.matrix_widget)
        matrix_layout.setSpacing(2)
        
        n_features = len(self.feature_names)
        
        # Column headers (x-axis features)
        for col, feat_x in enumerate(self.feature_names):
            label = QtWidgets.QLabel(feat_x)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("font-weight: bold; background: #4A6572; color: white; padding: 5px;")
            matrix_layout.addWidget(label, 0, col + 1)
        
        # Row headers (y-axis features) and cells
        for row, feat_y in enumerate(self.feature_names):
            # Row header
            label = QtWidgets.QLabel(feat_y)
            label.setAlignment(Qt.AlignCenter) 
            label.setStyleSheet("font-weight: bold; background: #4A6572; color: white; padding: 5px;")
            matrix_layout.addWidget(label, row + 1, 0)
            
            # Matrix cells - ONLY UPPER TRIANGLE + DIAGONAL
            for col, feat_x in enumerate(self.feature_names):
                if col < row:
                    # Lower triangle - empty placeholder
                    empty_widget = QtWidgets.QLabel("")
                    empty_widget.setFixedSize(120, 120)
                    empty_widget.setStyleSheet("background: #2A2A2A; border: none;")
                    matrix_layout.addWidget(empty_widget, row + 1, col + 1)
                    
                elif col == row:
                    # Diagonal - histogram
                    cell = self.create_histogram_cell(feat_x)
                    matrix_layout.addWidget(cell, row + 1, col + 1)
                    
                else:  # col > row - Upper triangle (unique pairs)
                    cell = self.create_scatter_cell(feat_x, feat_y)
                    matrix_layout.addWidget(cell, row + 1, col + 1)
        
        scroll.setWidget(self.matrix_widget)
        layout.addWidget(scroll)
        
        # Add legend
        self.add_legend(layout)
        
    def add_legend(self, layout):
        """Add legend explaining matrix structure"""
        legend_widget = QtWidgets.QWidget()
        legend_layout = QtWidgets.QHBoxLayout(legend_widget)
        
        # Diagonal
        diag_label = QtWidgets.QLabel("■ Diagonal: Feature histogram")
        diag_label.setStyleSheet("color: #3498db; padding: 2px;")
        legend_layout.addWidget(diag_label)
        
        # Upper triangle
        upper_label = QtWidgets.QLabel("● Upper: Unique feature pairs")
        upper_label.setStyleSheet("color: #e67e22; padding: 2px;")
        legend_layout.addWidget(upper_label)
        
        # Lower triangle
        lower_label = QtWidgets.QLabel("◻ Lower: Mirrored (hidden)")
        lower_label.setStyleSheet("color: #7f8c8d; padding: 2px;")
        legend_layout.addWidget(lower_label)
        
        legend_layout.addStretch()
        layout.addWidget(legend_widget)
        
    def create_histogram_cell(self, feature_name):
        """Create a cell showing feature histogram (diagonal)"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Create matplotlib figure
        fig = Figure(figsize=(1.2, 1.2), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        data = self.feature_data[feature_name]
        hist, bins = np.histogram(data, bins=30)
        ax.fill_between(bins[:-1], hist, color='#3498db', alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        layout.addWidget(canvas)
        
        # Make clickable
        widget.mousePressEvent = lambda e: self.on_cell_clicked(feature_name, feature_name)
        widget.setToolTip(f"Click to view {feature_name} histogram in main TF")
        
        return widget
        
    def create_scatter_cell(self, feat_x, feat_y):
        """Create a cell showing 2D scatter plot (off-diagonal)"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Create matplotlib figure
        fig = Figure(figsize=(1.2, 1.2), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        data_x = self.feature_data[feat_x]
        data_y = self.feature_data[feat_y]
        
        # Sample for performance
        if len(data_x) > 1000:
            indices = np.random.choice(len(data_x), 1000, replace=False)
            data_x = data_x[indices]
            data_y = data_y[indices]
            
        ax.scatter(data_x, data_y, s=1, alpha=0.6, color='#e67e22')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        layout.addWidget(canvas)
        
        # Make clickable
        widget.mousePressEvent = lambda e: self.on_cell_clicked(feat_x, feat_y)
        widget.setToolTip(f"Click to explore {feat_x} vs {feat_y}")
        
        return widget
        
    def on_cell_clicked(self, feat_x, feat_y):
        """Handle cell clicks with debounce"""
        print(f"Matrix cell clicked: {feat_x} vs {feat_y}")
    
        # Skip diagonal
        if feat_x == feat_y:
            print("   Diagonal cell - skipping")
            return
    
        # Simple debounce - disable temporarily
        self.setEnabled(False)
    
        # Find the main app window
        parent = self
        while parent and not hasattr(parent, 'open_feature_popup'):
            parent = parent.parent()
    
        if parent and hasattr(parent, 'open_feature_popup'):
            parent.open_feature_popup(feat_x, feat_y)
    
        # Re-enable after short delay
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, lambda: self.setEnabled(True))
            
    def update_matrix(self):
        """Update matrix with new feature data"""
        # Remove old matrix
        old_matrix = self.matrix_widget
        self.matrix_widget = QtWidgets.QWidget()
        self.layout().replaceWidget(old_matrix, self.matrix_widget)
        old_matrix.deleteLater()
        
        # Update feature names
        self.feature_names = list(self.feature_data.keys())
        
        # Create new matrix
        self.setup_ui()