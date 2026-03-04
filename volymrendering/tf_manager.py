import json
import os
import numpy as np
from PyQt5 import QtWidgets


class TFManager:
    TF_SAVE_FILE = "saved_tfs.json"

    def __init__(self, tf_selector, parent_window=None):
        self.tf_selector = tf_selector
        self.parent_window = parent_window
        self.saved_tfs = {}
        
        print("=== TFManager Initialization ===")
        print(f"TF file path: {os.path.abspath(self.TF_SAVE_FILE)}")
        
        # Load existing TFs FIRST
        self.load_tfs_from_disk()
        
        # Initialize selector AFTER loading
        self.update_tf_selector()
        
        print(f"Final saved_tfs: {list(self.saved_tfs.keys())}")
        print("=== TFManager Initialization Complete ===\n")

    def save_current_tf(self, points_x, points_y, colors, data_range=None):
        """Save current transfer function - MATCHING your file format."""
    
        # Get name from user
        name, ok = QtWidgets.QInputDialog.getText(
            self.parent_window, "Save Transfer Function", "Name:"
        )
        if not ok or not name:
            return False
    
        print(f"\n=== Saving TF '{name}' ===")
    
        # Create clean copies
        xs = [float(x) for x in points_x]
        ys = [float(y) for y in points_y]
        colors_list = [tuple(float(c) for c in color) for color in colors]
    
        # ===== FIX: Save in THE SAME FORMAT as your file =====
        # Your file uses x_abs, not x_rel!
        self.saved_tfs[name] = {
            'x_abs': xs,                    # ← Use x_abs, not x_rel
            'y': ys,
            'colors': colors_list,
            'version': 1,
            'converted_from': 'new_save'     # Optional marker
        }
    
        print(f"  Stored ABSOLUTE coordinates: {len(xs)} points")
        print(f"  Range: {min(xs):.1f} - {max(xs):.1f}")
    
        # Save to disk
        success = self.save_tfs_to_disk()
    
        # Update UI
        self.update_tf_selector()
        self.tf_selector.setCurrentText(name)
    
        print(f"✅ TF '{name}' saved")
        return True

    def save_tfs_to_disk(self):
        """Serialize ALL TFs to JSON file safely."""
        print(f"Saving {len(self.saved_tfs)} TFs to disk...")
    
        # Write to a TEMPORARY file first
        temp_file = self.TF_SAVE_FILE + ".tmp"
    
        try:
            # Write to temp file
            with open(temp_file, "w", encoding='utf-8') as f:
                json.dump(self.saved_tfs, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure it's physically written
        
            # Verify the temp file is valid JSON
            with open(temp_file, "r", encoding='utf-8') as f:
                test_load = json.load(f)
        
            # If verification passes, rename to actual file
            os.replace(temp_file, self.TF_SAVE_FILE)
        
            print(f"✅ Successfully saved {len(self.saved_tfs)} TFs: {list(self.saved_tfs.keys())}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to save TFs: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def load_tfs_from_disk(self):
        """Load ALL TFs from JSON file."""
        if os.path.exists(self.TF_SAVE_FILE):
            try:
                with open(self.TF_SAVE_FILE, "r") as f:
                    self.saved_tfs = json.load(f)
                print(f"✅ Loaded {len(self.saved_tfs)} TFs")
            except Exception as e:
                print(f"❌ Load failed: {e}")
                self.saved_tfs = {}
        else:
            print("📁 No saved TFs yet")
            self.saved_tfs = {}

    def load_selected_tf(self, idx, current_data_range=None):
        """Load TF and adapt to current data range."""
        if idx < 0:
            return None
            
        name = self.tf_selector.itemText(idx)
        
        if name not in self.saved_tfs:
            return self.create_default_tf()
        
        tf_data = self.saved_tfs[name]
        
        # Handle relative coordinates
        if 'x_rel' in tf_data and current_data_range:
            min_val, max_val = current_data_range
            range_width = max_val - min_val
            xs = [min_val + rel_x * range_width for rel_x in tf_data['x_rel']]
            return xs, tf_data['y'], tf_data['colors']
        
        # Handle absolute coordinates
        elif 'x_abs' in tf_data:
            return tf_data['x_abs'], tf_data['y'], tf_data['colors']
        
        return self.create_default_tf()

    def update_tf_selector(self):
        """Update combo box with TF names."""
        self.tf_selector.blockSignals(True)
        self.tf_selector.clear()
        
        for name in sorted(self.saved_tfs.keys()):
            self.tf_selector.addItem(name)
            
        if self.tf_selector.count() == 0:
            self.tf_selector.addItem("Default")
            
        self.tf_selector.blockSignals(False)

    def get_initial_tf_data(self, scalar_data, data_range=None):
        """Get initial TF data."""
        if "Default" in self.saved_tfs:
            return self.load_selected_tf(
                list(self.saved_tfs.keys()).index("Default"),
                current_data_range=data_range
            )
        return self.create_default_tf(scalar_data)

    def create_default_tf(self, scalar_data=None):
        """Create a default TF."""
        return [0.0, 255.0], [0.0, 1.0], [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]