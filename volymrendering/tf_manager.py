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

    def save_current_tf(self, points_x, points_y, colors):
        """Save current transfer function with a name."""
        name, ok = QtWidgets.QInputDialog.getText(
            self.parent_window, "Save Transfer Function", "Name:"
        )
        if ok and name:
            # Create deep copies
            xs = [float(x) for x in points_x]
            ys = [float(y) for y in points_y]
            colors = [tuple(float(c) for c in color) for color in colors]
            
            print(f"\n=== Saving TF '{name}' ===")
            print(f"Before save - Current TFs: {list(self.saved_tfs.keys())}")
            
            # Add/update the TF
            self.saved_tfs[name] = (xs, ys, colors)
            
            print(f"After adding - Current TFs: {list(self.saved_tfs.keys())}")
            
            # Save ALL TFs to disk
            success = self.save_tfs_to_disk()
            
            # Update UI
            self.update_tf_selector()
            self.tf_selector.setCurrentText(name)
            
            print(f"TF '{name}' saved successfully: {success}")
            return success
        return False

    def save_tfs_to_disk(self):
        """Serialize ALL TFs to JSON file."""
        print(f"Saving {len(self.saved_tfs)} TFs to disk...")
        
        out = {}
        for name, (xs, ys, colors) in self.saved_tfs.items():
            out[name] = {
                "x": xs,
                "y": ys,
                "colors": [list(c) for c in colors]
            }
        
        try:
            with open(self.TF_SAVE_FILE, "w") as f:
                json.dump(out, f, indent=2)
            print(f"✅ Successfully saved {len(out)} TFs: {list(out.keys())}")
            return True
        except Exception as e:
            print(f"❌ Failed to save TFs: {e}")
            return False

    def load_tfs_from_disk(self):
        """Load ALL TFs from JSON file if it exists."""
        print("Loading TFs from disk...")
        
        if os.path.exists(self.TF_SAVE_FILE):
            try:
                file_size = os.path.getsize(self.TF_SAVE_FILE)
                print(f"TF file exists, size: {file_size} bytes")
                
                with open(self.TF_SAVE_FILE, "r") as f:
                    data = json.load(f)
                
                print(f"📁 Loaded JSON data with {len(data)} TFs: {list(data.keys())}")
                
                # Clear and reload ALL TFs
                self.saved_tfs.clear()
                loaded_count = 0
                
                for name, tf in data.items():
                    try:
                        xs = [float(x) for x in tf["x"]]
                        ys = [float(y) for y in tf["y"]]
                        colors = [tuple(float(c) for c in color) for color in tf["colors"]]
                        self.saved_tfs[name] = (xs, ys, colors)
                        loaded_count += 1
                        print(f"  ✅ Loaded '{name}' with {len(xs)} points")
                    except Exception as e:
                        print(f"  ❌ Error loading TF '{name}': {e}")
                
                print(f"Successfully loaded {loaded_count}/{len(data)} TFs")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                self.saved_tfs = {}
            except Exception as e:
                print(f"❌ Failed to load TFs: {e}")
                self.saved_tfs = {}
        else:
            print(f"📁 TF file {self.TF_SAVE_FILE} does not exist yet")
            self.saved_tfs = {}

    def load_selected_tf(self, idx):
        """Load TF selected from combo box."""
        if idx < 0:
            print("❌ Invalid index in load_selected_tf")
            return None
            
        name = self.tf_selector.itemText(idx)
        print(f"\n=== Loading TF: '{name}' ===")
        print(f"Available TFs: {list(self.saved_tfs.keys())}")
        
        if name in self.saved_tfs:
            print(f"✅ Successfully loading TF: {name}")
            return self.saved_tfs[name]
        else:
            print(f"❌ TF '{name}' not found in saved_tfs")
            return None

    def update_tf_selector(self):
        """Refresh the combo box with saved TF names."""
        self.tf_selector.blockSignals(True)
        self.tf_selector.clear()
        
        print(f"Updating TF selector with {len(self.saved_tfs)} TFs: {list(self.saved_tfs.keys())}")
        
        # Add all saved TFs in alphabetical order
        for name in sorted(self.saved_tfs.keys()):
            self.tf_selector.addItem(name)
            
        # If no TFs exist, create a default placeholder
        if self.tf_selector.count() == 0:
            print("No TFs found, adding Default placeholder")
            self.tf_selector.addItem("Default")
            
        self.tf_selector.blockSignals(False)
        print(f"TF selector now has {self.tf_selector.count()} items")

    def get_initial_tf_data(self, scalar_data):
        """Get initial TF data - use Default if exists, otherwise create one."""
        print("\n=== Getting Initial TF Data ===")
        print(f"Available TFs: {list(self.saved_tfs.keys())}")
        
        if "Default" in self.saved_tfs:
            print("Using existing 'Default' TF")
            return self.saved_tfs["Default"]
        elif self.saved_tfs:
            # Use the first available TF
            first_tf_name = next(iter(self.saved_tfs.keys()))
            print(f"Using first available TF: '{first_tf_name}'")
            return self.saved_tfs[first_tf_name]
        else:
            # Create a default TF
            print("Creating new default TF")
            return self.create_default_tf(scalar_data)

    def create_default_tf(self, scalar_data):
        """Create default TF based on data histogram."""
        try:
            hist, bins = np.histogram(scalar_data, bins=256, range=(0, 255))
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            peaks = np.where(hist > hist.max() * 0.05)[0]
            
            if len(peaks) < 2:
                points_x = [0.0, 255.0]
                points_y = [0.0, 1.0]
                colors = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]
            else:
                points_x = list(bin_centers[peaks])
                points_y = list(np.clip(hist[peaks] / hist.max(), 0.0, 1.0))
                colors = [(1.0, 1.0, 1.0) for _ in points_x]
            
            print(f"Created default TF with {len(points_x)} points")
            return points_x, points_y, colors
        except Exception as e:
            print(f"Error creating default TF: {e}")
            # Fallback TF
            return [0.0, 255.0], [0.0, 1.0], [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]