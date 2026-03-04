# convert_tfs.py
"""
One-time conversion script for old TF format to new format
Run this ONCE to convert all your saved TFs
"""

import json
import os
import shutil
from datetime import datetime

def convert_saved_tfs(filename="saved_tfs.json"):
    """Convert old tuple format to new dictionary format"""
    
    print("="*60)
    print("TF CONVERSION UTILITY")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"❌ {filename} not found!")
        return False
    
    # Backup original
    backup_name = f"saved_tfs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy(filename, backup_name)
    print(f"✅ Backup created: {backup_name}")
    
    # Load the file
    with open(filename, "r") as f:
        data = json.load(f)
    
    print(f"\n📁 Loaded {len(data)} TFs from {filename}")
    
    # Convert
    converted_data = {}
    conversion_stats = {
        'converted': 0,
        'already_dict': 0,
        'skipped': 0,
        'errors': 0
    }
    
    for name, tf_data in data.items():
        try:
            print(f"\n🔍 Processing '{name}'...")
            
            # Case 1: Already in new dict format
            if isinstance(tf_data, dict):
                # Check if it has required fields
                if 'x_rel' in tf_data or 'x_abs' in tf_data:
                    converted_data[name] = tf_data
                    print(f"  ✅ Already in new format (v{tf_data.get('version', 1)})")
                    conversion_stats['already_dict'] += 1
                else:
                    # Dict but missing required keys - convert
                    print(f"  ⚠️  Dict missing required keys, converting...")
                    
                    # Try to extract whatever we can
                    xs = tf_data.get('x', tf_data.get('points_x', [0, 255]))
                    ys = tf_data.get('y', tf_data.get('points_y', [0, 1]))
                    colors = tf_data.get('colors', tf_data.get('colours', [[1,1,1], [1,1,1]]))
                    
                    converted_data[name] = {
                        'x_abs': [float(x) for x in xs],
                        'y': [float(y) for y in ys],
                        'colors': [list(float(c) for c in color) for color in colors],
                        'version': 1,
                        'converted_from': 'malformed_dict'
                    }
                    print(f"  ✅ Converted malformed dict")
                    conversion_stats['converted'] += 1
            
            # Case 2: Old tuple/list format
            elif isinstance(tf_data, (list, tuple)) and len(tf_data) == 3:
                xs, ys, colors = tf_data
                
                # Ensure proper types
                xs = [float(x) for x in xs]
                ys = [float(y) for y in ys]
                colors = [list(float(c) for c in color) for color in colors]
                
                converted_data[name] = {
                    'x_abs': xs,
                    'y': ys,
                    'colors': colors,
                    'version': 1,
                    'converted_from': 'legacy_tuple'
                }
                print(f"  ✅ Converted from legacy tuple format ({len(xs)} points)")
                conversion_stats['converted'] += 1
            
            # Case 3: Unknown format
            else:
                print(f"  ❌ Unknown format: {type(tf_data)}")
                print(f"  Creating default TF for '{name}'")
                
                converted_data[name] = {
                    'x_abs': [0.0, 255.0],
                    'y': [0.0, 1.0],
                    'colors': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    'version': 1,
                    'note': 'Created from unknown format'
                }
                conversion_stats['skipped'] += 1
                
        except Exception as e:
            print(f"  ❌ Error converting '{name}': {e}")
            # Create safe fallback
            converted_data[name] = {
                'x_abs': [0.0, 255.0],
                'y': [0.0, 1.0],
                'colors': [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                'version': 1,
                'error': str(e)
            }
            conversion_stats['errors'] += 1
    
    # Save converted file
    output_filename = "saved_tfs_converted.json"
    with open(output_filename, "w") as f:
        json.dump(converted_data, f, indent=2)
    
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"✅ Converted file saved as: {output_filename}")
    print(f"\nStatistics:")
    print(f"  - Already in new format: {conversion_stats['already_dict']}")
    print(f"  - Converted from legacy: {conversion_stats['converted']}")
    print(f"  - Created defaults: {conversion_stats['skipped']}")
    print(f"  - Errors (with fallback): {conversion_stats['errors']}")
    print(f"\n📊 Total TFs processed: {len(data)}")
    
    # Show sample of converted data
    print("\n🔍 Sample of first TF in new format:")
    first_name = next(iter(converted_data))
    print(f"  '{first_name}':")
    for key, value in converted_data[first_name].items():
        if key != 'colors':  # Don't print all colors
            print(f"    {key}: {value}")
        else:
            print(f"    colors: [{len(value)} entries]")
    
    print(f"\n📝 Next steps:")
    print(f"  1. Verify the converted file looks correct")
    print(f"  2. Rename it to replace your original:")
    print(f"     cp {output_filename} {filename}")
    print(f"  3. Or keep both and update your code to use the new file")
    
    return True

if __name__ == "__main__":
    convert_saved_tfs()