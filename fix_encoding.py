#!/usr/bin/env python3
"""
Fix encoding issue in custom_curvature_experiment.py
"""

def fix_encoding():
    """Fix the encoding issue in the file"""
    
    print("=== Fixing encoding issues ===")
    
    try:
        # Read the file as binary
        with open('src/experiments/custom_curvature_experiment.py', 'rb') as f:
            content = f.read()
        
        # Remove problematic characters
        content = content.replace(b'\x00', b'')  # Remove null bytes
        content = content.replace(b'\xff', b'')  # Remove 0xFF bytes
        
        # Write fixed content back to the original file
        with open('src/experiments/custom_curvature_experiment.py', 'wb') as f:
            f.write(content)
        
        print("✓ Encoding issues fixed")
        print("✓ File is now ready to run")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    fix_encoding() 