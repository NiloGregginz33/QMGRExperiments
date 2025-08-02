#!/usr/bin/env python3
"""
Aggressive encoding fix for custom_curvature_experiment.py
"""

import re

def aggressive_fix():
    """Fix all encoding issues aggressively"""
    
    print("=== Aggressive encoding fix ===")
    
    try:
        # Read the file as binary
        with open('src/experiments/custom_curvature_experiment.py', 'rb') as f:
            content = f.read()
        
        # Convert to string and remove all non-printable characters
        try:
            content_str = content.decode('utf-8', errors='ignore')
        except:
            content_str = content.decode('latin-1', errors='ignore')
        
        # Remove all non-printable characters except newlines and tabs
        content_str = re.sub(r'[^\x20-\x7E\n\r\t]', '', content_str)
        
        # Write fixed content
        with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
            f.write(content_str)
        
        print("✓ Aggressive encoding fix completed")
        print("✓ File should now be clean")
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    aggressive_fix() 