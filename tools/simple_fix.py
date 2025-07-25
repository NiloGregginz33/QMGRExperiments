#!/usr/bin/env python3
"""
Simple script to fix the specific indentation error on line 3101
"""

# Read the file
with open('src/experiments/custom_curvature_experiment.py', 'r') as f:
    lines = f.readlines()

# Fix line 3101 (index 3100)
if len(lines) > 3100:
    # The problematic line is likely a comment that's over-indented
    line_3101 = lines[3100]
    if line_3101.strip().startswith('#'):
        # Fix the indentation of the comment
        lines[3100] = '                    ' + line_3101.lstrip()
    elif line_3101.strip().startswith('margin ='):
        # Fix the indentation of the margin assignment
        lines[3100] = '                        ' + line_3101.lstrip()

# Write the fixed file
with open('src/experiments/custom_curvature_experiment.py', 'w') as f:
    f.writelines(lines)

print("Fixed line 3101!") 