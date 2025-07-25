#!/usr/bin/env python3
"""
Script to fix indentation issues in custom_curvature_experiment.py
"""

import re

def fix_indentation_issues():
    # Read the file
    with open('src/experiments/custom_curvature_experiment.py', 'r') as f:
        lines = f.readlines()
    
    # Fix specific indentation issues
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Fix the problematic if-else structure around line 2960
        if i >= 2955 and i <= 2975:
            if 'if not args.fast:' in line:
                fixed_lines.append(line)
                i += 1
                # Fix the next few lines
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('else:'):
                    if 'edge_lengths =' in lines[i]:
                        fixed_lines.append('            ' + lines[i].lstrip())
                    else:
                        fixed_lines.append(lines[i])
                    i += 1
                continue
        
        # Fix the try-except structure around line 3010
        if i >= 3005 and i <= 3020:
            if 'try:' in line:
                fixed_lines.append(line)
                i += 1
                # Fix the if statement and its contents
                while i < len(lines) and 'if args.solve_regge and not args.fast:' not in lines[i]:
                    fixed_lines.append(lines[i])
                    i += 1
                
                if i < len(lines):
                    fixed_lines.append(lines[i])  # The if statement
                    i += 1
                    # Fix the indentation of the next few lines
                    while i < len(lines) and (lines[i].strip().startswith('from ') or 
                                             lines[i].strip().startswith('n = ') or
                                             lines[i].strip().startswith('num_edges =') or
                                             lines[i].strip().startswith('edge_to_tri,')):
                        fixed_lines.append('                ' + lines[i].lstrip())
                        i += 1
                    continue
        
        # Fix the triangle inequality function indentation
        if i >= 3090 and i <= 3110:
            if 'def triangle_ineq(edge_lengths):' in line:
                fixed_lines.append('                ' + line.lstrip())
                i += 1
                # Fix the function body
                while i < len(lines) and (lines[i].strip().startswith('Dmat =') or
                                         lines[i].strip().startswith('cons =') or
                                         lines[i].strip().startswith('for tri in') or
                                         lines[i].strip().startswith('i, j, k =') or
                                         lines[i].strip().startswith('a, b, c =') or
                                         lines[i].strip().startswith('# Stronger triangle') or
                                         lines[i].strip().startswith('margin =') or
                                         lines[i].strip().startswith('cons.append(') or
                                         lines[i].strip().startswith('return np.array(')):
                    if lines[i].strip().startswith('#'):
                        fixed_lines.append('                    ' + lines[i].lstrip())
                    elif lines[i].strip().startswith('return'):
                        fixed_lines.append('                    ' + lines[i].lstrip())
                    else:
                        fixed_lines.append('                    ' + lines[i].lstrip())
                    i += 1
                continue
        
        # Default: keep the line as is
        fixed_lines.append(line)
        i += 1
    
    # Write the fixed file
    with open('src/experiments/custom_curvature_experiment.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print("Indentation issues fixed!")

if __name__ == "__main__":
    fix_indentation_issues() 