#!/usr/bin/env python3
"""
Fix all indentation issues in custom_curvature_experiment.py
"""

import re

# Read the file
with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix specific problematic patterns
fixes = [
    # Fix the try-except structure
    (r'(\s+)try:\s*\n(\s+)if args\.solve_regge and not args\.fast:', 
     r'\1try:\n\2if args.solve_regge and not args.fast:'),
    
    # Fix the minimize function call indentation
    (r'(\s+)result = minimize\(grad_norm, edge_lengths_t, method=\'SLSQP\',\s*\n(\s+)bounds=bounds, constraints=constraints,\s*\n(\s+)options=\{\'ftol\':1e-10, \'maxiter\':2000, \'disp\':False\}\)',
     r'\1result = minimize(grad_norm, edge_lengths_t, method=\'SLSQP\',\n\1    bounds=bounds, constraints=constraints,\n\1    options={\'ftol\':1e-10, \'maxiter\':2000, \'disp\':False})'),
    
    # Fix the Dmat_check line
    (r'(\s+)Dmat_check = edge_lengths_to_matrix\(stationary_edge_lengths, n\)',
     r'\1Dmat_check = edge_lengths_to_matrix(stationary_edge_lengths, n)'),
    
    # Fix the else statement
    (r'(\s+)else:\s*\n(\s+)# Get quantum measurement data',
     r'\1else:\n\1    # Get quantum measurement data'),
    
    # Fix the stationary_edge_lengths assignment
    (r'(\s+)stationary_edge_lengths = edge_lengths_t\s*\n(\s+)# Compute geometric quantities',
     r'\1stationary_edge_lengths = edge_lengths_t\n\1# Compute geometric quantities'),
    
    # Fix the regge_evolution_data append lines
    (r'(\s+)regge_evolution_data\[\'regge_edge_lengths_per_timestep\'\]\.append\(stationary_edge_lengths\.tolist\(\)\)',
     r'\1regge_evolution_data[\'regge_edge_lengths_per_timestep\'].append(stationary_edge_lengths.tolist())'),
    
    # Fix the stationary_solution assignment
    (r'(\s+)stationary_solution = \{',
     r'\1stationary_solution = {'),
    
    # Fix the elif statement
    (r'(\s+)elif args\.solve_regge and args\.fast:',
     r'\1elif args.solve_regge and args.fast:'),
    
    # Fix the else statement after elif
    (r'(\s+)else:\s*\n(\s+)# Create empty regge_evolution_data',
     r'\1else:\n\1    # Create empty regge_evolution_data'),
    
    # Fix the else statement at the end
    (r'(\s+)else:\s*\n(\s+)import traceback',
     r'\1else:\n\1    import traceback'),
    
    # Fix the duplicate else statement
    (r'(\s+)else:\s*\n(\s+)return ricci_scalar',
     r'\1return ricci_scalar'),
]

# Apply fixes
for pattern, replacement in fixes:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

# Write the fixed file
with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed indentation issues!") 