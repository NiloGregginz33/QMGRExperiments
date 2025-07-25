#!/usr/bin/env python3
"""
Fix indentation issues line by line
"""

# Read the file
with open('src/experiments/custom_curvature_experiment.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix specific lines
fixed_lines = []
for i, line in enumerate(lines):
    # Fix line 3022 - regge_evolution_data
    if i == 3021:  # 0-indexed
        fixed_lines.append('            regge_evolution_data = {\n')
    # Fix line 3023-3027 - the dictionary contents
    elif i in [3022, 3023, 3024, 3025, 3026]:
        fixed_lines.append('                ' + line.lstrip())
    # Fix line 3029-3030 - the print statements (they should be at the same level as the dictionary)
    elif i == 3028:  # 0-indexed
        fixed_lines.append('            }\n')
    elif i == 3029:  # 0-indexed
        fixed_lines.append('            print(f"🔍 DEBUG: distmat_per_timestep length: {len(distmat_per_timestep)}")\n')
    elif i == 3030:  # 0-indexed
        fixed_lines.append('            print(f"🔍 DEBUG: timesteps: {args.timesteps}")\n')
    # Fix line 3033 - the total_action function definition
    elif i == 3032:  # 0-indexed
        fixed_lines.append('            # Refactor: total_action and total_gradient always take a \'matter\' argument\n')
    elif i == 3033:  # 0-indexed
        fixed_lines.append('            def total_action(edge_lengths, matter):\n')
    # Fix line 3174 - the minimize function call
    elif i == 3173:  # 0-indexed
        fixed_lines.append('                result = minimize(grad_norm, edge_lengths_t, method=\'SLSQP\',\n')
    # Fix line 3175-3177 - the function arguments
    elif i in [3174, 3175, 3176]:
        fixed_lines.append('                    ' + line.lstrip())
    else:
        fixed_lines.append(line)

# Write the fixed file
with open('src/experiments/custom_curvature_experiment.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("Fixed total_action function indentation!") 