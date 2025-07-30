#!/usr/bin/env python3
"""
Fix Unicode and Emoji Issues in Custom Curvature Experiment

This script removes problematic Unicode characters and emojis that cause
encoding issues on Windows systems.
"""

import re
import os
import argparse
import sys

def fix_unicode_emojis(file_path):
    """Replace problematic Unicode characters with ASCII equivalents."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacements for problematic characters
    replacements = {
        '🚀': '[ROCKET]',
        '🔬': '[MICROSCOPE]',
        '✅': '[CHECK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '•': '-',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '⇒': '=>',
        '⇐': '<=',
        '⇑': '^^',
        '⇓': 'vv',
        '≈': '~',
        '≠': '!=',
        '≤': '<=',
        '≥': '>=',
        '±': '+/-',
        '×': 'x',
        '÷': '/',
        '∞': 'infinity',
        '∑': 'sum',
        '∫': 'integral',
        '∂': 'partial',
        '∇': 'nabla',
        'Δ': 'delta',
        'θ': 'theta',
        'φ': 'phi',
        'ψ': 'psi',
        'α': 'alpha',
        'β': 'beta',
        'γ': 'gamma',
        'δ': 'delta',
        'ε': 'epsilon',
        'ζ': 'zeta',
        'η': 'eta',
        'ι': 'iota',
        'κ': 'kappa',
        'λ': 'lambda',
        'μ': 'mu',
        'ν': 'nu',
        'ξ': 'xi',
        'ο': 'omicron',
        'π': 'pi',
        'ρ': 'rho',
        'σ': 'sigma',
        'τ': 'tau',
        'υ': 'upsilon',
        'χ': 'chi',
        'ω': 'omega',
        '∝': 'proportional to',
        'Λ': 'Lambda',
        'Σ': 'Sigma',
        '⟨': '<',
        '⟩': '>',
        '²': '^2',
        '³': '^3',
        '₁': '_1',
        '₂': '_2',
        '₃': '_3',
        '₄': '_4',
        '₅': '_5',
        '₆': '_6',
        '₇': '_7',
        '₈': '_8',
        '₉': '_9',
        '₀': '_0'
    }
    
    # Apply replacements
    for unicode_char, ascii_replacement in replacements.items():
        content = content.replace(unicode_char, ascii_replacement)
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed Unicode characters in {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Fix Unicode and emoji issues in files')
    parser.add_argument('--file', type=str, help='File to fix Unicode characters in')
    parser.add_argument('files', nargs='*', help='Files to fix Unicode characters in (positional arguments)')
    
    args = parser.parse_args()
    
    # Determine which files to process
    files_to_process = []
    
    if args.file:
        files_to_process.append(args.file)
    
    if args.files:
        files_to_process.extend(args.files)
    
    # If no files specified, use default behavior
    if not files_to_process:
        # Default behavior - fix the custom curvature experiment file
        experiment_file = "src/experiments/custom_curvature_experiment.py"
        
        if os.path.exists(experiment_file):
            fix_unicode_emojis(experiment_file)
            print("[SUCCESS] Unicode fixes applied to custom curvature experiment!")
        else:
            print(f"[ERROR] File not found: {experiment_file}")
        
        # Fix the analyze experiment script
        analyze_file = "tools/analyze_experiment.py"
        
        if os.path.exists(analyze_file):
            fix_unicode_emojis(analyze_file)
            print("[SUCCESS] Unicode fixes applied to analyze experiment script!")
        else:
            print(f"[ERROR] File not found: {analyze_file}")
        
        print("[SUCCESS] All Unicode fixes applied successfully!")
        return
    
    # Process specified files
    success_count = 0
    total_count = len(files_to_process)
    
    for file_path in files_to_process:
        if fix_unicode_emojis(file_path):
            success_count += 1
    
    print(f"\n[SUMMARY] Successfully processed {success_count}/{total_count} files")
    
    if success_count == total_count:
        print("[SUCCESS] All files processed successfully!")
    else:
        print("[WARNING] Some files could not be processed. Check the output above for details.")

if __name__ == "__main__":
    main() 