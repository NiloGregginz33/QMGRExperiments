#!/usr/bin/env python3
"""
Fix Unicode and Emoji Issues in Custom Curvature Experiment

This script removes problematic Unicode characters and emojis that cause
encoding issues on Windows systems.
"""

import re
import os

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
    # Fix the custom curvature experiment file
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

if __name__ == "__main__":
    main() 