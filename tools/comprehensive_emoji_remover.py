#!/usr/bin/env python3
"""
Comprehensive Emoji Remover for Custom Curvature Experiment

This script removes all emojis and problematic Unicode characters that cause
encoding issues on Windows systems.
"""

import re
import os
import sys

def remove_all_emojis(file_path):
    """Remove all emojis and problematic Unicode characters."""
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Comprehensive emoji replacements
    replacements = {
        '🚀': '[ROCKET]',
        '📊': '[DATA]',
        '✅': '[CHECK]',
        '❌': '[ERROR]',
        '⚠️': '[WARNING]',
        '⚡': '[FAST]',
        '🎯': '[TARGET]',
        '💾': '[SAVE]',
        '⚙️': '[SETTINGS]',
        '🔬': '[MICROSCOPE]',
        '🌌': '[GEOMETRY]',
        '📐': '[CURVATURE]',
        '⚛️': '[QUBITS]',
        '⏰': '[TIMESTEPS]',
        '🖥️': '[DEVICE]',
        '🌍': '[EARTH]',
        '🕳️': '[BLACK_HOLE]',
        '🎉': '[SUCCESS]',
        '⏱️': '[TIME]',
        '🕐': '[CLOCK]',
        '📈': '[CHART]',
        '🔄': '[LOOP]',
        '📁': '[FOLDER]',
        '📄': '[FILE]',
        '🔗': '[LINK]',
        '🔧': '[TOOLS]',
        '📊': '[DATA]',
        '🕳️': '[BLACK_HOLE]',
        '📈': '[CHART]',
        '💾': '[SAVE]',
        '🎉': '[SUCCESS]',
        '📊': '[DATA]',
        '⏱️': '[TIME]',
        '🕐': '[CLOCK]',
        '📈': '[CHART]',
        '🔄': '[LOOP]',
        '⚡': '[FAST]',
        '📁': '[FOLDER]',
        '💾': '[SAVE]',
        '📄': '[FILE]',
        '🔗': '[LINK]',
        '🔬': '[SCIENCE]',
        '🌌': '[GEOMETRY]',
        '📐': '[CURVATURE]',
        '⚛️': '[QUBITS]',
        '⏰': '[TIMESTEPS]',
        '🖥️': '[DEVICE]',
        '⚡': '[FAST]',
        '🌍': '[EARTH]',
        '🕳️': '[BLACK_HOLE]'
    }
    
    # Apply replacements
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Also remove any remaining emoji-like characters using regex
    # This catches any emojis we might have missed
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    content = emoji_pattern.sub('[EMOJI]', content)
    
    # Write the fixed content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Removed all emojis from {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_emoji_remover.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    remove_all_emojis(file_path)

if __name__ == "__main__":
    main() 