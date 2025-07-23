#!/usr/bin/env python3
"""Test script to check imports."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Add quantum module to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from quantum.rt_surface_oracle import RTSurfaceOracle, create_boundary_graph
    print("✓ RT Surface Oracle imported successfully")
except Exception as e:
    print(f"✗ RT Surface Oracle import failed: {e}")

try:
    from quantum.happy_code import HaPPYCode
    print("✓ HaPPY Code imported successfully")
except Exception as e:
    print(f"✗ HaPPY Code import failed: {e}")

try:
    from quantum.region_decoders import DecoderSynthesizer, create_ewr_test_cases
    print("✓ Region Decoders imported successfully")
except Exception as e:
    print(f"✗ Region Decoders import failed: {e}")

print("All imports tested!") 