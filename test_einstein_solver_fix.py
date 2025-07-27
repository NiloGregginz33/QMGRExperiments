#!/usr/bin/env python3
"""
Test script to verify Einstein solver fix for spherical geometry
"""

import sys
import os
sys.path.append('src')

from experiments.custom_curvature_experiment import (
    compute_curvature_tensor_from_entanglement,
    compute_einstein_tensor,
    analyze_einstein_entanglement_relation
)
import numpy as np

def test_spherical_ricci_scalar():
    """Test that spherical geometry produces positive Ricci scalar"""
    
    print("🧪 Testing Einstein Solver Fix for Spherical Geometry")
    print("=" * 60)
    
    # Create test data
    num_qubits = 11
    mi_matrix = np.ones((num_qubits, num_qubits)) * 0.1  # Small MI values
    coordinates = np.random.randn(num_qubits, 2)  # Random coordinates
    entropy_per_timestep = [0.5, 0.6, 0.7, 0.8, 0.9]  # Sample entropy evolution
    
    # Test spherical geometry
    print("\n📊 Testing SPHERICAL geometry:")
    spherical_analysis = analyze_einstein_entanglement_relation(
        mi_matrix, coordinates, entropy_per_timestep, num_qubits, geometry="spherical"
    )
    
    ricci_scalar_spherical = spherical_analysis['ricci_scalar']
    print(f"  Ricci Scalar: {ricci_scalar_spherical:.6f}")
    print(f"  Expected: R > 0 for spherical geometry")
    print(f"  Result: {'✅ POSITIVE' if ricci_scalar_spherical > 0 else '❌ NEGATIVE'}")
    
    # Test hyperbolic geometry for comparison
    print("\n📊 Testing HYPERBOLIC geometry:")
    hyperbolic_analysis = analyze_einstein_entanglement_relation(
        mi_matrix, coordinates, entropy_per_timestep, num_qubits, geometry="hyperbolic"
    )
    
    ricci_scalar_hyperbolic = hyperbolic_analysis['ricci_scalar']
    print(f"  Ricci Scalar: {ricci_scalar_hyperbolic:.6f}")
    print(f"  Expected: R < 0 for hyperbolic geometry")
    print(f"  Result: {'✅ NEGATIVE' if ricci_scalar_hyperbolic < 0 else '❌ POSITIVE'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print(f"  Spherical Ricci Scalar: {ricci_scalar_spherical:.6f} {'✅' if ricci_scalar_spherical > 0 else '❌'}")
    print(f"  Hyperbolic Ricci Scalar: {ricci_scalar_hyperbolic:.6f} {'✅' if ricci_scalar_hyperbolic < 0 else '❌'}")
    
    if ricci_scalar_spherical > 0 and ricci_scalar_hyperbolic < 0:
        print("\n🎉 SUCCESS: Einstein solver now correctly handles both geometries!")
        return True
    else:
        print("\n❌ FAILURE: Einstein solver still has issues")
        return False

if __name__ == "__main__":
    success = test_spherical_ricci_scalar()
    sys.exit(0 if success else 1) 