#!/usr/bin/env python3
"""
Test script for emergent geometry teleportation integration in custom curvature experiment.
"""

import sys
import os
import subprocess
import tempfile
import json

def test_emergent_teleportation_integration():
    """Test the emergent geometry teleportation integration."""
    
    print("🧪 Testing Emergent Geometry Teleportation Integration")
    print("=" * 60)
    
    # Test parameters
    test_params = {
        '--num_qubits': '5',
        '--geometry': 'hyperbolic',
        '--curvature': '1.0',
        '--timesteps': '2',
        '--shots': '100',
        '--device': 'simulator',
        '--emergent_geometry_teleportation': '',
        '--teleportation_embedding_dim': '2',
        '--teleportation_node_pairs': 'auto',
        '--teleportation_fidelity_threshold': '0.7'
    }
    
    # Build command
    cmd = ['python', 'src/experiments/custom_curvature_experiment.py']
    for param, value in test_params.items():
        if value:
            cmd.extend([param, value])
        else:
            cmd.append(param)
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        print()
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            print()
        
        # Check if the experiment completed successfully
        if result.returncode == 0:
            print("✅ Experiment completed successfully!")
            
            # Look for teleportation analysis in output
            if "Emergent geometry teleportation analysis completed" in result.stdout:
                print("✅ Teleportation analysis executed successfully!")
                
                # Extract teleportation results
                if "Fidelity-distance correlation:" in result.stdout:
                    print("✅ Teleportation correlation analysis completed!")
                
                if "Node pairs tested:" in result.stdout:
                    print("✅ Teleportation node pair testing completed!")
                
                print("\n🎉 All teleportation integration tests passed!")
                return True
            else:
                print("❌ Teleportation analysis not found in output")
                return False
        else:
            print(f"❌ Experiment failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Experiment timed out")
        return False
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False

def test_teleportation_functions():
    """Test the teleportation functions directly."""
    
    print("\n🔧 Testing Teleportation Functions")
    print("=" * 40)
    
    try:
        # Import the functions
        sys.path.append('src')
        from experiments.custom_curvature_experiment import (
            compute_emergent_geometry_teleportation,
            analyze_teleportation_geometry_correlation,
            create_teleportation_geometry_plots
        )
        
        print("✅ Successfully imported teleportation functions")
        
        # Test with dummy data
        import numpy as np
        
        # Create dummy MI matrix
        num_qubits = 5
        mi_matrix = np.random.rand(num_qubits, num_qubits)
        mi_matrix = (mi_matrix + mi_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(mi_matrix, 0)  # Zero diagonal
        
        print("✅ Created test mutual information matrix")
        
        # Test teleportation analysis
        teleportation_results = compute_emergent_geometry_teleportation(
            mi_matrix=mi_matrix,
            num_qubits=num_qubits,
            node_pairs="auto",
            embedding_dim=2,
            device_name="simulator",
            shots=10
        )
        
        if teleportation_results and 'fidelities' in teleportation_results:
            print("✅ Teleportation analysis function works!")
            print(f"   - Fidelities computed: {len(teleportation_results['fidelities'])}")
            print(f"   - Correlation: {teleportation_results.get('fidelity_distance_correlation', 0):.3f}")
        else:
            print("❌ Teleportation analysis function failed")
            return False
        
        # Test correlation analysis
        curvature_results = {
            'geometry': 'hyperbolic',
            'curvature': 1.0,
            'gromov_delta': 0.1,
            'mean_distance': 1.5,
            'angle_sums': [np.pi, np.pi, np.pi]
        }
        
        correlation_analysis = analyze_teleportation_geometry_correlation(
            teleportation_results, curvature_results, 'hyperbolic'
        )
        
        if correlation_analysis and 'insights' in correlation_analysis:
            print("✅ Correlation analysis function works!")
            print(f"   - Insights generated: {len(correlation_analysis['insights'])}")
        else:
            print("❌ Correlation analysis function failed")
            return False
        
        print("\n🎉 All teleportation function tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Function test error: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 Emergent Geometry Teleportation Integration Test Suite")
    print("=" * 70)
    
    # Test 1: Function imports and basic functionality
    test1_passed = test_teleportation_functions()
    
    # Test 2: Full integration test
    test2_passed = test_emergent_teleportation_integration()
    
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Teleportation Functions Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Integration Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Integration is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the integration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 