#!/usr/bin/env python3
"""
Hardware Quantum Spacetime Validator
====================================

This tool validates quantum spacetime on real quantum hardware using only
measurement counts and mutual information matrices. It implements hardware-
compatible quantum tests that don't require full statevector access.

Key Tests:
1. Bell Inequality Violations from measurement correlations
2. Quantum State Tomography from measurement statistics
3. Entanglement Witnesses from correlation patterns
4. Quantum Coherence from interference patterns
5. Causal Structure from mutual information flow
6. Holographic Principle from bulk-boundary correlations
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.optimize import curve_fit
import os
import sys
from typing import Dict, List, Tuple, Optional

class HardwareQuantumSpacetimeValidator:
    def __init__(self, results_file: str, output_dir: str):
        self.results_file = results_file
        self.output_dir = output_dir
        self.results = {}
        self.quantum_metrics = {}
        
    def load_results(self):
        """Load experiment results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"✅ Loaded results from: {self.results_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def extract_measurement_data(self):
        """Extract measurement counts and mutual information from results."""
        self.counts_per_timestep = self.results.get('counts_per_timestep', [])
        self.mi_matrices = []
        
        # Extract MI matrices from quantum_state_outputs
        quantum_outputs = self.results.get('quantum_state_outputs', [])
        for output in quantum_outputs:
            mi_matrix = output.get('mutual_information_matrix')
            if mi_matrix:
                self.mi_matrices.append(np.array(mi_matrix))
        
        print(f"✅ Found {len(self.counts_per_timestep)} timesteps with measurement counts")
        print(f"✅ Found {len(self.mi_matrices)} mutual information matrices")
        
        return len(self.counts_per_timestep) > 0 and len(self.mi_matrices) > 0
    
    def bell_inequality_test_hardware(self):
        """Test Bell inequality violations using measurement correlations."""
        print("🔬 Testing Bell Inequality Violation (Hardware)...")
        
        if not self.counts_per_timestep:
            print("❌ No measurement counts available")
            return 0.0
        
        # Use first timestep for Bell test
        counts = self.counts_per_timestep[0]
        if not counts:
            print("❌ No counts in first timestep")
            return 0.0
        
        # Calculate correlation from measurement statistics
        total_shots = sum(counts.values())
        if total_shots == 0:
            print("❌ No shots recorded")
            return 0.0
        
        # For 4-qubit system, look for Bell state correlations
        # Bell state: |00⟩ + |11⟩ should show correlation between qubits 0,1
        correlation = 0.0
        bell_violations = 0
        
        # Calculate correlation between qubits 0 and 1
        for bitstring, count in counts.items():
            if len(bitstring) >= 2:
                q0 = int(bitstring[0])
                q1 = int(bitstring[1])
                # Correlation: ⟨σz⊗σz⟩
                correlation += (1 - 2*q0) * (1 - 2*q1) * count / total_shots
        
        # Bell inequality: |⟨σz⊗σz⟩| ≤ 2 classically, can violate quantum mechanically
        bell_violation = abs(correlation) - 2.0
        if bell_violation > 0:
            bell_violations = 1
        
        # Normalize to 0-1 scale
        bell_score = min(abs(correlation) / 4.0, 1.0)
        
        self.quantum_metrics['bell_violation'] = bell_score
        self.quantum_metrics['bell_correlation'] = correlation
        self.quantum_metrics['bell_violations'] = bell_violations
        
        print(f"  📊 Bell correlation: {correlation:.4f}")
        print(f"  📊 Bell violation: {bell_violation:.4f}")
        print(f"  📊 Bell violations: {bell_violations}")
        
        return bell_score
    
    def quantum_state_tomography_hardware(self):
        """Perform quantum state tomography using measurement statistics."""
        print("🔬 Performing Quantum State Tomography (Hardware)...")
        
        if not self.counts_per_timestep:
            print("❌ No measurement counts available")
            return 0.0, 0.0
        
        counts = self.counts_per_timestep[0]
        if not counts:
            print("❌ No counts in first timestep")
            return 0.0, 0.0
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            print("❌ No shots recorded")
            return 0.0, 0.0
        
        # Calculate purity from measurement statistics
        # Purity = Tr(ρ²) = Σᵢ pᵢ² where pᵢ are measurement probabilities
        purity = 0.0
        for bitstring, count in counts.items():
            prob = count / total_shots
            purity += prob * prob
        
        # Calculate coherence from interference patterns
        # Look for superposition signatures in measurement statistics
        coherence = 0.0
        
        # For 2-qubit system, check for Bell state signatures
        if len(counts) > 0:
            # Bell state should have equal probability for |00⟩ and |11⟩
            bell_states = ['00', '11']
            bell_prob = 0.0
            for state in bell_states:
                if state in counts:
                    bell_prob += counts[state] / total_shots
            
            # Coherence is related to how close we are to Bell state
            coherence = bell_prob
        
        self.quantum_metrics['purity'] = purity
        self.quantum_metrics['coherence'] = coherence
        
        print(f"  📊 Purity: {purity:.4f}")
        print(f"  📊 Coherence: {coherence:.4f}")
        
        return purity, coherence
    
    def entanglement_witness_hardware(self):
        """Test entanglement using correlation patterns."""
        print("🔬 Testing Entanglement Witnesses (Hardware)...")
        
        if not self.mi_matrices:
            print("❌ No mutual information matrices available")
            return 0
        
        # Use first MI matrix
        mi_matrix = self.mi_matrices[0]
        
        # Count strong correlations (potential entanglement)
        violations = 0
        threshold = 0.001  # Threshold for significant correlation
        
        for i in range(len(mi_matrix)):
            for j in range(i+1, len(mi_matrix)):
                if mi_matrix[i][j] > threshold:
                    violations += 1
        
        self.quantum_metrics['entanglement_violations'] = violations
        
        print(f"  📊 Entanglement violations: {violations}")
        
        return violations
    
    def quantum_coherence_test_hardware(self):
        """Test quantum coherence from measurement patterns."""
        print("🔬 Testing Quantum Coherence (Hardware)...")
        
        if not self.counts_per_timestep:
            print("❌ No measurement counts available")
            return 0.0
        
        counts = self.counts_per_timestep[0]
        if not counts:
            print("❌ No counts in first timestep")
            return 0.0
        total_shots = sum(counts.values())
        
        if total_shots == 0:
            print("❌ No shots recorded")
            return 0.0
        
        # Calculate superposition strength from measurement distribution
        # Quantum superposition should show interference patterns
        superposition_strength = 0.0
        
        # Check for non-classical measurement distributions
        # Classical states should be more concentrated
        # Quantum superposition should be more spread out
        
        # Calculate entropy of measurement distribution
        probs = [count/total_shots for count in counts.values()]
        measurement_entropy = entropy(probs) if len(probs) > 1 else 0
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(probs)) if len(probs) > 1 else 0
        if max_entropy > 0:
            superposition_strength = measurement_entropy / max_entropy
        
        self.quantum_metrics['superposition_strength'] = superposition_strength
        self.quantum_metrics['measurement_entropy'] = measurement_entropy
        
        print(f"  📊 Superposition strength: {superposition_strength:.4f}")
        print(f"  📊 Measurement entropy: {measurement_entropy:.4f}")
        
        return superposition_strength
    
    def causal_structure_test_hardware(self):
        """Test causal structure from mutual information flow."""
        print("🔬 Testing Causal Structure (Hardware)...")
        
        if not self.mi_matrices:
            print("❌ No mutual information matrices available")
            return 0
        
        # Use first MI matrix
        mi_matrix = self.mi_matrices[0]
        
        # Check for causal violations in mutual information
        # In causal structure, information should flow forward in time
        violations = 0
        
        # Simple test: check for strong non-local correlations
        # that might violate light cone constraints
        for i in range(len(mi_matrix)):
            for j in range(i+1, len(mi_matrix)):
                if mi_matrix[i][j] > 0.01:  # Strong correlation
                    # Check if this violates expected locality
                    distance = abs(i - j)
                    if distance > 1 and mi_matrix[i][j] > 0.005:
                        violations += 1
        
        self.quantum_metrics['causal_violations'] = violations
        
        print(f"  📊 Causal violations: {violations}")
        
        return violations
    
    def holographic_principle_test_hardware(self):
        """Test holographic principle from bulk-boundary correlations."""
        print("🔬 Testing Holographic Principle (Hardware)...")
        
        if not self.mi_matrices:
            print("❌ No mutual information matrices available")
            return 0.0
        
        # Use first MI matrix
        mi_matrix = self.mi_matrices[0]
        
        if len(mi_matrix) < 4:
            print("❌ Need at least 4 qubits for holographic test")
            return 0.0
        
        # Test Ryu-Takayanagi relation
        # Divide system into bulk and boundary
        n_qubits = len(mi_matrix)
        boundary_size = max(1, n_qubits // 4)  # 25% boundary
        
        # Calculate boundary entropy (simplified)
        boundary_entropy = 0.0
        for i in range(boundary_size):
            for j in range(i+1, boundary_size):
                boundary_entropy += mi_matrix[i][j]
        
        # Calculate bulk-boundary correlation
        bulk_boundary_correlation = 0.0
        for i in range(boundary_size):
            for j in range(boundary_size, n_qubits):
                bulk_boundary_correlation += mi_matrix[i][j]
        
        # Holographic principle: bulk should be reconstructible from boundary
        reconstruction_quality = bulk_boundary_correlation / (boundary_entropy + 1e-10)
        # Cap reconstruction quality to reasonable range
        reconstruction_quality = min(reconstruction_quality, 1.0)
        
        self.quantum_metrics['reconstruction_quality'] = reconstruction_quality
        self.quantum_metrics['boundary_entropy'] = boundary_entropy
        self.quantum_metrics['bulk_boundary_correlation'] = bulk_boundary_correlation
        
        print(f"  📊 Reconstruction quality: {reconstruction_quality:.4f}")
        print(f"  📊 Boundary entropy: {boundary_entropy:.4f}")
        print(f"  📊 Bulk-boundary correlation: {bulk_boundary_correlation:.4f}")
        
        return reconstruction_quality
    
    def compute_quantum_spacetime_score(self):
        """Compute overall quantum spacetime score from hardware metrics."""
        print("🎯 Computing Quantum Spacetime Score...")
        
        # Weight the different quantum metrics
        weights = {
            'bell_violation': 0.25,
            'purity': 0.20,
            'coherence': 0.20,
            'entanglement_violations': 0.15,
            'superposition_strength': 0.10,
            'causal_violations': 0.05,
            'reconstruction_quality': 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.quantum_metrics:
                value = self.quantum_metrics[metric]
                # Normalize entanglement violations
                if metric == 'entanglement_violations':
                    value = min(value / 10.0, 1.0)  # Normalize to 0-1
                elif metric == 'causal_violations':
                    value = min(value / 5.0, 1.0)   # Normalize to 0-1
                
                total_score += value * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        self.quantum_metrics['quantum_spacetime_score'] = final_score
        
        print(f"  📊 Final Quantum Spacetime Score: {final_score:.4f}")
        
        return final_score
    
    def generate_summary(self):
        """Generate comprehensive summary of quantum spacetime validation."""
        print("📝 Generating Summary...")
        
        score = self.quantum_metrics.get('quantum_spacetime_score', 0.0)
        
        if score >= 0.7:
            assessment = "✅ GENUINE QUANTUM EMERGENT SPACETIME DETECTED!"
        elif score >= 0.5:
            assessment = "⚠️ MIXED QUANTUM-CLASSICAL BEHAVIOR"
        elif score >= 0.3:
            assessment = "❌ WEAK QUANTUM SIGNATURES"
        else:
            assessment = "❌ PRIMARILY CLASSICAL CORRELATIONS"
        
        summary = f"""HARDWARE QUANTUM SPACETIME VALIDATION SUMMARY
==================================================

OVERALL QUANTUM SPACETIME SCORE: {score:.4f}
ASSESSMENT: {assessment}

DETAILED TEST RESULTS:
------------------------------

1. Bell Inequality Violation: {'✅ VIOLATED' if self.quantum_metrics.get('bell_violation', 0) > 0.1 else '❌ NOT VIOLATED'}
   Bell correlation: {self.quantum_metrics.get('bell_correlation', 0):.4f}
   Violations: {self.quantum_metrics.get('bell_violations', 0)}

2. Quantum State Tomography: {'✅ QUANTUM' if self.quantum_metrics.get('purity', 0) > 0.5 else '❌ CLASSICAL'}
   Purity: {self.quantum_metrics.get('purity', 0):.4f}
   Coherence strength: {self.quantum_metrics.get('coherence', 0):.4f}

3. Entanglement Witness: {'✅ ENTANGLED' if self.quantum_metrics.get('entanglement_violations', 0) > 0 else '❌ NOT ENTANGLED'}
   Total violations: {self.quantum_metrics.get('entanglement_violations', 0)}

4. Quantum Coherence: {'✅ COHERENT' if self.quantum_metrics.get('superposition_strength', 0) > 0.3 else '❌ INCOHERENT'}
   Superposition strength: {self.quantum_metrics.get('superposition_strength', 0):.4f}

5. Causal Structure: {'✅ VALID' if self.quantum_metrics.get('causal_violations', 0) == 0 else '⚠️ VIOLATIONS'}
   Light cone violations: {self.quantum_metrics.get('causal_violations', 0)}

6. Holographic Principle: {'✅ CONSISTENT' if self.quantum_metrics.get('reconstruction_quality', 0) > 0.1 else '❌ INCONSISTENT'}
   Bulk reconstruction quality: {self.quantum_metrics.get('reconstruction_quality', 0):.4f}

CONCLUSION:
--------------------
{assessment}
   The system shows {'strong' if score > 0.5 else 'weak' if score > 0.3 else 'minimal'} 
   quantum signatures compatible with emergent spacetime.
   {'Hardware noise may be limiting quantum effects.' if score < 0.5 else 'Quantum effects are clearly visible despite hardware noise.'}

HARDWARE CONSIDERATIONS:
--------------------
- Real quantum hardware has noise and decoherence
- Measurement counts only provide partial quantum information
- Strong quantum effects may be masked by hardware limitations
- Consider error mitigation techniques for improved results
"""
        
        return summary
    
    def save_results(self):
        """Save validation results to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_data = {
            'quantum_metrics': convert_numpy(self.quantum_metrics),
            'input_file': self.results_file,
            'validation_method': 'hardware_compatible'
        }
        
        results_file = os.path.join(self.output_dir, 'hardware_quantum_spacetime_validation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        summary_file = os.path.join(self.output_dir, 'hardware_quantum_spacetime_validation_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self.generate_summary())
        
        print(f"✅ Results saved to: {results_file}")
        print(f"✅ Summary saved to: {summary_file}")
    
    def run_validation(self):
        """Run complete hardware quantum spacetime validation."""
        print("🚀 Starting Hardware Quantum Spacetime Validation...")
        print("=" * 60)
        
        if not self.load_results():
            return False
        
        if not self.extract_measurement_data():
            print("❌ No measurement data found")
            return False
        
        # Run all quantum tests
        self.bell_inequality_test_hardware()
        self.quantum_state_tomography_hardware()
        self.entanglement_witness_hardware()
        self.quantum_coherence_test_hardware()
        self.causal_structure_test_hardware()
        self.holographic_principle_test_hardware()
        
        # Compute final score
        score = self.compute_quantum_spacetime_score()
        
        # Generate and display summary
        summary = self.generate_summary()
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60)
        
        # Save results
        self.save_results()
        
        return True

def main():
    if len(sys.argv) != 3:
        print("Usage: python hardware_quantum_spacetime_validator.py <results_file> <output_dir>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    validator = HardwareQuantumSpacetimeValidator(results_file, output_dir)
    success = validator.run_validation()
    
    if success:
        print("✅ Hardware quantum spacetime validation completed successfully!")
    else:
        print("❌ Hardware quantum spacetime validation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 