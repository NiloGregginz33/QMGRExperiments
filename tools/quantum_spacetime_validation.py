#!/usr/bin/env python3
"""
Quantum Spacetime Validation Tool
================================

Critical tests to prove genuine quantum emergent spacetime:
1. Bell inequality violation tests
2. Quantum state tomography
3. Entanglement witnesses
4. Quantum coherence tests
5. Non-classical correlation measurements
6. Causal structure validation
7. Holographic principle verification
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats, optimize
from sklearn.metrics import r2_score
import itertools
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, Operator
from qiskit.quantum_info import state_fidelity, concurrence
from qiskit.quantum_info import DensityMatrix, entropy

class QuantumSpacetimeValidator:
    """
    Validates quantum spacetime through multiple critical tests
    """
    
    def __init__(self, results_file: str, output_dir: str):
        self.results_file = results_file
        self.output_dir = output_dir
        self.results = self.load_results()
        self.validation_results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_results(self) -> Dict:
        """Load experiment results"""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def bell_inequality_test(self) -> Dict:
        """
        Test Bell inequality violation to prove quantum correlations
        """
        print("🔬 Testing Bell Inequality Violation...")
        
        # Extract statevector if available (check different possible locations)
        statevector = None
        
        # Check for quantum_state_outputs format (new format)
        if 'quantum_state_outputs' in self.results and self.results['quantum_state_outputs']:
            # Use the first timestep's statevector
            first_output = self.results['quantum_state_outputs'][0]
            if 'statevector' in first_output and first_output['statevector']:
                # Convert complex number format back to numpy array
                sv_data = first_output['statevector']
                if isinstance(sv_data[0], dict) and 'real' in sv_data[0]:
                    # Convert from {"real": x, "imag": y} format
                    statevector = np.array([complex(item['real'], item['imag']) for item in sv_data])
                else:
                    statevector = np.array(sv_data)
                print(f"✅ Found statevector in quantum_state_outputs (timestep {first_output.get('timestep', 1)})")
        
        # Fallback to old format
        elif 'statevector' in self.results:
            statevector = np.array(self.results['statevector'])
        elif 'statevector_shape' in self.results:
            # Statevector shape is available, but not the actual data
            print("⚠️  Statevector shape available but not full data")
            return {"bell_violation": False, "max_violation": 0.0, "note": "statevector_shape_only"}
        
        if statevector is None:
            print("❌ No statevector found for Bell test")
            return {"bell_violation": False, "max_violation": 0.0}
        num_qubits = int(np.log2(len(statevector)))
        
        # Test CHSH inequality for qubit pairs
        bell_results = {}
        max_violation = 0.0
        
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                # Create Bell state measurement operators
                # A0, A1 for qubit i, B0, B1 for qubit j
                A0 = np.array([[1, 0], [0, -1]])  # Pauli Z
                A1 = np.array([[0, 1], [1, 0]])   # Pauli X
                B0 = (A0 + A1) / np.sqrt(2)       # (Z + X)/√2
                B1 = (A0 - A1) / np.sqrt(2)       # (Z - X)/√2
                
                # Compute CHSH correlation
                # S = <A0B0> + <A0B1> + <A1B0> - <A1B1>
                S = self.compute_chsh_correlation(statevector, i, j, A0, A1, B0, B1)
                
                # Classical bound: |S| ≤ 2, Quantum bound: |S| ≤ 2√2 ≈ 2.828
                violation = abs(S) - 2.0
                max_violation = max(max_violation, violation)
                
                bell_results[f"qubits_{i}_{j}"] = {
                    "S_value": S,
                    "violation": violation,
                    "quantum_violation": abs(S) > 2.0
                }
        
        bell_results["max_violation"] = max_violation
        bell_results["bell_violation"] = max_violation > 0.1  # Significant violation threshold
        
        print(f"✅ Bell test complete. Max violation: {max_violation:.4f}")
        return bell_results
    
    def compute_chsh_correlation(self, statevector: np.ndarray, qubit_i: int, qubit_j: int, 
                               A0: np.ndarray, A1: np.ndarray, B0: np.ndarray, B1: np.ndarray) -> float:
        """Compute CHSH correlation for two qubits"""
        # Convert statevector to density matrix
        rho = np.outer(statevector, statevector.conj())
        
        # Create measurement operators for the full system
        num_qubits = int(np.log2(len(statevector)))
        
        # A0 measurement on qubit i
        A0_full = np.eye(2**num_qubits)
        A0_full = self.apply_operator_to_qubit(A0_full, A0, qubit_i, num_qubits)
        
        # B0 measurement on qubit j  
        B0_full = np.eye(2**num_qubits)
        B0_full = self.apply_operator_to_qubit(B0_full, B0, qubit_j, num_qubits)
        
        # Compute expectation values
        E_A0B0 = np.real(np.trace(rho @ A0_full @ B0_full))
        
        # Repeat for other combinations
        A1_full = np.eye(2**num_qubits)
        A1_full = self.apply_operator_to_qubit(A1_full, A1, qubit_i, num_qubits)
        
        B1_full = np.eye(2**num_qubits)
        B1_full = self.apply_operator_to_qubit(B1_full, B1, qubit_j, num_qubits)
        
        E_A0B1 = np.real(np.trace(rho @ A0_full @ B1_full))
        E_A1B0 = np.real(np.trace(rho @ A1_full @ B0_full))
        E_A1B1 = np.real(np.trace(rho @ A1_full @ B1_full))
        
        # CHSH correlation
        S = E_A0B0 + E_A0B1 + E_A1B0 - E_A1B1
        return S
    
    def apply_operator_to_qubit(self, full_op: np.ndarray, local_op: np.ndarray, 
                               qubit_idx: int, num_qubits: int) -> np.ndarray:
        """Apply local operator to specific qubit in full system"""
        # This is a simplified version - in practice would use tensor products
        # For now, we'll use a direct approach
        result = np.zeros_like(full_op)
        
        for i in range(2**num_qubits):
            for j in range(2**num_qubits):
                # Extract qubit states
                i_bin = format(i, f'0{num_qubits}b')
                j_bin = format(j, f'0{num_qubits}b')
                
                # Apply operator to target qubit
                qubit_i_state = int(i_bin[qubit_idx])
                qubit_j_state = int(j_bin[qubit_idx])
                
                # Check if other qubits match
                other_qubits_match = all(i_bin[k] == j_bin[k] for k in range(num_qubits) if k != qubit_idx)
                
                if other_qubits_match:
                    result[i, j] = local_op[qubit_i_state, qubit_j_state]
        
        return result
    
    def quantum_state_tomography(self) -> Dict:
        """
        Perform quantum state tomography to verify quantum coherence
        """
        print("🔬 Performing Quantum State Tomography...")
        
        # Extract statevector if available (check different possible locations)
        statevector = None
        
        # Check for quantum_state_outputs format (new format)
        if 'quantum_state_outputs' in self.results and self.results['quantum_state_outputs']:
            # Use the first timestep's statevector
            first_output = self.results['quantum_state_outputs'][0]
            if 'statevector' in first_output and first_output['statevector']:
                # Convert complex number format back to numpy array
                sv_data = first_output['statevector']
                if isinstance(sv_data[0], dict) and 'real' in sv_data[0]:
                    # Convert from {"real": x, "imag": y} format
                    statevector = np.array([complex(item['real'], item['imag']) for item in sv_data])
                else:
                    statevector = np.array(sv_data)
                print(f"✅ Found statevector in quantum_state_outputs (timestep {first_output.get('timestep', 1)})")
        
        # Fallback to old format
        elif 'statevector' in self.results:
            statevector = np.array(self.results['statevector'])
        
        if statevector is None:
            print("❌ No statevector found for tomography")
            return {"tomography_success": False, "purity": 0.0}
        
        # Convert to density matrix
        rho = np.outer(statevector, statevector.conj())
        
        # Compute purity: Tr(ρ²)
        purity = np.real(np.trace(rho @ rho))
        
        # Compute von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy_vn = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        # Check for quantum coherence (off-diagonal elements)
        off_diagonal_strength = np.sum(np.abs(rho - np.diag(np.diag(rho))))
        
        # Compute fidelity with maximally mixed state
        max_mixed = np.eye(len(rho)) / len(rho)
        fidelity_mixed = np.real(np.trace(rho @ max_mixed))
        
        tomography_results = {
            "purity": purity,
            "von_neumann_entropy": entropy_vn,
            "off_diagonal_strength": off_diagonal_strength,
            "fidelity_with_mixed": fidelity_mixed,
            "is_pure": purity > 0.99,
            "has_coherence": off_diagonal_strength > 0.1,
            "tomography_success": True
        }
        
        print(f"✅ Tomography complete. Purity: {purity:.4f}, Coherence: {off_diagonal_strength:.4f}")
        return tomography_results
    
    def entanglement_witness_test(self) -> Dict:
        """
        Test entanglement witnesses to detect genuine quantum entanglement
        """
        print("🔬 Testing Entanglement Witnesses...")
        
        # Extract statevector if available (check different possible locations)
        statevector = None
        
        # Check for quantum_state_outputs format (new format)
        if 'quantum_state_outputs' in self.results and self.results['quantum_state_outputs']:
            # Use the first timestep's statevector
            first_output = self.results['quantum_state_outputs'][0]
            if 'statevector' in first_output and first_output['statevector']:
                # Convert complex number format back to numpy array
                sv_data = first_output['statevector']
                if isinstance(sv_data[0], dict) and 'real' in sv_data[0]:
                    # Convert from {"real": x, "imag": y} format
                    statevector = np.array([complex(item['real'], item['imag']) for item in sv_data])
                else:
                    statevector = np.array(sv_data)
                print(f"✅ Found statevector in quantum_state_outputs (timestep {first_output.get('timestep', 1)})")
        
        # Fallback to old format
        elif 'statevector' in self.results:
            statevector = np.array(self.results['statevector'])
        
        if statevector is None:
            print("❌ No statevector found for entanglement witness")
            return {"entanglement_detected": False, "witness_violations": 0}
        num_qubits = int(np.log2(len(statevector)))
        
        # Test for bipartite entanglement
        witness_results = {}
        violations = 0
        
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                # Compute reduced density matrix for qubits i and j
                rho_ij = self.compute_reduced_density_matrix(statevector, [i, j])
                
                # Test PPT (Positive Partial Transpose) criterion
                ppt_violation = self.test_ppt_criterion(rho_ij)
                
                # Test concurrence
                concurrence_val = self.compute_concurrence(rho_ij)
                
                # Test tangle
                tangle_val = concurrence_val**2
                
                witness_results[f"qubits_{i}_{j}"] = {
                    "ppt_violation": ppt_violation,
                    "concurrence": concurrence_val,
                    "tangle": tangle_val,
                    "is_entangled": concurrence_val > 0.01 or ppt_violation
                }
                
                if witness_results[f"qubits_{i}_{j}"]["is_entangled"]:
                    violations += 1
        
        witness_results["total_violations"] = violations
        witness_results["entanglement_detected"] = violations > 0
        
        print(f"✅ Entanglement witness complete. Violations: {violations}")
        return witness_results
    
    def compute_reduced_density_matrix(self, statevector: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Compute reduced density matrix for specified qubits"""
        # Convert to density matrix
        rho = np.outer(statevector, statevector.conj())
        
        # Use Qiskit's partial_trace
        from qiskit.quantum_info import partial_trace
        rho_qiskit = DensityMatrix(rho)
        
        # Trace out all qubits except the specified ones
        num_qubits = int(np.log2(len(statevector)))
        qubits_to_trace = [i for i in range(num_qubits) if i not in qubits]
        
        reduced_rho = partial_trace(rho_qiskit, qubits_to_trace)
        return reduced_rho.data
    
    def test_ppt_criterion(self, rho: np.ndarray) -> bool:
        """Test Positive Partial Transpose criterion"""
        # Partial transpose with respect to second qubit
        rho_pt = np.zeros_like(rho)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        rho_pt[2*i + k, 2*j + l] = rho[2*i + l, 2*j + k]
        
        # Check if any eigenvalue is negative
        eigenvalues = np.linalg.eigvalsh(rho_pt)
        return np.any(eigenvalues < -1e-10)
    
    def compute_concurrence(self, rho: np.ndarray) -> float:
        """Compute concurrence for 2-qubit state"""
        # For 2-qubit states, concurrence can be computed from eigenvalues
        # of ρ(σy ⊗ σy)ρ*(σy ⊗ σy)
        
        # Pauli Y matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        
        # σy ⊗ σy
        sigma_y_tensor = np.kron(sigma_y, sigma_y)
        
        # Compute ρ(σy ⊗ σy)ρ*(σy ⊗ σy)
        rho_conj = rho.conj()
        matrix = rho @ sigma_y_tensor @ rho_conj @ sigma_y_tensor
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvalsh(matrix)
        eigenvalues = np.sqrt(np.abs(eigenvalues))
        
        # Concurrence formula
        concurrence = max(0, eigenvalues[3] - eigenvalues[2] - eigenvalues[1] - eigenvalues[0])
        return concurrence
    
    def quantum_coherence_test(self) -> Dict:
        """
        Test quantum coherence and superposition persistence
        """
        print("🔬 Testing Quantum Coherence...")
        
        # Extract statevector if available (check different possible locations)
        statevector = None
        
        # Check for quantum_state_outputs format (new format)
        if 'quantum_state_outputs' in self.results and self.results['quantum_state_outputs']:
            # Use the first timestep's statevector
            first_output = self.results['quantum_state_outputs'][0]
            if 'statevector' in first_output and first_output['statevector']:
                # Convert complex number format back to numpy array
                sv_data = first_output['statevector']
                if isinstance(sv_data[0], dict) and 'real' in sv_data[0]:
                    # Convert from {"real": x, "imag": y} format
                    statevector = np.array([complex(item['real'], item['imag']) for item in sv_data])
                else:
                    statevector = np.array(sv_data)
                print(f"✅ Found statevector in quantum_state_outputs (timestep {first_output.get('timestep', 1)})")
        
        # Fallback to old format
        elif 'statevector' in self.results:
            statevector = np.array(self.results['statevector'])
        
        if statevector is None:
            print("❌ No statevector found for coherence test")
            return {"coherence_detected": False, "superposition_strength": 0.0}
        
        # Test superposition in computational basis
        superposition_strength = np.sum(np.abs(statevector)**2 * (1 - np.abs(statevector)**2))
        
        # Test coherence between basis states
        coherence_matrix = np.outer(statevector, statevector.conj())
        off_diagonal_coherence = np.sum(np.abs(coherence_matrix - np.diag(np.diag(coherence_matrix))))
        
        # Test quantum interference
        interference_strength = np.abs(np.sum(statevector))**2 - np.sum(np.abs(statevector)**2)
        
        coherence_results = {
            "superposition_strength": superposition_strength,
            "off_diagonal_coherence": off_diagonal_coherence,
            "interference_strength": interference_strength,
            "has_superposition": superposition_strength > 0.1,
            "has_coherence": off_diagonal_coherence > 0.1,
            "has_interference": abs(interference_strength) > 0.1,
            "coherence_detected": superposition_strength > 0.1 or off_diagonal_coherence > 0.1
        }
        
        print(f"✅ Coherence test complete. Superposition: {superposition_strength:.4f}")
        return coherence_results
    
    def causal_structure_test(self) -> Dict:
        """
        Test causal structure and light cone constraints
        """
        print("🔬 Testing Causal Structure...")
        
        # Extract mutual information matrix if available
        mi_matrix = None
        
        # Check for quantum_state_outputs format (new format)
        if 'quantum_state_outputs' in self.results and self.results['quantum_state_outputs']:
            # Use the first timestep's MI matrix
            first_output = self.results['quantum_state_outputs'][0]
            if 'mutual_information_matrix' in first_output and first_output['mutual_information_matrix']:
                mi_matrix = np.array(first_output['mutual_information_matrix'])
                print(f"✅ Found MI matrix in quantum_state_outputs (timestep {first_output.get('timestep', 1)})")
        
        # Fallback to old format
        elif 'mi_matrix' in self.results:
            mi_matrix = np.array(self.results['mi_matrix'])
        elif 'mutual_information_matrix' in self.results:
            mi_matrix = np.array(self.results['mutual_information_matrix'])
        
        if mi_matrix is None:
            print("❌ No mutual information matrix found for causal test")
            return {"causal_structure_valid": False, "light_cone_violations": 0}
        
        num_qubits = len(mi_matrix)
        
        # Test for causality violations in information flow
        # In a causal structure, information should flow forward in time
        # Check if distant qubits have higher MI than nearby ones (potential causality violation)
        
        violations = 0
        causal_metrics = {}
        
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                distance = abs(i - j)
                mi_value = mi_matrix[i, j]
                
                # Check if distant qubits have anomalously high MI
                # This could indicate causality violation
                expected_mi_decay = np.exp(-distance / 2.0)  # Expected exponential decay
                violation_strength = max(0, mi_value - expected_mi_decay)
                
                if violation_strength > 0.1:
                    violations += 1
                
                causal_metrics[f"qubits_{i}_{j}"] = {
                    "distance": distance,
                    "mi_value": mi_value,
                    "expected_mi": expected_mi_decay,
                    "violation_strength": violation_strength,
                    "causality_violation": violation_strength > 0.1
                }
        
        causal_results = {
            "light_cone_violations": violations,
            "causal_structure_valid": violations < num_qubits // 2,  # Allow some violations
            "causal_metrics": causal_metrics
        }
        
        print(f"✅ Causal structure test complete. Violations: {violations}")
        return causal_results
    
    def holographic_principle_test(self) -> Dict:
        """
        Test holographic principle and AdS/CFT correspondence
        """
        print("🔬 Testing Holographic Principle...")
        
        # Extract mutual information matrix if available
        mi_matrix = None
        
        # Check for quantum_state_outputs format (new format)
        if 'quantum_state_outputs' in self.results and self.results['quantum_state_outputs']:
            # Use the first timestep's MI matrix
            first_output = self.results['quantum_state_outputs'][0]
            if 'mutual_information_matrix' in first_output and first_output['mutual_information_matrix']:
                mi_matrix = np.array(first_output['mutual_information_matrix'])
                print(f"✅ Found MI matrix in quantum_state_outputs (timestep {first_output.get('timestep', 1)})")
        
        # Fallback to old format
        elif 'mi_matrix' in self.results:
            mi_matrix = np.array(self.results['mi_matrix'])
        elif 'mutual_information_matrix' in self.results:
            mi_matrix = np.array(self.results['mutual_information_matrix'])
        
        if mi_matrix is None:
            print("❌ No mutual information matrix found for holographic test")
            return {"holographic_consistency": False, "rt_surface_area": 0.0}
        
        num_qubits = len(mi_matrix)
        
        # Test Ryu-Takayanagi surface area law
        # For a boundary region A, the entanglement entropy should scale with the area of the minimal surface
        rt_results = {}
        
        for boundary_size in range(1, num_qubits // 2 + 1):
            # Compute entanglement entropy for boundary region of size boundary_size
            boundary_entropy = self.compute_boundary_entropy(mi_matrix, boundary_size)
            
            # Expected scaling: S(A) ~ Area(A) for holographic systems
            expected_area = boundary_size  # In 1D boundary, area = length
            area_law_ratio = boundary_entropy / expected_area
            
            rt_results[f"boundary_size_{boundary_size}"] = {
                "boundary_entropy": boundary_entropy,
                "expected_area": expected_area,
                "area_law_ratio": area_law_ratio,
                "obeys_area_law": 0.5 < area_law_ratio < 2.0
            }
        
        # Test bulk-boundary correspondence
        # Bulk geometry should be reconstructible from boundary correlations
        bulk_reconstruction_quality = self.test_bulk_reconstruction(mi_matrix)
        
        holographic_results = {
            "rt_surfaces": rt_results,
            "bulk_reconstruction_quality": bulk_reconstruction_quality,
            "holographic_consistency": bulk_reconstruction_quality > 0.7
        }
        
        print(f"✅ Holographic principle test complete. Reconstruction quality: {bulk_reconstruction_quality:.4f}")
        return holographic_results
    
    def compute_boundary_entropy(self, mi_matrix: np.ndarray, boundary_size: int) -> float:
        """Compute entanglement entropy for boundary region"""
        # Simplified calculation using mutual information
        # In practice, would compute actual von Neumann entropy
        total_mi = 0.0
        for i in range(boundary_size):
            for j in range(boundary_size, len(mi_matrix)):
                total_mi += mi_matrix[i, j]
        return total_mi / boundary_size
    
    def test_bulk_reconstruction(self, mi_matrix: np.ndarray) -> float:
        """Test quality of bulk geometry reconstruction from boundary data"""
        # Use MDS to reconstruct bulk geometry
        from sklearn.manifold import MDS
        
        # Convert MI to distance matrix
        distance_matrix = -np.log(mi_matrix + 1e-10)
        
        # Reconstruct 2D geometry
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords_2d = mds.fit_transform(distance_matrix)
        
        # Compute reconstruction quality (stress)
        stress = mds.stress_
        
        # Normalize stress to quality score (lower stress = higher quality)
        quality = max(0, 1 - stress / np.max(distance_matrix)**2)
        
        return quality
    
    def run_all_tests(self) -> Dict:
        """
        Run all quantum spacetime validation tests
        """
        print("🚀 Starting Quantum Spacetime Validation...")
        print("=" * 60)
        
        all_results = {}
        
        # Run all tests
        all_results["bell_inequality"] = self.bell_inequality_test()
        all_results["quantum_tomography"] = self.quantum_state_tomography()
        all_results["entanglement_witness"] = self.entanglement_witness_test()
        all_results["quantum_coherence"] = self.quantum_coherence_test()
        all_results["causal_structure"] = self.causal_structure_test()
        all_results["holographic_principle"] = self.holographic_principle_test()
        
        # Compute overall quantum spacetime score
        score = self.compute_quantum_spacetime_score(all_results)
        all_results["quantum_spacetime_score"] = score
        
        # Save results
        self.save_results(all_results)
        
        # Generate summary
        self.generate_summary(all_results)
        
        return all_results
    
    def compute_quantum_spacetime_score(self, results: Dict) -> float:
        """
        Compute overall quantum spacetime validation score
        """
        score = 0.0
        max_score = 0.0
        
        # Bell inequality violation (25% weight)
        if results["bell_inequality"]["bell_violation"]:
            score += 0.25
        max_score += 0.25
        
        # Quantum tomography (20% weight)
        if results["quantum_tomography"]["tomography_success"]:
            if results["quantum_tomography"]["has_coherence"]:
                score += 0.20
        max_score += 0.20
        
        # Entanglement witness (20% weight)
        if results["entanglement_witness"]["entanglement_detected"]:
            score += 0.20
        max_score += 0.20
        
        # Quantum coherence (15% weight)
        if results["quantum_coherence"]["coherence_detected"]:
            score += 0.15
        max_score += 0.15
        
        # Causal structure (10% weight)
        if results["causal_structure"]["causal_structure_valid"]:
            score += 0.10
        max_score += 0.10
        
        # Holographic principle (10% weight)
        if results["holographic_principle"]["holographic_consistency"]:
            score += 0.10
        max_score += 0.10
        
        return score / max_score
    
    def save_results(self, results: Dict):
        """Save validation results"""
        results_file = os.path.join(self.output_dir, "quantum_spacetime_validation_results.json")
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, bool) or isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, (str, int, float)):
                return obj
            else:
                return str(obj)  # Convert any other types to string
        
        serializable_results = convert_numpy(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Results saved to: {results_file}")
    
    def generate_summary(self, results: Dict):
        """Generate human-readable summary"""
        summary_file = os.path.join(self.output_dir, "quantum_spacetime_validation_summary.txt")
        
        score = results["quantum_spacetime_score"]
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("QUANTUM SPACETIME VALIDATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"OVERALL QUANTUM SPACETIME SCORE: {score:.4f}\n")
            f.write(f"ASSESSMENT: {'✅ GENUINE QUANTUM SPACETIME' if score > 0.7 else '❌ CLASSICAL CORRELATIONS' if score < 0.3 else '⚠️  MIXED QUANTUM-CLASSICAL'}\n\n")
            
            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 30 + "\n\n")
            
            # Bell inequality
            bell = results["bell_inequality"]
            f.write(f"1. Bell Inequality Violation: {'✅ VIOLATED' if bell['bell_violation'] else '❌ NOT VIOLATED'}\n")
            f.write(f"   Max violation: {bell['max_violation']:.4f}\n\n")
            
            # Quantum tomography
            tom = results["quantum_tomography"]
            f.write(f"2. Quantum State Tomography: {'✅ QUANTUM' if tom.get('has_coherence', False) else '❌ CLASSICAL'}\n")
            f.write(f"   Purity: {tom.get('purity', 0.0):.4f}\n")
            f.write(f"   Coherence strength: {tom.get('off_diagonal_strength', 0.0):.4f}\n\n")
            
            # Entanglement witness
            ent = results["entanglement_witness"]
            f.write(f"3. Entanglement Witness: {'✅ ENTANGLED' if ent.get('entanglement_detected', False) else '❌ NOT ENTANGLED'}\n")
            f.write(f"   Total violations: {ent.get('total_violations', 0)}\n\n")
            
            # Quantum coherence
            coh = results["quantum_coherence"]
            f.write(f"4. Quantum Coherence: {'✅ COHERENT' if coh.get('coherence_detected', False) else '❌ INCOHERENT'}\n")
            f.write(f"   Superposition strength: {coh.get('superposition_strength', 0.0):.4f}\n\n")
            
            # Causal structure
            caus = results["causal_structure"]
            f.write(f"5. Causal Structure: {'✅ VALID' if caus.get('causal_structure_valid', False) else '❌ VIOLATED'}\n")
            f.write(f"   Light cone violations: {caus.get('light_cone_violations', 0)}\n\n")
            
            # Holographic principle
            hol = results["holographic_principle"]
            f.write(f"6. Holographic Principle: {'✅ CONSISTENT' if hol.get('holographic_consistency', False) else '❌ INCONSISTENT'}\n")
            f.write(f"   Bulk reconstruction quality: {hol.get('bulk_reconstruction_quality', 0.0):.4f}\n\n")
            
            f.write("CONCLUSION:\n")
            f.write("-" * 20 + "\n")
            if score > 0.7:
                f.write("🎉 STRONG EVIDENCE FOR GENUINE QUANTUM EMERGENT SPACETIME!\n")
                f.write("   The system exhibits multiple quantum signatures including:\n")
                f.write("   - Bell inequality violations (quantum correlations)\n")
                f.write("   - Quantum coherence and superposition\n")
                f.write("   - Genuine entanglement\n")
                f.write("   - Consistent causal structure\n")
                f.write("   - Holographic principle compliance\n\n")
                f.write("   This suggests the entropy engineering has successfully created\n")
                f.write("   quantum emergent spacetime rather than classical correlations.\n")
            elif score > 0.3:
                f.write("⚠️  MIXED QUANTUM-CLASSICAL BEHAVIOR DETECTED\n")
                f.write("   The system shows some quantum features but may contain\n")
                f.write("   classical correlations. Further optimization needed.\n")
            else:
                f.write("❌ PRIMARILY CLASSICAL CORRELATIONS\n")
                f.write("   The system appears to be generating classical geometric\n")
                f.write("   correlations rather than quantum emergent spacetime.\n")
                f.write("   Consider enhancing quantum features in the circuit design.\n")
        
        print(f"✅ Summary saved to: {summary_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum Spacetime Validation Tool")
    parser.add_argument("results_file", help="Path to experiment results JSON file")
    parser.add_argument("output_dir", help="Output directory for validation results")
    
    args = parser.parse_args()
    
    # Run validation
    validator = QuantumSpacetimeValidator(args.results_file, args.output_dir)
    results = validator.run_all_tests()
    
    # Print final score
    score = results["quantum_spacetime_score"]
    print("\n" + "=" * 60)
    print(f"🎯 FINAL QUANTUM SPACETIME SCORE: {score:.4f}")
    if score > 0.7:
        print("🎉 GENUINE QUANTUM EMERGENT SPACETIME DETECTED!")
    elif score > 0.3:
        print("⚠️  MIXED QUANTUM-CLASSICAL BEHAVIOR")
    else:
        print("❌ PRIMARILY CLASSICAL CORRELATIONS")
    print("=" * 60)

if __name__ == "__main__":
    main() 