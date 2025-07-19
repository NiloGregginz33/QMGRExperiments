#!/usr/bin/env python3
"""
Test script to debug hardware mutual information calculation
"""

import numpy as np

def test_bitstring_parsing():
    """Test the bitstring parsing logic"""
    print("=== Testing Bitstring Parsing (FIXED) ===")
    
    # Simulate hardware counts (7 qubits)
    n = 7
    total_shots = 20000
    
    # Create some test counts with known patterns
    test_counts = {
        "0000000": 1000,  # All zeros
        "1111111": 1000,  # All ones
        "1010101": 800,   # Alternating
        "0101010": 800,   # Alternating
        "1100110": 600,   # Some pattern
        "0011001": 600,   # Some pattern
    }
    
    print("Test counts:")
    for bitstring, count in test_counts.items():
        print(f"  {bitstring}: {count}")
    
    # Test mutual information calculation for qubits 0 and 1
    i, j = 0, 1
    print(f"\nCalculating MI between qubits {i} and {j}:")
    
    # Extract marginal probabilities for qubits i and j
    p_i_0 = 0.0  # P(qubit i = 0)
    p_i_1 = 0.0  # P(qubit i = 1)
    p_j_0 = 0.0  # P(qubit j = 0)
    p_j_1 = 0.0  # P(qubit j = 1)
    p_ij_00 = 0.0  # P(qubit i = 0, qubit j = 0)
    p_ij_01 = 0.0  # P(qubit i = 0, qubit j = 1)
    p_ij_10 = 0.0  # P(qubit i = 1, qubit j = 0)
    p_ij_11 = 0.0  # P(qubit i = 1, qubit j = 1)
    
    for bitstring, count in test_counts.items():
        if len(bitstring) >= n:
            # Extract bits for qubits i and j (FIXED: Qiskit bitstrings are in little-endian order)
            bit_i = int(bitstring[i])
            bit_j = int(bitstring[j])
            
            print(f"  Bitstring: {bitstring}, bit_{i}={bit_i}, bit_{j}={bit_j}")
            
            # Update marginal probabilities
            if bit_i == 0:
                p_i_0 += count
                if bit_j == 0:
                    p_ij_00 += count
                else:
                    p_ij_01 += count
            else:
                p_i_1 += count
                if bit_j == 0:
                    p_ij_10 += count
                else:
                    p_ij_11 += count
    
    # Normalize probabilities
    p_i_0 /= total_shots
    p_i_1 /= total_shots
    p_j_0 /= total_shots
    p_j_1 /= total_shots
    p_ij_00 /= total_shots
    p_ij_01 /= total_shots
    p_ij_10 /= total_shots
    p_ij_11 /= total_shots
    
    print(f"\nMarginal probabilities:")
    print(f"  P(qubit {i} = 0): {p_i_0:.4f}")
    print(f"  P(qubit {i} = 1): {p_i_1:.4f}")
    print(f"  P(qubit {j} = 0): {p_j_0:.4f}")
    print(f"  P(qubit {j} = 1): {p_j_1:.4f}")
    
    print(f"\nJoint probabilities:")
    print(f"  P(qubit {i} = 0, qubit {j} = 0): {p_ij_00:.4f}")
    print(f"  P(qubit {i} = 0, qubit {j} = 1): {p_ij_01:.4f}")
    print(f"  P(qubit {i} = 1, qubit {j} = 0): {p_ij_10:.4f}")
    print(f"  P(qubit {i} = 1, qubit {j} = 1): {p_ij_11:.4f}")
    
    # Calculate mutual information: I(X;Y) = sum p(x,y) * log(p(x,y)/(p(x)*p(y)))
    mi_value = 0.0
    if p_ij_00 > 0 and p_i_0 > 0 and p_j_0 > 0:
        term = p_ij_00 * np.log(p_ij_00 / (p_i_0 * p_j_0))
        mi_value += term
        print(f"  Term 00: {term:.6f}")
    if p_ij_01 > 0 and p_i_0 > 0 and p_j_1 > 0:
        term = p_ij_01 * np.log(p_ij_01 / (p_i_0 * p_j_1))
        mi_value += term
        print(f"  Term 01: {term:.6f}")
    if p_ij_10 > 0 and p_i_1 > 0 and p_j_0 > 0:
        term = p_ij_10 * np.log(p_ij_10 / (p_i_1 * p_j_0))
        mi_value += term
        print(f"  Term 10: {term:.6f}")
    if p_ij_11 > 0 and p_i_1 > 0 and p_j_1 > 0:
        term = p_ij_11 * np.log(p_ij_11 / (p_i_1 * p_j_1))
        mi_value += term
        print(f"  Term 11: {term:.6f}")
    
    print(f"\nMutual Information: {mi_value:.6f}")
    return mi_value

def test_real_hardware_counts():
    """Test with real hardware counts from the experiment"""
    print("\n=== Testing Real Hardware Counts (FIXED) ===")
    
    # Extract some real counts from the experiment
    real_counts = {
        "1110100": 157,
        "0001101": 121,
        "0001100": 221,
        "0101011": 72,
        "1001010": 311,
        "1011001": 109,
        "1011010": 168,
        "1110011": 71,
        "1101001": 84,
        "0000100": 329,
    }
    
    n = 7
    total_shots = sum(real_counts.values())
    
    print(f"Total shots: {total_shots}")
    print("Sample counts:")
    for bitstring, count in list(real_counts.items())[:5]:
        print(f"  {bitstring}: {count}")
    
    # Test MI calculation for qubits 0 and 1
    i, j = 0, 1
    print(f"\nCalculating MI between qubits {i} and {j}:")
    
    # Extract marginal probabilities for qubits i and j
    p_i_0 = 0.0
    p_i_1 = 0.0
    p_j_0 = 0.0
    p_j_1 = 0.0
    p_ij_00 = 0.0
    p_ij_01 = 0.0
    p_ij_10 = 0.0
    p_ij_11 = 0.0
    
    for bitstring, count in real_counts.items():
        if len(bitstring) >= n:
            # Extract bits for qubits i and j (FIXED: Qiskit bitstrings are in little-endian order)
            bit_i = int(bitstring[i])
            bit_j = int(bitstring[j])
            
            # Update marginal probabilities
            if bit_i == 0:
                p_i_0 += count
                if bit_j == 0:
                    p_ij_00 += count
                else:
                    p_ij_01 += count
            else:
                p_i_1 += count
                if bit_j == 0:
                    p_ij_10 += count
                else:
                    p_ij_11 += count
    
    # Normalize probabilities
    p_i_0 /= total_shots
    p_i_1 /= total_shots
    p_j_0 /= total_shots
    p_j_1 /= total_shots
    p_ij_00 /= total_shots
    p_ij_01 /= total_shots
    p_ij_10 /= total_shots
    p_ij_11 /= total_shots
    
    print(f"Marginal probabilities:")
    print(f"  P(qubit {i} = 0): {p_i_0:.4f}")
    print(f"  P(qubit {i} = 1): {p_i_1:.4f}")
    print(f"  P(qubit {j} = 0): {p_j_0:.4f}")
    print(f"  P(qubit {j} = 1): {p_j_1:.4f}")
    
    print(f"Joint probabilities:")
    print(f"  P(qubit {i} = 0, qubit {j} = 0): {p_ij_00:.4f}")
    print(f"  P(qubit {i} = 0, qubit {j} = 1): {p_ij_01:.4f}")
    print(f"  P(qubit {i} = 1, qubit {j} = 0): {p_ij_10:.4f}")
    print(f"  P(qubit {i} = 1, qubit {j} = 1): {p_ij_11:.4f}")
    
    # Calculate mutual information
    mi_value = 0.0
    if p_ij_00 > 0 and p_i_0 > 0 and p_j_0 > 0:
        term = p_ij_00 * np.log(p_ij_00 / (p_i_0 * p_j_0))
        mi_value += term
    if p_ij_01 > 0 and p_i_0 > 0 and p_j_1 > 0:
        term = p_ij_01 * np.log(p_ij_01 / (p_i_0 * p_j_1))
        mi_value += term
    if p_ij_10 > 0 and p_i_1 > 0 and p_j_0 > 0:
        term = p_ij_10 * np.log(p_ij_10 / (p_i_1 * p_j_0))
        mi_value += term
    if p_ij_11 > 0 and p_i_1 > 0 and p_j_1 > 0:
        term = p_ij_11 * np.log(p_ij_11 / (p_i_1 * p_j_1))
        mi_value += term
    
    print(f"Mutual Information: {mi_value:.6f}")
    return mi_value

if __name__ == "__main__":
    test_bitstring_parsing()
    test_real_hardware_counts() 