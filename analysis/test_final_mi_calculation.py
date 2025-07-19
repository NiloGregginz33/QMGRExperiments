#!/usr/bin/env python3
"""
Final test for mutual information calculation with correct bitstring parsing
"""

import numpy as np

def calculate_mi_from_counts(counts, n, i, j):
    """Calculate mutual information between qubits i and j from counts"""
    total_shots = sum(counts.values())
    
    # Extract marginal probabilities for qubits i and j
    p_i_0 = 0.0  # P(qubit i = 0)
    p_i_1 = 0.0  # P(qubit i = 1)
    p_j_0 = 0.0  # P(qubit j = 0)
    p_j_1 = 0.0  # P(qubit j = 1)
    p_ij_00 = 0.0  # P(qubit i = 0, qubit j = 0)
    p_ij_01 = 0.0  # P(qubit i = 0, qubit j = 1)
    p_ij_10 = 0.0  # P(qubit i = 1, qubit j = 0)
    p_ij_11 = 0.0  # P(qubit i = 1, qubit j = 1)
    
    for bitstring, count in counts.items():
        if len(bitstring) >= n:
            # Extract bits for qubits i and j (reverse order for Qiskit)
            bit_i = int(bitstring[-(i+1)])
            bit_j = int(bitstring[-(j+1)])
            
            # Update marginal probabilities (FIXED: now updating both i and j)
            if bit_i == 0:
                p_i_0 += count
                if bit_j == 0:
                    p_ij_00 += count
                    p_j_0 += count
                else:
                    p_ij_01 += count
                    p_j_1 += count
            else:
                p_i_1 += count
                if bit_j == 0:
                    p_ij_10 += count
                    p_j_0 += count
                else:
                    p_ij_11 += count
                    p_j_1 += count
    
    # Normalize probabilities
    p_i_0 /= total_shots
    p_i_1 /= total_shots
    p_j_0 /= total_shots
    p_j_1 /= total_shots
    p_ij_00 /= total_shots
    p_ij_01 /= total_shots
    p_ij_10 /= total_shots
    p_ij_11 /= total_shots
    
    # Calculate mutual information: I(X;Y) = sum p(x,y) * log(p(x,y)/(p(x)*p(y)))
    mi_value = 0.0
    if p_ij_00 > 0 and p_i_0 > 0 and p_j_0 > 0:
        mi_value += p_ij_00 * np.log(p_ij_00 / (p_i_0 * p_j_0))
    if p_ij_01 > 0 and p_i_0 > 0 and p_j_1 > 0:
        mi_value += p_ij_01 * np.log(p_ij_01 / (p_i_0 * p_j_1))
    if p_ij_10 > 0 and p_i_1 > 0 and p_j_0 > 0:
        mi_value += p_ij_10 * np.log(p_ij_10 / (p_i_1 * p_j_0))
    if p_ij_11 > 0 and p_i_1 > 0 and p_j_1 > 0:
        mi_value += p_ij_11 * np.log(p_ij_11 / (p_i_1 * p_j_1))
    
    return mi_value, {
        'p_i_0': p_i_0, 'p_i_1': p_i_1,
        'p_j_0': p_j_0, 'p_j_1': p_j_1,
        'p_ij_00': p_ij_00, 'p_ij_01': p_ij_01,
        'p_ij_10': p_ij_10, 'p_ij_11': p_ij_11
    }

def test_with_real_hardware_counts():
    """Test with real hardware counts from the experiment"""
    print("=== Testing with Real Hardware Counts (FIXED) ===")
    
    # Extract real counts from the experiment (first timestep)
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
        "1001100": 181,
        "1011101": 85,
        "0000000": 240,
        "0111110": 170,
        "1001111": 115,
        "1001110": 233,
        "0011001": 72,
        "0011110": 173,
        "1101111": 66,
        "0001011": 116,
        "0010000": 174,
        "0011100": 164,
        "0011010": 288,
        "0011101": 84,
        "1110000": 197,
        "1111111": 135,
        "0101000": 182,
        "0111111": 83,
        "0001001": 97,
        "0010011": 149,
        "0111000": 146,
        "0001000": 184,
        "1011000": 218,
        "1011011": 88,
        "0110000": 134,
        "1001001": 91,
        "1111100": 180,
        "1101011": 81,
        "1101100": 208,
        "1111101": 81,
        "1100110": 200,
        "1000000": 286,
        "0110100": 157,
        "0000111": 134,
        "0000001": 130,
        "0011000": 115,
        "1001011": 154,
        "0101100": 154,
        "1111001": 75,
        "1010000": 326,
        "0011011": 150,
        "1111000": 183,
        "0110001": 81,
        "0101010": 137,
        "0111101": 82,
        "1010001": 151,
        "1111110": 262,
        "0111010": 215,
        "0000110": 322,
        "0001111": 105,
        "0101001": 89,
        "0111001": 66,
        "1000011": 161,
        "1101000": 130,
        "0111100": 172,
        "1011110": 329,
        "0100011": 101,
        "1001101": 96,
        "0000101": 163,
        "0000010": 259,
        "0101101": 80,
        "0010100": 227,
        "1001000": 183,
        "0010010": 346,
        "0111011": 102,
        "0001110": 226,
        "1111010": 114,
        "0001010": 260,
        "1010100": 239,
        "0100111": 80,
        "0101110": 159,
        "1011111": 123,
        "0100100": 167,
        "1101110": 142,
        "1110110": 324,
        "1101101": 99,
        "0011111": 84,
        "1111011": 48,
        "1100010": 221,
        "0010111": 105,
        "1000110": 223,
        "1110101": 61,
        "1110010": 127,
        "0010001": 87,
        "1100001": 62,
        "1011100": 172,
        "1100100": 177,
        "1110111": 143,
        "1010010": 199,
        "0100000": 199,
        "1000010": 332,
        "0100101": 63,
        "1000101": 137,
        "0000011": 137,
        "0110111": 111,
        "1010110": 320,
        "1100101": 96,
        "1100000": 127,
        "1101010": 164,
        "1000001": 147,
        "0010101": 111,
        "0100010": 176,
        "0101111": 69,
        "0110101": 73,
        "1010101": 90,
        "0110010": 273,
        "1000100": 266,
        "1000111": 104,
        "1100011": 101,
        "0010110": 182,
        "0110110": 224,
        "0110011": 148,
        "1100111": 90,
        "0100001": 85,
        "1010111": 159,
        "0100110": 207,
        "1010011": 109,
        "1110001": 77
    }
    
    n = 7
    total_shots = sum(real_counts.values())
    
    print(f"Total shots: {total_shots}")
    print(f"Number of unique bitstrings: {len(real_counts)}")
    
    # Test MI calculation for several qubit pairs
    test_pairs = [(0, 1), (0, 2), (1, 2), (0, 6), (3, 4)]
    
    for i, j in test_pairs:
        print(f"\n--- MI between qubits {i} and {j} ---")
        
        mi_value, probs = calculate_mi_from_counts(real_counts, n, i, j)
        
        print(f"Marginal probabilities:")
        print(f"  P(qubit {i} = 0): {probs['p_i_0']:.4f}")
        print(f"  P(qubit {i} = 1): {probs['p_i_1']:.4f}")
        print(f"  P(qubit {j} = 0): {probs['p_j_0']:.4f}")
        print(f"  P(qubit {j} = 1): {probs['p_j_1']:.4f}")
        
        print(f"Joint probabilities:")
        print(f"  P(qubit {i} = 0, qubit {j} = 0): {probs['p_ij_00']:.4f}")
        print(f"  P(qubit {i} = 0, qubit {j} = 1): {probs['p_ij_01']:.4f}")
        print(f"  P(qubit {i} = 1, qubit {j} = 0): {probs['p_ij_10']:.4f}")
        print(f"  P(qubit {i} = 1, qubit {j} = 1): {probs['p_ij_11']:.4f}")
        
        print(f"Mutual Information: {mi_value:.6f}")
        
        # Check if probabilities sum to 1
        marginal_sum_i = probs['p_i_0'] + probs['p_i_1']
        marginal_sum_j = probs['p_j_0'] + probs['p_j_1']
        joint_sum = probs['p_ij_00'] + probs['p_ij_01'] + probs['p_ij_10'] + probs['p_ij_11']
        print(f"Marginal sum (i): {marginal_sum_i:.6f} (should be ~1.0)")
        print(f"Marginal sum (j): {marginal_sum_j:.6f} (should be ~1.0)")
        print(f"Joint sum: {joint_sum:.6f} (should be ~1.0)")

def test_bitstring_parsing():
    """Test bitstring parsing with a few examples"""
    print("\n=== Testing Bitstring Parsing ===")
    
    test_bitstrings = ["1110100", "0001101", "0001100"]
    n = 7
    
    for bitstring in test_bitstrings:
        print(f"\nBitstring: {bitstring}")
        for i in range(min(3, n)):
            bit = int(bitstring[-(i+1)])
            print(f"  Qubit {i}: {bit}")

if __name__ == "__main__":
    test_bitstring_parsing()
    test_with_real_hardware_counts() 