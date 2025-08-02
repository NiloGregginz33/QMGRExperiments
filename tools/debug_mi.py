import numpy as np

# Test data from the experiment
counts = {'101': 17, '001': 10, '010': 15, '110': 12, '100': 11, '011': 15, '111': 12, '000': 8}
total = sum(counts.values())
print(f'Total shots: {total}')

# Convert to probabilities
probs = {k: v/total for k, v in counts.items()}
print(f'Probabilities: {probs}')

def get_bit(bitstring, k):
    """Extract bit k from bitstring with little-endian"""
    return int(bitstring[-(k+1)])

def entropy_from_probs(probs_dict):
    """Calculate von Neumann entropy from probability distribution"""
    entropy = 0.0
    for p in probs_dict.values():
        if p > 0 and p <= 1:
            entropy -= p * np.log2(p)
    return entropy

def get_subsystem_probs(qubits):
    """Get probability distribution for a subsystem of qubits"""
    subsystem_probs = {}
    for bitstring, p in probs.items():
        # Extract bits for the specified qubits
        subsystem_bits = ""
        for qubit in qubits:
            bit_val = get_bit(bitstring, qubit)
            subsystem_bits += str(bit_val)
        
        # Aggregate probabilities
        if subsystem_bits in subsystem_probs:
            subsystem_probs[subsystem_bits] += p
        else:
            subsystem_probs[subsystem_bits] = p
    
    return subsystem_probs

# Test individual qubit entropies
print("\nTesting individual qubit entropies:")
for i in range(3):
    probs_i = get_subsystem_probs([i])
    S_i = entropy_from_probs(probs_i)
    print(f"Qubit {i}: {probs_i} -> S({i}) = {S_i:.4f}")

# Test pairwise entropies
print("\nTesting pairwise entropies:")
for i in range(3):
    for j in range(i+1, 3):
        probs_i = get_subsystem_probs([i])
        probs_j = get_subsystem_probs([j])
        probs_ij = get_subsystem_probs([i, j])
        
        S_i = entropy_from_probs(probs_i)
        S_j = entropy_from_probs(probs_j)
        S_ij = entropy_from_probs(probs_ij)
        
        MI = S_i + S_j - S_ij
        
        print(f"Qubits {i},{j}:")
        print(f"  S({i}) = {S_i:.4f}")
        print(f"  S({j}) = {S_j:.4f}")
        print(f"  S({i},{j}) = {S_ij:.4f}")
        print(f"  MI = {S_i:.4f} + {S_j:.4f} - {S_ij:.4f} = {MI:.4f}")
        print() 