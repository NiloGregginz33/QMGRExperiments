#!/usr/bin/env python3
"""
Debug script to understand bitstring parsing
"""

def debug_bitstring_parsing():
    """Debug the bitstring parsing issue"""
    print("=== Debugging Bitstring Parsing ===")
    
    # Test with a simple 7-qubit bitstring
    bitstring = "1110100"
    n = 7
    
    print(f"Bitstring: {bitstring}")
    print(f"Length: {len(bitstring)}")
    print(f"Expected qubits: {n}")
    
    print("\nTesting different parsing methods:")
    
    # Method 1: Original (wrong) method
    print("Method 1 (Original - WRONG):")
    for i in range(n):
        bit = int(bitstring[-(i+1)])
        print(f"  Qubit {i}: bitstring[-({i}+1)] = bitstring[-{i+1}] = '{bitstring[-(i+1)]}' = {bit}")
    
    # Method 2: Fixed method
    print("\nMethod 2 (Fixed):")
    for i in range(n):
        bit = int(bitstring[i])
        print(f"  Qubit {i}: bitstring[{i}] = '{bitstring[i]}' = {bit}")
    
    # Method 3: Reverse the bitstring first
    print("\nMethod 3 (Reverse first):")
    reversed_bitstring = bitstring[::-1]
    print(f"  Reversed: {bitstring} -> {reversed_bitstring}")
    for i in range(n):
        bit = int(reversed_bitstring[i])
        print(f"  Qubit {i}: reversed[{i}] = '{reversed_bitstring[i]}' = {bit}")
    
    # Method 4: Use the original method but with reversed indexing
    print("\nMethod 4 (Original with correct indexing):")
    for i in range(n):
        bit = int(bitstring[-(n-i)])
        print(f"  Qubit {i}: bitstring[-({n}-{i})] = bitstring[-{n-i}] = '{bitstring[-(n-i)]}' = {bit}")

def test_with_real_counts():
    """Test with real hardware counts"""
    print("\n=== Testing with Real Hardware Counts ===")
    
    real_counts = {
        "1110100": 157,
        "0001101": 121,
        "0001100": 221,
    }
    
    n = 7
    
    for bitstring, count in real_counts.items():
        print(f"\nBitstring: {bitstring} (count: {count})")
        
        # Test both methods
        print("  Method 1 (Original):")
        for i in range(min(3, n)):
            bit = int(bitstring[-(i+1)])
            print(f"    Qubit {i}: {bit}")
        
        print("  Method 2 (Fixed):")
        for i in range(min(3, n)):
            bit = int(bitstring[i])
            print(f"    Qubit {i}: {bit}")

if __name__ == "__main__":
    debug_bitstring_parsing()
    test_with_real_counts() 