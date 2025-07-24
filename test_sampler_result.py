#!/usr/bin/env python3

import sys
sys.path.append('src')

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# Create a simple circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

print("Circuit:")
print(qc)

# Connect to IBM
service = QiskitRuntimeService()
backend = service.backend("ibm_brisbane")

# Run the circuit
sampler = Sampler(backend)
transpiled_qc = transpile(qc, backend)
job = sampler.run([transpiled_qc], shots=10)
result = job.result()

print("\nResult type:", type(result))
print("Result attributes:", [attr for attr in dir(result) if not attr.startswith('_')])

if hasattr(result, '_pub_results'):
    pub_result = result._pub_results[0]
    print("\nPub result type:", type(pub_result))
    print("Pub result attributes:", [attr for attr in dir(pub_result) if not attr.startswith('_')])
    
    if hasattr(pub_result, 'data'):
        data = pub_result.data
        print("\nData type:", type(data))
        print("Data attributes:", [attr for attr in dir(data) if not attr.startswith('_')])
        
        if hasattr(data, 'get_bitstrings'):
            bitstrings = data.get_bitstrings()
            print("\nBitstrings:", bitstrings)
        else:
            print("\nNo get_bitstrings method")
            print("Data items:", data.items)
            print("Data keys:", data.keys)
            print("Data meas:", data.meas)
            print("Data shape:", data.shape)
            print("Data size:", data.size)
            print("Data values:", data.values)
            
            # Explore the BitArray
            bitarray = data.meas
            print("\nBitArray type:", type(bitarray))
            print("BitArray attributes:", [attr for attr in dir(bitarray) if not attr.startswith('_')])
            print("BitArray shape:", bitarray.shape)
            print("BitArray num_shots:", bitarray.num_shots)
            print("BitArray num_bits:", bitarray.num_bits)
            
            # Try to get the bitstrings
            if hasattr(bitarray, 'get_bitstrings'):
                bitstrings = bitarray.get_bitstrings()
                print("\nBitstrings from BitArray:", bitstrings)
            else:
                print("\nNo get_bitstrings method on BitArray")
                # Try to access the data directly
                print("BitArray data:", bitarray)
else:
    print("\nNo _pub_results attribute") 