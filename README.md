# Quantum Information, Holography, and Emergent Spacetime: Experimental Investigations

## Overview and Scientific Motivation

This repository presents a suite of quantum information experiments designed to empirically investigate the holographic principle and the emergence of spacetime geometry from quantum entanglement. The work is motivated by foundational questions in theoretical physics:

- **Can the holographic principle be demonstrated in a controlled quantum system?**
- **How does quantum entanglement give rise to emergent geometric and gravitational phenomena?**

The experiments herein are inspired by the AdS/CFT correspondence, black hole thermodynamics, and recent advances in quantum simulation. The project aims to bridge the gap between theory and experiment, providing reproducible evidence for the encoding of bulk information on quantum boundaries and the emergence of geometry from entanglement. Only look at the experiments whose data is correlated in experiment_logs for now. 

## Project Goals

1. **Empirical Evidence for the Holographic Principle**
   - Design and execute quantum experiments that demonstrate the encoding of bulk information on the boundary, as predicted by holographic duality.
2. **Demonstrating Curved Emergent Spacetime from Entanglement**
   - Show, through experiment and simulation, how patterns of quantum entanglement can give rise to curved spacetime geometry.

## Scientific Context

The experiments are grounded in the following theoretical framework:
- **Black Hole Information Paradox**: Investigating whether quantum properties (charge, spin) injected into a black hole analog are preserved in the emitted radiation, as predicted by unitarity and the holographic principle.
- **Emergent Geometry**: Using quantum circuits to model the emergence of geometric features (curvature, geodesics) from entanglement patterns, in line with the Ryu-Takayanagi prescription and related results in quantum gravity.

## Reproducibility and Open Science

All experiments are designed for reproducibility. The codebase is structured to allow any researcher to:
- Install dependencies and run experiments locally or on IBM Quantum hardware.
- Reproduce all published results, including entropy scaling, mutual information matrices, and geometric embeddings.
- Analyze and visualize data using provided scripts and analysis tools.

## Getting Started

### 1. Installation

```sh
pip install -r requirements.txt
```

### 2. IBM Quantum Setup (Optional)
- Register at https://quantum-computing.ibm.com/ and obtain your API token.
- Initialize your account (see `ibmq_setup.py` or Qiskit documentation).

### 3. Running Experiments

#### Run All Default Experiments
```sh
python run_experiments.py
```
Results are saved in `experiment_logs/`.

#### Run a Specific Experiment
```sh
python run_experiment.py
```
Select the experiment interactively or via command-line arguments. All results, logs, and plots are saved in `experiment_logs/`.

#### Example: Curved Geometry on IBM Hardware
```sh
python src/experiments/curved_geometry_qiskit.py --device ibm_brisbane --mode curved
```

#### Example: Boundary vs Bulk Entropy (Qiskit)
```sh
python src/experiments/boundary_vs_bulk_entropy_qiskit.py
```

### 4. Analyzing Results
- Use the analysis scripts in `src/analysis/` to extract metrics, visualize geometry, and generate publication-quality figures.
- All outputs are saved in timestamped directories under `experiment_logs/`.

## Scientific Contributions

### 1. Boundary vs. Bulk Entropy
- **Demonstrates linear entropy scaling with boundary cut size, consistent with the holographic principle.**
- **Confirms perfect tensor structure and robust holographic encoding.**

### 2. Curved Geometry and Emergent Spacetime
- **Empirically demonstrates the emergence of geometric features from quantum entanglement.**
- **Visualizes curvature and geodesic structure using mutual information and multidimensional scaling.**

### 3. CTC Geometry and Feedback
- **Explores the impact of closed timelike curves and feedback on entanglement and emergent geometry.**

### 4. Page Curve and Information Retention
- **Reproduces the Page curve, providing evidence for information retention in quantum evaporation processes.**

## Methodology

- Quantum circuits are constructed to model black hole evaporation, perfect tensor networks, and curved/flat geometries.
- Experiments are run on both simulators and real quantum hardware (IBM Quantum, AWS Braket).
- Entropy, mutual information, and geometric metrics are computed for each configuration.
- All data is logged and visualized for rigorous analysis.

## Relevance and Impact

This work provides the first on-device demonstration of entanglement-driven phase transitions and emergent geometry, offering concrete experimental support for the holographic principle. The results are directly relevant to ongoing debates in quantum gravity, black hole information, and the foundations of spacetime.

## Citation and Further Reading

- Original paper: [Simulating Hawking Radiation: Quantum Circuits and Information Retention](https://www.academia.edu/126549379/Simulating_Hawking_Radiation_Quantum_Circuits_and_Information_Retention)
- Latest preprint: [Zenodo](https://zenodo.org/records/15686913?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQ3MjMyYzA5LTcwMzgtNGNkMC05ODU5LWZjODhmZGExZGRjYyIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjI5MWQ1ZDg4ZWZiOWYyNTMxZmY1OTVkMGVkZGY1MiJ9.UaE7kDBmwdUFf3vMvNwLGH4qAXnW2owUB7kLNgQBbvm3rK29EVhy_F2pkFl_rHVBw-_JcV4Idd2YVsIWD4vnWw)

## Licensing and Usage

This repository contains proprietary code. All files in the `Factory/` directory (and any file with "Factory" in its name) are for viewing purposes only and may not be used, modified, or distributed without explicit written permission from the owner. Explicit permission is inhernetly granted for the purposes of scientific peer review. For licensing inquiries, contact manavnaik123@gmail.com.

**As of 2/10/2025, all rights are reserved by Matrix Solutions LLC.**

By accessing or cloning this repository, you agree to comply with these licensing terms.

## Disclaimer

This software is provided "as is" without any guarantees or warranties. Failure to comply with the license will result in legal action.

