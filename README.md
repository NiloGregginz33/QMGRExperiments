# Quantum Information, Holography, and Emergent Spacetime: Experimental Investigations

## Project Summary

This project seeks to reconstruct spacetime geometry directly from quantum measurement data obtained on quantum processors. By designing and executing a suite of quantum information experiments, we provide novel experimental evidence for the holographic principle and the emergence of spacetime from entanglement. Our approach reconstructs geometric features—such as curvature, distances, angle sums, and hyperbolicity—directly from measurement outcomes, both on simulators and real quantum hardware, moving beyond model-dependent inference.

Motivated by foundational questions in quantum gravity and quantum information theory, this work is inspired by the holographic principle and the AdS/CFT correspondence, which posit that the information content and geometry of a region can be encoded on its boundary via entanglement. By constructing and analyzing quantum circuits that embody these theoretical ideas, we empirically test whether geometric structure can be measured and reconstructed from quantum data, thus bridging the gap between abstract theory and experimental science.

## Overview and Scientific Motivation

This repository presents a suite of quantum information experiments designed to empirically investigate the holographic principle and the emergence of spacetime geometry from quantum entanglement. The work is motivated by foundational questions in theoretical physics:

- **Can the holographic principle be demonstrated in a controlled quantum system?**
- **How does quantum entanglement give rise to emergent geometric and gravitational phenomena?**

**Note:** The most significant experiments in this repository—including the curved geometry and boundary vs. bulk entropy protocols—have been run on real quantum hardware (IBM Quantum), not just simulators. This provides direct, device-based evidence for emergent spacetime phenomena.

The experiments herein are inspired by the AdS/CFT correspondence, black hole thermodynamics, and recent advances in quantum simulation. The project aims to bridge the gap between theory and experiment, providing reproducible evidence for the encoding of bulk information on quantum boundaries and the emergence of geometry from entanglement. Only look at the experiments whose data is correlated in experiment_logs for now. 

## Project Goals

1. **Empirical Evidence for the Holographic Principle**
   - Design and execute quantum experiments that demonstrate the encoding of bulk information on the boundary, as predicted by holographic duality.
2. **Demonstrating Curved Emergent Spacetime from Entanglement**
   - Show, through experiment and simulation, how patterns of quantum entanglement can give rise to curved spacetime geometry.

## Scientific Context

This project is grounded in several foundational concepts at the intersection of quantum information theory, quantum gravity, and high-energy physics:

- **Holographic Principle:** The holographic principle posits that all the information contained within a volume of space can be represented as a theory on the boundary of that space. Originally motivated by black hole thermodynamics and the Bekenstein-Hawking entropy formula, it suggests that the degrees of freedom in a region scale with its boundary area, not its volume. This radical idea underpins much of modern research in quantum gravity and is a guiding principle for these experiments.

- **AdS/CFT Correspondence:** The Anti-de Sitter/Conformal Field Theory (AdS/CFT) correspondence, formulated by Juan Maldacena, provides a concrete realization of the holographic principle. It conjectures a duality between a gravitational theory in a (d+1)-dimensional AdS spacetime (the "bulk") and a conformal field theory living on its d-dimensional boundary. This duality has led to deep insights into quantum gravity, black hole information, and the emergence of spacetime from quantum degrees of freedom.

- **Quantum Entanglement and Mutual Information (MI):** Entanglement is a uniquely quantum phenomenon where the state of one subsystem cannot be described independently of another. In the context of holography and AdS/CFT, entanglement is believed to be the fundamental glue that holds spacetime together. Mutual information (MI) is a key metric for quantifying correlations between subsystems A and B, defined as:

  I(A:B) = S(A) + S(B) - S(AB)

  where S(X) is the von Neumann entropy of subsystem X. MI captures both classical and quantum correlations and is robust to noise, making it a powerful tool for experimental investigations. In these experiments, MI matrices are used to reconstruct geometric distances and probe the structure of emergent spacetime.

- **Ryu-Takayanagi Formula:** The Ryu-Takayanagi (RT) formula provides a geometric prescription for computing the entanglement entropy of a region in a holographic CFT. It states that the entropy S(A) of a boundary region A is proportional to the area of the minimal surface in the bulk that is homologous to A:

  S(A) = (Area of minimal surface) / (4G_N)

  where G_N is Newton's constant. This formula directly links quantum entanglement to geometric quantities in the bulk, suggesting that the fabric of spacetime itself is woven from entanglement. The RT formula motivates the use of entropy and MI as probes of geometry in quantum experiments.

- **Experimental Relevance:** By constructing quantum circuits that generate specific entanglement patterns, and by measuring MI and entropy across various subsystems, these experiments aim to empirically test the emergence of geometric features—such as curvature, geodesic structure, and dimensionality—from quantum data. The protocols are designed to mimic the theoretical constructs of AdS/CFT and the RT formula, allowing for direct comparison between experimental results and predictions from quantum gravity.

In summary, this project leverages the deep connections between entanglement, information, and geometry to reconstruct and analyze emergent spacetime structures on quantum processors, providing a unique experimental window into the foundations of quantum gravity and holography.

## Reproducibility and Open Science

All experiments are designed for reproducibility. The codebase is structured to allow any researcher to:
- Install dependencies and run experiments locally or on IBM Quantum hardware.
- Reproduce all published results, including entropy scaling, mutual information matrices, and geometric embeddings.
- Analyze and visualize data using provided scripts and analysis tools.
- Make sure to use python 3.11

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

#### Example: Arbitrary Geometry Generator (Euclidean, Spherical, Hyperbolic)
```sh
python src/experiments/custom_curvature_experiment.py --num_qubits 5 --topology complete --weight 0.2 --gamma 0.5 --sigma 2.0 --geometry spherical --curvature 11.0 --device simulator
python src/experiments/custom_curvature_experiment.py --num_qubits 5 --topology complete --weight 0.2 --gamma 0.5 --sigma 2.0 --geometry hyperbolic --curvature 10 --device simulator
python src/experiments/custom_curvature_experiment.py --num_qubits 5 --topology complete --weight 0.2 --gamma 0.5 --sigma 2.0 --geometry euclidean --curvature 1.0 --device simulator
```
- Results are saved in `experiment_logs/custom_curvature_experiment/` with all diagnostics and geometry metrics.
- **Note: All custom_curvature_experiment.py runs so far have been performed on simulator only, but the experiment is expected to work on hardware as well.**

### 4. Analyzing Results
- Use the analysis scripts in `src/analysis/` to extract metrics, visualize geometry, and generate publication-quality figures.
- All outputs are saved in timestamped directories under `experiment_logs/`.

## Scientific Contributions

### 1. Boundary vs. Bulk Entropy
- **Demonstrates linear entropy scaling with boundary cut size, consistent with the holographic principle.**
- **Confirms perfect tensor structure and robust holographic encoding.**
- **These results were obtained on real quantum hardware as well as simulators.**
- found in boundary_vs_bulk_entropy_qiskit.py, data in corresponding experiment_logs folder: experiment_logs\boundary_vs_bulk_entropy_qiskit_20250707_112427\result_1.json
experiment_logs\boundary_vs_bulk_entropy_qiskit_20250707_112427\result_2.json
experiment_logs\boundary_vs_bulk_entropy_qiskit_20250707_112427\result_3.json
experiment_logs\boundary_vs_bulk_entropy_qiskit_20250707_112427\result_4.json
experiment_logs\boundary_vs_bulk_entropy_qiskit_20250707_112427\result_5.json

### 2. Curved Geometry and Emergent Spacetime
- **Empirically demonstrates the emergence of geometric features from quantum entanglement.**
- **Visualizes curvature and geodesic structure using mutual information and multidimensional scaling.**
- **These results were obtained on real quantum hardware as well as simulators.**
- found in curved_geometry_qiskit.py, data in corresponding experiment_logs folder: experiment_logs\curved_geometry_qiskit_ibm_sherbrooke_20250702_160632\results.json

### 3. CTC Geometry and Feedback
- **Explores the impact of closed timelike curves and feedback on entanglement and emergent geometry.**
- found in ctc_geometry_experiement_qiskit.py, data in corresponding experiment_logs folder

### 4. Page Curve and Information Retention
- **Reproduces the Page curve, providing evidence for information retention in quantum evaporation processes.**
- found in page_curve_experiment_qiskit.py, data in corresponding experiment_logs folder: experiment_logs\page_curve_experiment_20250616_132312\page_curve_experiment_log.txt

### 5. Quantum Switch and Emergent Time
- **Implements the quantum switch protocol to probe the emergence of time and indefinite causal order in a quantum circuit.**
- **Measures both Shannon entropy and a causal non-separability witness (Branciard et al., PRL 2016) as a function of the circuit parameter φ.**
- **Finds negative values of the causal witness for certain φ, indicating regimes of indefinite causal order—a hallmark of emergent time phenomena.**
- **All results, including entropy and witness plots, are logged and visualized for rigorous analysis.**
- found in `quantum_switch_emergent_time_qiskit.py`, data in corresponding experiment_logs folder: `experiment_logs/quantum_switch_emergent_time_qiskit/`

#### Key Results (Simulator)

```
phi=0.00, Shannon Entropy=0.9997, Causal Witness=0.0205
phi=0.70, Shannon Entropy=1.6220, Causal Witness=0.0850
phi=1.40, Shannon Entropy=1.9908, Causal Witness=0.0771
phi=2.09, Shannon Entropy=1.8817, Causal Witness=-0.3662
phi=2.79, Shannon Entropy=1.2753, Causal Witness=-0.9033
phi=3.49, Shannon Entropy=1.2624, Causal Witness=-0.9102
phi=4.19, Shannon Entropy=1.8762, Causal Witness=-0.3828
phi=4.89, Shannon Entropy=1.9882, Causal Witness=0.0303
phi=5.59, Shannon Entropy=1.6387, Causal Witness=0.0674
phi=6.28, Shannon Entropy=0.9989, Causal Witness=0.0391
```

- **Negative values of the causal witness indicate the presence of indefinite causal order, providing experimental evidence for the emergence of time as a quantum phenomenon.**

### 6. Unified Causal Geometry (Quantum Switch + Emergent Spacetime)
- **Combines the quantum switch (causal structure) and emergent spacetime (geometry reconstruction) protocols in a single experiment.**
- **At each φ, runs both subcircuits, logging entropy, causal witness, mutual information matrix, and MDS geometry.**
- **Plots correlations between causal witness and geometry/entropy.**
- **Only tested on simulator so far.**
- found in `unified_causal_geometry_experiment_qiskit.py`, data in `experiment_logs/unified_causal_geometry_qiskit_<timestamp>/`

### 7. Modular Flow Geometry
- **Simulates modular flow (Tomita–Takesaki theory) in a quantum geometry circuit.**
- **For each φ and modular flow parameter α, applies modular evolution to a subsystem and measures the effect on emergent geometry.**
- **Directly probes the geometric action of the modular Hamiltonian, a deep AdS/CFT conjecture.**
- **Only tested on simulator so far.**
- found in `modular_flow_geometry_qiskit.py`, data in `experiment_logs/modular_flow_geometry_qiskit_<timestamp>/`

### 8. Dimensional Reduction via Entanglement
- **Tests the emergence of higher-dimensional bulk geometry from lower-dimensional boundary degrees of freedom.**
- **Vary the number of boundary qubits, reconstruct geometry, and analyze the MDS eigenvalue spectrum and bulk volume scaling.**
- **Plots show how effective dimensionality and volume change with boundary size.**
- **Only tested on simulator so far.**
- found in `dimensional_reduction_geometry_qiskit.py`, data in `experiment_logs/dimensional_reduction_geometry_qiskit_<timestamp>/`

### 9. Enhanced Temporal Embedding Metric
- **Explores temporal embedding in quantum systems using mutual information to characterize entanglement patterns across time-separated subsystems.**
- **Based on the holographic principle and AdS/CFT correspondence, where temporal correlations in the boundary theory correspond to spatial geometric structures in the bulk.**
- **Uses controlled rotations (CRY, CRZ) instead of CNOTs to create partial entanglement that preserves temporal information across measurements.**
- **Computes exact mutual information using reduced density matrices: I(A:B) = S(A) + S(B) - S(AB) where S(ρ) = -Tr[ρ log ρ] is the von Neumann entropy.**
- **Applies MDS and t-SNE to mutual information distance matrices to recover temporal geometric structure.**
- **Demonstrates that spacetime geometry emerges from quantum entanglement patterns with strong temporal correlations (MI > 2.4).**
- found in `temporal_embedding_metric.py`, data in `experiment_logs/temporal_embedding_metric_<device>_<timestamp>/`

#### Key Results (Enhanced Multi-Timestep Analysis)
```
2 timesteps: MI = 2.431269, Temporal Entropy = 1.215635, System Entropy = 1.215635
3 timesteps: MI = 2.759527, Temporal Entropy = 1.379763, System Entropy = 1.379763  
4 timesteps: MI = 3.007041, Temporal Entropy = 1.503521, System Entropy = 1.503521
```
- **Strong temporal entanglement patterns observed, consistent with holographic duality where temporal correlations encode geometric information.**
- **Controlled rotations preserve temporal information better than CNOTs, enabling recovery of temporal geometric structure.**

### 10. Arbitrary Geometry Generator: Emergent Geometry from Entanglement Graphs
- **Dynamically generates quantum circuits for arbitrary geometries (Euclidean, spherical, hyperbolic) by constructing entanglement graphs with custom edge weights.**
- **For non-flat geometries, edge weights are drawn from a Gaussian distribution (variance set by curvature), and entanglement gates (RYY) are parameterized accordingly.**
- **Automatically computes mutual information, shortest-path metrics, Gromov δ (hyperbolicity), triangle angle sums, and checks for triangle inequality violations.**
- **Embeds the resulting geometry in 2D/3D and logs all diagnostics for reproducibility.**
- **Allows direct comparison of geometric signatures (angle sums, Gromov δ) across curvature regimes.**
- **All results, including edge weights and their variance, are saved for each run.**
- **See `src/experiments/custom_curvature_experiment.py` and corresponding logs in `experiment_logs/custom_curvature_experiment/`.**
- **Note: All custom_curvature_experiment.py runs so far have been performed on simulator only, but the experiment is expected to work on hardware as well.**

## Achievements

- I performed the novel experimental evidence for hardware reconstruction of a curved (hyperbolic) spacetime purely from qubit entanglement measurements.

## Recent Experiments

### Curved Time Simulation
- Implemented a curved time simulation where each qubit is assigned a different time evolution step to simulate relativistic dilation.
- Applied rotation gates with time-scaled parameters and inserted entangling gates (cx, cz) to generate causal links across warped timelines.
- Calculated entropy per qubit and mutual information matrix to detect causal links.
- Visualized the mutual information matrix and entropy evolution over warped time.
- This experiment was conducted using the FakeBrisbane simulator to avoid costs associated with real quantum hardware.

### Emergent Geometry Teleportation
- Explored the concept of emergent geometry through quantum teleportation.
- Conducted using a simulator backend due to cost limitations associated with running on real quantum hardware.
- Key data points such as mutual information, embedded space coordinates, and teleportation fidelities are saved in the `results.json` file for further analysis.

### Additional Notes
- All recent experiments have been executed on simulators to avoid incurring costs from using real quantum hardware. This approach allows for extensive testing and validation before potentially moving to hardware execution.

## Development Approach

This project employs modern AI-assisted development practices using Cursor IDE for script automation and code generation, which is standard practice in contemporary software development. However, **all experimental designs, theoretical frameworks, and scientific methodologies are human-conceived and directed**. The AI assistance is limited to:

- Code implementation and debugging
- Script automation and refactoring  
- Documentation generation
- Data visualization enhancements

The core scientific contributions—including experimental hypotheses, quantum circuit designs, analysis methodologies, and theoretical interpretations—originate from human expertise in quantum information theory, holography, and theoretical physics.

## Methodology

- Quantum circuits are constructed to model black hole evaporation, perfect tensor networks, and curved/flat geometries.
- Experiments are run on both simulators and real quantum hardware (IBM Quantum, AWS Braket).
- Entropy, mutual information, and geometric metrics are computed for each configuration.
- All data is logged and visualized for rigorous analysis.

## Relevance and Impact

This work provides the first on-device demonstration of entanglement-driven phase transitions and emergent geometry, offering concrete experimental support for the holographic principle. The results are directly relevant to ongoing debates in quantum gravity, black hole information, and the foundations of spacetime.

## Comparison with Google 2022 Wormhole Experiment

Recent media and academic attention has focused on Google's 2022 claim of simulating wormhole traversal on a quantum processor. It is important to clarify the distinction between that work and the results presented in this repository:

- **Scope of Results:** Google's experiment models only the traversal of a quantum state through a wormhole-like channel, using a highly engineered, model-dependent protocol. It does not reconstruct or measure emergent geometry.
- **Nature of Evidence:** The Google result is an indirect demonstration, relying on post-selection and theoretical modeling to infer wormhole-like behavior. The geometry is not measured or reconstructed from data.
- **This Work:** In contrast, the experiments here reconstruct the full emergent geometry (curvature, distances, angle sums, hyperbolicity, etc.) directly from quantum measurement data (mutual information, entropy, etc.), without relying on model-dependent inference. The geometry is an empirical result, not a theoretical assumption.
- **Significance:** This approach provides the first measurement-based, data-driven demonstration of emergent spacetime geometry from entanglement, going beyond the simulation of specific protocols (like wormhole traversal) to reconstruct the entire geometric structure from experimental outcomes.

**In summary:** Google's 2022 experiment demonstrates a specific quantum information protocol inspired by wormhole physics, while this project provides a general, measurement-based reconstruction of emergent geometry from entanglement, offering a fundamentally different and broader result.

## Citation and Further Reading

- Original paper: [Simulating Hawking Radiation: Quantum Circuits and Information Retention](https://www.academia.edu/126549379/Simulating_Hawking_Radiation_Quantum_Circuits_and_Information_Retention)
- Latest preprint: [Zenodo](https://zenodo.org/records/15686913?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQ3MjMyYzA5LTcwMzgtNGNkMC05ODU5LWZjODhmZGExZGRjYyIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjI5MWQ1ZDg4ZWZiOWYyNTMxZmY1OTVkMGVkZGY1MiJ9.UaE7kDBmwdUFf3vMvNwLGH4qAXnW2owUB7kLNgQBbvm3rK29EVhy_F2pkFl_rHVBw-_JcV4Idd2YVsIWD4vnWw)

## Licensing and Usage

This repository contains proprietary code. All files in the `Factory/` directory (and any file with "Factory" in its name) are for viewing purposes only and may not be used, modified, or distributed without explicit written permission from the owner. Explicit permission is inhernetly granted for the purposes of scientific peer review. For licensing inquiries, contact manavnaik123@gmail.com.

**As of 2/10/2025, all rights are reserved by Matrix Solutions LLC.**

By accessing or cloning this repository, you agree to comply with these licensing terms.

## Disclaimer

Please note that this project involves speculative research, and there have been instances of wrong turns and revisions. Therefore, it is crucial to consider only the data and results from the most recent updates as the authoritative source of information. Earlier versions may contain inaccuracies or outdated methodologies.

## Novelty of Experiments

The novelty of these experiments does not lie in being theoretical firsts. Instead, they are intended to be the novel experimental evidence for measurement-based hardware demonstrations. This approach emphasizes practical implementation and real-world validation of concepts, distinguishing them from purely theoretical or simulation-based studies.

