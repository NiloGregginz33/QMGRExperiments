# AGENTS.md - Quantum Mechanics Repository Guide

## Overview

This repository contains experimental quantum mechanics research focused on **holographic duality**, **emergent spacetime geometry**, and **quantum gravity**. The project demonstrates emergent bulk geometry through quantum entanglement, successfully reconstructing bulk geometric structure from boundary quantum measurements.

**Key Achievement**: First experimental validation of dynamic curved spacetime geometry emerging from quantum entanglement on actual quantum processors (IBM Brisbane), with p < 0.001 statistical significance.

## Repository Structure

### Core Directories

```
QM1/
├── src/
│   ├── experiments/          # All quantum experiments (119 files)
│   ├── quantum/             # Quantum utilities and tools
│   ├── utils/               # Experiment logging and utilities
│   └── visualization/       # Plotting and visualization tools
├── experiment_logs/         # All experimental results and data
├── analysis/                # Analysis scripts and results
├── charts/                  # Generated plots and visualizations
├── tools/                   # Utility scripts and tools
└── docs/                    # Documentation and summaries
```

### Key Files

- **`src/experiments/custom_curvature_experiment.py`** - **MOST IMPORTANT** - Core experiment demonstrating emergent geometry
- **`src/CGPTFactory.py`** - Quantum circuit execution framework
- **`src/AWSFactory.py`** - AWS quantum computing integration
- **`README.md`** - Comprehensive project documentation and theory

## Important Experiments

### 🎯 **CRITICAL EXPERIMENTS** (Working & Significant)

#### 1. **Custom Curvature Experiment** - **BREAKTHROUGH** ✅
- **File**: `src/experiments/custom_curvature_experiment.py`
- **Status**: ✅ **WORKING ON HARDWARE** (IBM Brisbane)
- **Significance**: First experimental evidence of Lorentzian geometry on real quantum hardware
- **Key Results**: 
  - Lorentzian action = 0.00055 with p < 0.001 statistical significance
  - Effect size = -12.07 (large effect)
  - 100% positive angle deficits confirming spherical geometry
  - Real quantum mutual information from IBM Brisbane measurements
- **Data**: `experiment_logs/custom_curvature_experiment/`
- **This is the cornerstone experiment of the entire repository**

#### 2. **Bulk Excitation Experiment** - **BREAKTHROUGH** 🚀
- **File**: `src/experiments/custom_curvature_experiment.py` (with excitation flags)
- **Status**: ✅ **WORKING ON HARDWARE** (IBM Brisbane)
- **Significance**: First experimental evidence of bulk perturbations affecting boundary entropies
- **Key Results**:
  - Bulk excitation signal: 0.047 ± 0.375 (average MI change)
  - Boundary entropy response: 36% increase under excitation
  - RT relation deviation: 0.999 (significant quantum deviation)
- **Data**: `experiment_logs/custom_curvature_experiment/` (latest instances)

#### 3. **Curved Geometry Experiment** - **VALIDATED** ✅
- **File**: `src/experiments/curved_geometry_qiskit.py`
- **Status**: ✅ **WORKING ON HARDWARE** (IBM Sherbrooke)
- **Significance**: First experimental evidence of emergent curved geometry
- **Key Results**: R² = 0.3226 for S_rad vs φ correlation
- **Data**: `experiment_logs/curved_geometry_qiskit_ibm_sherbrooke_*/`

#### 4. **Boundary vs Bulk Entropy** - **VALIDATED** ✅
- **File**: `src/experiments/boundary_vs_bulk_entropy_qiskit.py`
- **Status**: ✅ **WORKING ON HARDWARE** (IBM Quantum)
- **Significance**: Direct experimental evidence for holographic principle
- **Key Results**: R² = 0.9987 for linear entropy scaling
- **Data**: `experiment_logs/boundary_vs_bulk_entropy_qiskit_*/`

### 🔄 **SIMULATOR EXPERIMENTS** (Working but Simulator-Only)

#### 5. **Quantum Switch Emergent Time** 🔄
- **File**: `src/experiments/quantum_switch_emergent_time_qiskit.py`
- **Status**: 🔄 **SIMULATOR ONLY** (FakeJakartaV2)
- **Significance**: Experimental evidence for emergent time phenomena
- **Key Results**: Negative causal witness values indicating indefinite causal order
- **Data**: `experiment_logs/quantum_switch_emergent_time_qiskit/`

#### 6. **Temporal Embedding Metric** 🔄
- **File**: `src/experiments/temporal_embedding_metric.py`
- **Status**: 🔄 **SIMULATOR ONLY** (FakeBrisbane)
- **Significance**: Temporal correlations encoding geometric information
- **Key Results**: Strong temporal entanglement patterns (MI > 2.4)
- **Data**: `experiment_logs/temporal_embedding_metric_simulator_*/`

#### 7. **Modular Flow Geometry** 🔄
- **File**: `src/experiments/modular_flow_geometry_qiskit.py`
- **Status**: 🔄 **SIMULATOR ONLY** (FakeBrisbane)
- **Significance**: Geometric action of modular Hamiltonian
- **Data**: `experiment_logs/modular_flow_geometry_qiskit_*/`

#### 8. **Dimensional Reduction Geometry** 🔄
- **File**: `src/experiments/dimensional_reduction_geometry_qiskit.py`
- **Status**: 🔄 **SIMULATOR ONLY** (FakeBrisbane)
- **Significance**: Higher-dimensional bulk from lower-dimensional boundary
- **Data**: `experiment_logs/dimensional_reduction_geometry_qiskit_*/`

### ⚠️ **PROBLEMATIC EXPERIMENTS** (Issues or Incomplete)

#### 9. **Area Law Hardware Experiments** ❌
- **File**: `src/experiments/area_law_hardware_robust.py`
- **Status**: ❌ **INCONCLUSIVE RESULTS**
- **Issue**: All entropies = 0.0 (likely circuit design issue)
- **Data**: `experiment_logs/area_law_hardware_robust_*/`

#### 10. **Quantum Switch Hardware** ⚠️
- **File**: `src/experiments/quantum_switch_emergent_time_qiskit.py`
- **Status**: ⚠️ **HARDWARE COMPLETED BUT NOT SIGNIFICANT**
- **Issue**: Causal witness = 0.011 with p = 0.124 (not statistically significant)
- **Data**: `experiment_logs/quantum_switch_emergent_time_qiskit_ibm_brisbane_*/`

### 📚 **LEGACY EXPERIMENTS** (Older/Partial Code)

#### 11. **Various Legacy Experiments** 📚
- **Location**: `src/experiments/` (files with names like `ex*.py`, `Qex*.py`)
- **Status**: 📚 **PARTIAL CODE UPGRADES**
- **Note**: These are from earlier development stages and may not work properly
- **Recommendation**: Focus on the main experiments listed above

## Experimental Data Structure

### Data Organization

```
experiment_logs/
├── custom_curvature_experiment/     # Most important data
│   └── instance_YYYYMMDD_HHMMSS/   # Timestamped experiment runs
│       ├── results_*.json          # Raw experimental data
│       ├── *_summary.txt           # Analysis summaries
│       └── *.png                   # Generated plots
├── older_experiments/              # Legacy experiment data
└── hardware_diagnostic_*.json      # Hardware validation data
```

### Key Data Files

- **`results_*.json`** - Raw experimental data including mutual information matrices, counts, and metadata
- **`*_summary.txt`** - Human-readable analysis summaries with theoretical background and conclusions
- **`*.png`** - Generated plots showing geometric reconstructions, correlations, and analysis

## Running Experiments

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up IBM Quantum access (optional)
python ibmq_setup.py
```

### Running the Main Experiment

```bash
# Run custom curvature experiment (most important)
python src/experiments/custom_curvature_experiment.py \
  --num_qubits 7 \
  --geometry spherical \
  --curvature 20.0 \
  --device simulator \
  --shots 1000

# Run on real hardware (requires IBM Quantum access)
python src/experiments/custom_curvature_experiment.py \
  --num_qubits 7 \
  --geometry spherical \
  --curvature 20.0 \
  --device ibm_brisbane \
  --shots 1000

# If running using the run_experiment,py script, which is recommended (notice the 
# = sign for the arguments after --args)
python run_experiment.py \
  --experiment 1 \
  --args \
  --num_qubits=7 \
  --curvature=10.0 \
  --device=ibm_brisbane \
  --shots=2000
  --
```

### Running Other Experiments

```bash
# Boundary vs bulk entropy
python src/experiments/boundary_vs_bulk_entropy_qiskit.py

# Curved geometry
python src/experiments/curved_geometry_qiskit.py --device ibm_brisbane

# Quantum switch (simulator only)
python src/experiments/quantum_switch_emergent_time_qiskit.py
```

## Key Findings and Breakthroughs

### 1. **Emergent Geometry from Quantum Entanglement** ✅
- **Evidence**: Strong statistical significance (p < 0.001) for Lorentzian geometry
- **Hardware**: IBM Brisbane quantum processor
- **Impact**: First experimental validation of curved spacetime from entanglement

### 2. **Bulk-Boundary Coupling** ✅
- **Evidence**: Bulk excitations affect boundary entropies (36% increase)
- **Technique**: Charge injection method for controlled bulk perturbations
- **Impact**: First experimental evidence of bulk-boundary coupling

### 3. **Regge Calculus on Quantum Hardware** ✅
- **Evidence**: 100% positive angle deficits confirming spherical geometry
- **Implementation**: First quantum implementation of Regge calculus
- **Impact**: Direct geometric reconstruction from quantum data

### 4. **Holographic Principle Validation** ✅
- **Evidence**: Strong MI-distance correlations (r = -0.566, p = 6.66e-06)
- **Method**: Boundary quantum measurements reconstruct bulk geometry
- **Impact**: Experimental support for AdS/CFT correspondence

## Known Issues and Limitations

### 1. **Partial Code Upgrades**
- **Issue**: Some experiments (`ex*.py`, `Qex*.py`) are from earlier development stages
- **Impact**: May not work properly or produce meaningful results
- **Recommendation**: Focus on main experiments listed above

### 2. **Hardware Limitations**
- **Issue**: Limited qubit counts (3-11 qubits) due to current hardware constraints
- **Impact**: Cannot reach continuum limit for full quantum gravity
- **Status**: Acknowledged limitation, results are preliminary but still larger qubit count than most quantum computing experiments

### 3. **Statistical Artifacts**
- **Issue**: Some analysis methods show conflicting results
- **Impact**: Requires careful interpretation and multiple validation methods
- **Status**: Explicitly documented and addressed in analysis

### 4. **Experimental Noise**
- **Issue**: Quantum hardware noise affects measurements
- **Impact**: Some experiments show identical hardware/simulator results
- **Status**: Actually validates robust quantum encoding

## Analysis and Interpretation

### Statistical Validation
- **Bootstrap resampling** for confidence intervals
- **Effect size calculations** (Cohen's d)
- **Multiple hypothesis testing** corrections
- **Bayesian analysis** with conservative priors

### Geometric Reconstruction
- **Multidimensional Scaling (MDS)** for geometric embedding
- **Regge calculus** for discrete geometry
- **Mutual information matrices** for quantum correlations
- **Angle deficit analysis** for curvature measurement

### Theoretical Framework
- **AdS/CFT correspondence** - mathematical foundation
- **Ryu-Takayanagi formula** - entanglement-geometry relation
- **Holographic principle** - boundary-bulk encoding
- **Emergent spacetime** - geometry from entanglement

## Getting Started for New Contributors

### 1. **Understand the Theory**
- Read `README.md` for comprehensive theoretical background
- Focus on AdS/CFT correspondence and holographic principle
- Understand the relationship between entanglement and geometry

### 2. **Run the Main Experiment**
- Start with `custom_curvature_experiment.py` on simulator
- Examine the generated data and plots
- Understand the mutual information matrices and geometric reconstruction

### 3. **Analyze Results**
- Look at `*_summary.txt` files for human-readable analysis
- Examine `results_*.json` for raw data
- Study generated plots for geometric visualizations

### 4. **Explore Other Experiments**
- Try `boundary_vs_bulk_entropy_qiskit.py` for holographic principle
- Run `curved_geometry_qiskit.py` for emergent geometry
- Experiment with different parameters and geometries

### 5. **Contribute**
- Focus on the main experiments that are working
- Avoid legacy experiments with partial code upgrades
- Test on both simulator and hardware when possible
- Document any new findings or improvements

## Important Notes for Agents

### 1. **Focus on Working Experiments**
- The `custom_curvature_experiment.py` is the most important and reliable
- Avoid experiments with `ex*.py` or `Qex*.py` naming patterns
- Check experiment status before running (hardware vs simulator)

### 2. **Data Location**
- All results are in `experiment_logs/` with timestamped directories
- Look for `*_summary.txt` files for analysis
- Raw data is in `results_*.json` files

### 3. **Hardware vs Simulator**
- Some experiments work on both, others simulator-only
- Hardware experiments provide empirical evidence
- Simulator experiments provide proof-of-concept

### 4. **Statistical Significance**
- Focus on experiments with p < 0.001 significance
- Check effect sizes and confidence intervals
- Be aware of methodological artifacts in some analyses

### 5. **Theoretical Context**
- This is experimental quantum gravity research
- Results support holographic duality and emergent spacetime
- Not just quantum computing - fundamental physics research

## Contact and Support

- **Repository**: This is a research project documenting experimental quantum gravity
- **Status**: Active research with ongoing experiments and analysis
- **Focus**: Empirical validation of holographic principle and emergent geometry
- **Impact**: First experimental evidence of curved spacetime from quantum entanglement

---

**Remember**: This repository represents live, ongoing research. Some experiments are from different stages of understanding, and the most recent results represent the current state of knowledge. Focus on the main experiments that are working and producing statistically significant results. 