# Project Goals

This project is dedicated to two primary scientific goals:

1. **Empirical Evidence for the Holographic Principle**
   - To design and run quantum experiments that provide direct, reproducible evidence for the holographic principle—showing that information about a quantum system can be fully encoded on its boundary, as predicted by modern theoretical physics.

2. **Demonstrating Curved Emergent Spacetime from Entanglement**
   - To experimentally demonstrate how curved spacetime geometry can emerge from patterns of quantum entanglement, using both simulation and real quantum hardware. The aim is to move beyond theory and visualize the emergence of geometry from quantum information.

These goals drive the design of all experiments and analysis in this repository. I understand that I start talking about black hole thermodynamics but this is meant to illustrate the line of reasoning I used to devise my experimental methods.

I began by investigating whether injected properties like charge and spin remain encoded in idealized Hawking radiation. That black-hole‑inspired work quickly evolved into a broader investigation of how spacetime can emerge from quantum entanglement. Using the AWSFactory scripts, I have demonstrated these "geometry from entanglement" effects on real AWS Braket hardware. Feedback on the methodology and results is welcome—these preliminary experiments on real quantum hardware suggest behavior consistent with holographic models, though they have not yet been formally peer reviewed yet though a pre print has been published (see below). Hardware-based results were obtained prior to repository optimization for local testing. Final repo structure maintains AWS execution compatibility, though recent edits were locally validated due to hardware cost constraints.

These circuits run without anomalies on readily accessible quantum hardware, reproducing Leonard Susskind's predictions for information preservation. More importantly, they reveal emergent spacetime behavior consistent with holographic models—something I was able to visualize directly using the AWSFactory tools.

Then, 

a modifiable 4D manifold (and other geometries) emerging from entanglement as a result of following the evidence of the first experiment.

i feel like a lot of us did not really want to wait literal trillions of years to spend doing or arguing about in order to know the next layer of physics

DO IT YOURSELF

Literally the code is written and you are free to use under the license and see and just copy. So please, if you have the free time just run the code yourself I made this very easy for reproducibility and idk its more convenient on everyone I feel.

If you want to run the code on your own machine, use ** pip install -r requirements.txt **, maybe do this in a venv too (but you dont have to), heres a link in case you opt for it https://stackoverflow.com/questions/43069780/how-to-create-virtual-env-with-python-3. this is because we are using python for this so please install it. it helps to go through them in order to see how things keep building over time. Also, if you want to run the quantum experiments you have to initialize your IBM Quantum account, but the simulations work the exact same as the quantum code so if that seems like too much effort, you dont have to. Files are executed by using python [name of the python file, including the .py at the end] HINT STOP AT ex9 everything past that point is just a misunderstanding I had. Also all the code outputs are logged in the DOCS folder. Installing a WSL environment may be necessary but this can be simple https://learn.microsoft.com/en-us/windows/wsl/install. Ill try to add a file that can initialize the IBM Q environment as well. set IBM_QUANTUM_API_KEY=your_api_key is the command line command you need to set the env variable on windows, afterwhich running the ibmq_setup.py code should initialize the rest of the variables for you. Ask if you want to experiment with the factory code since the bulk of experiments are in the factory code (some outputs are there for you to see, specifically in CGPTFactory.py)
For the energent spacetime dynamics, we are using a different provider, specifically AWS BRAKET, just because I ran out of free minutes on IBM when I had these thoughts and they hella verify identity so you cant just make a new account. type ** aws configure ** in your terminal is the command to set it up, though you will have to configure an actual aws account, likely via web portal. Make sure to give the account BRAKET and S3. If you have any issues with the code feel free to message me, I know its a little messy. You can also use the helper script provided for IBM backends (ibmq_setup.py) but for aws youre going to have to do it manually. 

# ⚠️ Note on Deprecated Scripts

**The individual experiment scripts (`ex1.py`, `ex2.py`, `ex3.py`, etc.) are deprecated.**

- These files represent early or legacy experiments and are no longer maintained.
- For all current and high-level experiments, use the provided runners:
  - `run_experiment.py` — to select and run any experiment interactively or by number
  - `run_experiments.py` — to run a fixed set of demonstration experiments in sequence

**Do not run the `ex1.py`, `ex2.py`, ... scripts directly. Use the experiment runners above for up-to-date results and logging.**

---

# Running Quantum Information & Geometry Experiments

## How to Use `run_experiments.py`

This repository provides a script to run the main quantum information and geometry experiments, including hardware and simulator variants. All results and logs are saved in the `experiment_logs` directory for reproducibility and analysis.

### **Step-by-Step Instructions**

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **(Optional) Set up your IBM Quantum account:**
   - Register at https://quantum-computing.ibm.com/ and get your API token.
   - Initialize your account (see ibmq_setup.py or Qiskit docs).

3. **Run the experiments:**
   ```sh
   python run_experiments.py
   ```
   This will run the default set of experiments and save results in `experiment_logs/`.

4. **To run the curved geometry experiment on IBM hardware:**
   ```sh
   python src/experiments/curved_geometry_qiskit.py --device ibm_brisbane --mode curved
   ```
   - Replace `ibm_brisbane` with your desired IBM backend if needed.
   - Results and plots will be saved in a timestamped folder under `experiment_logs/`.

5. **Check your results:**
   - Look in the `experiment_logs/` directory for output folders, results.json, summary.txt, and plots.

For more details on each experiment, see the scientific summaries below and the documentation in the `Docs/` folder.

---

# How to Select and Run Any Experiment (`run_experiment.py`)

The script `run_experiment.py` is the recommended way to *choose and run any experiment* from `src/experiments/` (including sub-experiments in `run_simple_experiments.py`). It dynamically lists all available experiments, lets you select one interactively or via command-line, and supports running on different quantum devices.

## Interactive Mode (Recommended)

Simply run:

```bash
python run_experiment.py
```

You will see a numbered menu of all available experiments and sub-experiments, for example:

```
Available experiments:
  1. Emergent Spacetime
  2. Curved Geometry Analysis
  3. Test Experiment
  4. Star Geometry Experiment
  5. Boundary Vs Bulk Entropy Experiment
  6. Simple: Holographic (sub-experiment)
  7. Simple: Temporal (sub-experiment)
  8. Simple: Contradictions (sub-experiment)
Select an experiment by number: 1
```

You will be prompted to select the experiment by number. The script will run the selected experiment with the default device (`simulator`) and shots (`1024`).

## Command-Line Mode

You can also specify the experiment number and options directly:

```bash
python run_experiment.py --experiment 2 --device ionq --shots 500
```

- `--experiment N` : Run experiment number N (see the list printed by the script)
- `--device DEVICE` : Choose the quantum device (`simulator`, `ionq`, `rigetti`, `oqc`)
- `--shots N` : Number of shots (default: 1024)

## Output

- All results, logs, and plots are saved in the `experiment_logs/` directory, organized by experiment and device.
- The script prints a summary and any errors to the console.

## Adding New Experiments

- Any new `.py` file added to `src/experiments/` (except those starting with `run_` or named `__init__.py`/`experiment_logger.py`) will automatically appear in the menu.
- Sub-experiments in `run_simple_experiments.py` are always included.

## Comparison: `run_experiment.py` vs `run_experiments.py`

- **`run_experiment.py`**: Lets you *choose* and run any experiment (or sub-experiment) interactively or by number. This is the main entry point for running individual experiments.
- **`run_experiments.py`**: Runs a fixed set of demonstration experiments in sequence. Use this if you want to run all default experiments at once, not for selecting a specific one.

For advanced batch runs, see also `run_all_experiments.py` and `run_hardware_experiments.py`.

---

# Extracting Metrics & Geometry: Advanced Analysis Functions

In addition to running the main experiments, you can extract additional metrics and geometric data from your circuits and results using a suite of analysis functions in `src/CGPTFactory.py`. These functions allow you to probe entropy, mutual information, and emergent geometry at a deeper level, or to analyze your own circuits and outputs.

## Key Metric & Geometry Functions

- **Entropy & Information:**
  - `block_entropy(qc, block_qubits)`: Compute the von Neumann entropy of a specified block of qubits in a circuit.
  - `calculate_shannon_entropy(counts, num_shots)`: Compute the classical (Shannon) entropy from measurement counts.
  - `calculate_von_neumann_entropy(qc, num_radiation_qubits)`: Compute total, black hole, and radiation entropies for a black hole/radiation circuit.
  - `calculate_subsystem_qiskit_entropy(qc)`: Compute the entropy of each individual qubit in a circuit (requires Qiskit Aer simulator).
  - `analyze_von_neumann_qiskit_entropy(statevector)`: Analyze the von Neumann entropy of a statevector (for advanced users).

- **Curvature & Geometry:**
  - `compute_plaquette_curvature(qc, rows, cols, plaquettes)`: Compute Ricci-like curvature for each plaquette (square) in a 2D lattice circuit.
  - `compute_face_curvature(qc, faces)`: Compute face-based curvature (mutual information between pairs of qubit pairs) for a list of faces.
  - `compute_cell_curvature(qc, cells, faces_of_cell)`: Compute cell-based curvature (sum of face entropies minus cell entropy) for a list of cells.
  - `build_faces_of_cell(cells, faces)`: Utility to map each cell to its bounding faces (needed for cell curvature).
  - `list_plaquettes(rows, cols)`, `list_3d_faces(L)`, `list_3d_cells(L)`: Utilities to generate geometry for 2D/3D lattices.

- **Mutual Information:**
  - `run_entanglement_measurements(theta_dict)`: Build a circuit with variable entangling angles and extract pairwise mutual information for each edge.

## Example Usage

You can import and use these functions in your own scripts, or modify the experiment files to call them for additional analysis. For example:

```python
from src.CGPTFactory import (
    block_entropy, calculate_shannon_entropy, calculate_von_neumann_entropy,
    calculate_subsystem_qiskit_entropy, analyze_von_neumann_qiskit_entropy,
    compute_plaquette_curvature, compute_face_curvature, compute_cell_curvature,
    build_faces_of_cell, list_plaquettes
)

# Example: Compute block entropy for a custom circuit
qc = ...  # your QuantumCircuit
block_S = block_entropy(qc, block_qubits=[0,1])

# Example: Compute curvature for a 2D lattice
rows, cols = 3, 3
plaquettes = list_plaquettes(rows, cols)
curvatures = compute_plaquette_curvature(qc, rows, cols, plaquettes)

# Example: Compute subsystem entropy
entropies = calculate_subsystem_qiskit_entropy(qc)
```

You can also adapt the experiment scripts in `src/experiments/` to call these functions on the circuits/results they generate, or use them in your own analysis pipelines.

**Tip:** For geometric analysis (curvature, mutual information), see the `curved_geometry` and `emergent_spacetime` experiment files for practical examples.

---

## Curved Geometry Experiment (`curved_geometry_qiskit.py`)

**Scientific Basis:**  
This experiment is designed to explore how quantum information and entanglement behave in systems that mimic *curved* versus *flat* spacetime geometries. The goal is to empirically probe the relationship between quantum entanglement and emergent geometry, providing insight into how spacetime curvature can arise from quantum information patterns—a key prediction of the holographic principle and modern quantum gravity theories.

**Methodology:**  
- Quantum circuits are constructed to simulate both flat and curved geometries using a 6-qubit system.
- For each geometry, the experiment sweeps a parameter (φ) that controls the circuit's entangling gates, effectively tuning the "curvature."
- The circuits are run on IBM Quantum hardware (or simulator), and measurement data is collected for each configuration.
- The experiment computes mutual information between all pairs of qubits, then uses multidimensional scaling (MDS) to embed the resulting "distance" matrix into 2D and 3D space, visualizing the emergent geometry.
- Additional metrics such as entanglement entropy, triangle angle sums, and Gaussian curvature (via the angle deficit method) are extracted to quantify geometric features.
- Results are compared between the flat and curved cases to reveal how curvature affects quantum information distribution and entanglement structure.

**Key Results:**  
- The experiment demonstrates that the pattern of quantum entanglement changes in a way that reflects the underlying geometry (flat vs. curved).
- Curvature can be directly visualized and quantified from the quantum data, providing empirical support for the idea that spacetime geometry can emerge from entanglement.

**Outputs:**  
- All results, plots (e.g., mutual information heatmaps), and geometric embeddings are saved in a timestamped folder under `experiment_logs/curved_geometry_qiskit_<device>_<timestamp>/`.
- A summary file explains the theoretical background, methodology, and key findings for each run.

---

## Tutorial: Analyzing Entanglement Geodesics

You can analyze the emergent geometric structure of your quantum experiments using the interactive script:

```
python src/analysis/analyze_entanglement_geodesic.py
```

This tool lets you compute and visualize the minimal "entanglement geodesic" (shortest path in entanglement distance) between any two subsystems, for any geometry and parameter setting in your experiment logs.

### **Step-by-Step Usage**
1. **Run the script:**
   ```
   python src/analysis/analyze_entanglement_geodesic.py
   ```
2. **Select the experiment log directory** (usually just enter `1` for the most recent run).
3. **Select geometry:** Enter `1` for flat or `2` for curved. For a full analysis, run the script for both.
4. **Select phi values:** Enter `all` to analyze all available φ (recommended), or specify a range/list (e.g., `1-3`, `1,3,5`).
5. **Subsystem pairs:**
   - Enter `all` to analyze all single-qubit pairs (recommended for a complete scan).
   - Or specify custom pairs, e.g., `0->5;1,2->3,4` (A->B;C->D).
6. **Plot each geodesic?** Enter `n` for summary only, or `y` to see a plot for each case.

### **What You Get**
- For each φ and subsystem pair, the script prints the minimal geodesic path and its length.
- At the end, a summary table shows all geodesic lengths for your selections.
- (Optional) Plots show the geodesic path in the emergent geometry.

### **Interpreting Results**
- **Shorter geodesics** indicate stronger entanglement and closer emergent geometry.
- **Longer or bent geodesics** (especially in the curved case) reflect the effect of curvature on quantum information flow.
- **Compare flat vs. curved** to see how curvature changes the entanglement structure.
- **Look for patterns** as φ varies—transitions may signal changes in the underlying geometry.

### **Best Practices for a Complete Picture**
- Run the script for both flat and curved geometries.
- Use `all` for φ and subsystem pairs to get a full matrix of geodesic distances.
- Optionally, visualize selected geodesics for deeper insight.

This analysis provides direct, empirical evidence for the geometric encoding of quantum information and supports the study of holography in your experiments.

---

# How to Run the Summary Geometry Analysis

After running the curved geometry experiment, you can perform a comprehensive analysis of the emergent geometry using the summary analysis tool. This script will:
- Plot geodesic deviation (Q3-Q4 length vs φ)
- Overlay MDS embeddings for all φ to visualize geometry evolution
- Plot mean Gaussian curvature vs φ
- Print a formal summary suitable for scientific write-ups

**To run the summary analysis:**
```
python src/analysis/summary_geometry_analysis.py
```

**What to expect:**
- Several plots will appear, showing geodesic deviation, geometry overlays, and curvature.
- A summary will be printed in the terminal, interpreting the results in the context of holographic duality and emergent spacetime.
- The script automatically uses the latest experiment log in `experiment_logs/`.

You can use this tool to quickly generate figures and text for presentations or papers based on your latest experiment results.


Simulating a Black Hole In my experiments, I model the black hole and radiation system as follows:

Black Hole as a Qubit:

A single qubit represents the black hole's quantum state, capturing the fundamental idea that black holes have a finite set of quantum properties (mass, charge, spin) encoded in their event horizon, in addition you can view this system as the Hawking radiation particle pairs split, they are entangled because of conservation laws. Another way to look at this is from the POV of the information theorist, and the radiation is just interaction through entanglement with the 2D surface of the holographic principle. In either case, for the sake of our computational model we will treat the black hole as a quantum object. I have always heard that black holes have "no hairs" in that charge, angular momentum, and mass are the only properties needed to describe a black hole. Which is about as many as you need to describe quantum objects esp in the context of this experiment, which is why we are fine using it. This is an introduction to the methods used to demonstrate emergent spacetime rather than a definitive claim about black holes.

Radiation as Qubits:

Additional qubits represent the emitted Hawking radiation, which entangle with the black hole qubit as the system evolves. 

Evaporation Process:

Radiation qubits are sequentially entangled with the black hole qubit to mimic the emission of Hawking radiation over time. 

Charge Injection:

I simulate charge injections into the black hole using Pauli gates (X for positive charge, Z for negative charge) to analyze whether these properties are encoded in the radiation.

Limitations of the Simulation While the quantum circuit captures certain aspects of black hole physics, there are critical limitations:

No Spatial Geometry:

The simulation does not account for the curved spacetime geometry near a black hole, which plays a significant role in Hawking radiation. 
Information Encoding Simplified:

The black hole qubit directly interacts with the radiation qubits, but the real-world mechanism for encoding information in Hawking radiation remains an open question.  This is ok since we are just using this as a basis for our theoretical understanding for the emergent curved spacetime from entanglement generation later on.

No Event Horizon Constraints:

In reality, the event horizon prevents signals from escaping a black hole. This simulation treats the black hole as a quantum system without such spatial constraints.

Idealized Dynamics:

The system assumes perfect unitary evolution and ignores potential complications from decoherence or quantum gravity effects. Despite these simplifications, the model aims to test whether key principles—such as information preservation and entanglement—can be explored in a controlled, quantum mechanical context.

Experimental Goals The primary questions driving this work are:

Information Encoding: Does the radiation qubits' state reflect the black hole's quantum properties, such as injected charge or spin? 

Static vs. Dynamic Behavior: Does the black hole encode information globally in a time-independent manner (static encoding), or does the radiation evolve dynamically based on the timing of interactions? 

Insights into the Information Paradox: Can the results offer new insights into how curved spacetime emerges?

Methodology Here's an overview of how the experiments are set up:

System Initialization:

The black hole qubit is initialized in a superposition state, entangled with radiation qubits via controlled gates (CX). 

Charge Injections:

I alternate between injecting positive (X gate) and negative (Z gate) charges into the black hole qubit, testing how these affect the radiation's quantum state. 

Evaporation Simulation:

Radiation qubits are sequentially entangled with the black hole qubit to mimic the gradual release of Hawking radiation. 

Measurements:

Radiation qubits are measured to analyze how information is encoded. Entanglement entropy and statevector evolution are tracked to study the distribution of information. 

Results 

Experiment A: Static Behavior Measurement: The radiation collapsed into a single dominant state (0000) with 100% probability. 

Implication: This suggests limited entanglement or static encoding, where the radiation doesn't reflect dynamic changes in the black hole's state. 

Experiment B: Dynamic Behavior Measurement: Outcomes showed a diverse distribution (0000, 0101, 1111, 1010) with nearly equal probabilities. 

Implication: This suggests time-dependent encoding, where radiation qubits retain memory of past charge injections. But I dont hear many people on both sides of the debate about static or dynamic, but experimentally this shows that literally just depends but you can get it to act like both. Sorry to anyone that was known to be hardcore in these camps. youre right, just hear the other person out too

Charge Preservation 

Across all experiments, the emitted radiation consistently encoded injected charge states, supporting the hypothesis that information is preserved. except when dealing with green color charge specifically (mistake, it is always preserved)

RECONSTRUCTS PAGE CURVE AND IF U JUST LOOK AT THE CHART ITS LOG NOT LIN SO GOOD EVIDENCE OF RYU-TAK(see AWSFactory)

Happy codes dont work on hardware yet so i cant compare to the existing literature. **However, the emergent spacetime demonstration has since been executed on real AWS Braket hardware—see `AWSFactory` for the latest results.** **Since has also been done on IBM Hardware**

Questions for the Community

Does this static vs dynamic element play a role into how the emergent spacetime was constructed uniquely by my charge injection method?

Are there improvements I could make to better simulate the entanglement and evaporation process? Static vs. Dynamic Encoding:

Do the results suggest new insights into whether information release is time-dependent or global? 

Implications for the Information Paradox:

Can these findings contribute to understanding how quantum information behaves under different circummstances?

Code https://github.com/NiloGregginz33/QMGRExperiments

I'd love feedback on:

Whether the methodology and results align with established physics, or if any of these experiments have been done before this on this topic. Any suggestions for refining the model or testing additional scenarios. 

Basically if you agree with all that stuff, then please see the result of every file basically overall ive seen charge info conserved, spin conserved, mass conserved. Everything that the holographic princple holds and even if you take it straight from Hawking. You will see Hawking Radiation, carry detectable electric charge, deviating from the idea that Hawking Radiation is purely thermal. Information is generally conserved in color charge. I wish this was all a simulation artifact but it definitely works on real quantum computers like the coded experiments that start w a Q mean. So theory or not something weird is happening, because its sooo close to what Leonard Susskind was saying about all information needing to be preserved.  And if its not sufficient, is the circuit I execute on IBMs Quantum Computer because i feel like thats physical, quantum, and also their service is free any maniac at home can access a quantum computer and have access to free real physical quantum qubits exhibiting behavior that we expect to see if black holes were emitting Hawking Radiation. This insight into black holes may be interesting but is secondary to the charge injection method's use for constructing emergent spacetimes.

RELEVANCE:
THIS EXPERIMENT IS NOT RANDOM< THIS CHARGE INJECTION METHOD PR GATE EQUIVILENT OPERATIONS ARE USED TO CREATE CURVED SPACETIMES

Sorry for doubting you Dr Susskind, ppl on stack exchange just said idk so I had to check if I could continue to use youtube for sources. 

So this basically means nothing new, a very good sign I think :)

and seems like yes, Leonard Susskind is not just talking theoretically anymore this means bc I just showed that not only does it work with the Hawking Radiation found on black holes millions of years far and into the future, but also rather on real physical quantum systems and states we can see and manipulate clearly on earth here and now that also exhibit this same behavior mimicking Hawking Radiation, as well as showed emergent geometry from entanglement accessible to all from here, and IBM Q computers stood out bc they are literally free access open source to their equipment so I figured why not share how to use the tools that are publically freely available in this economy.

WHY IT MATTERS:
SHOWS PHYSICAL EVIDENCE THAT PERFECTLY ALIGNS WITH THEORETICAL PREDICTIONS, IN FAVOR OF SUSSKINDS ARGUMENTS IN A WAY ANYONE WITH A LAPTOP CAN REPLICATE EASILY AT HOME


CONCLUSIONS
THE HOLOGRAPHIC PRINCPLE IS NOT JUST THEORETICAL 

Hit on the credebility of firewall paradox too kinda huh

Probably means Maldecena is right too


Maximal Shannon and Von Neumman entropy means time is an emergent property (I was suffering from a manic episode when I wrote this part, and some work got deleted after, so im keeping this in here but take it with a mountain of salt - Im leaving it in case I recover the evidence. I apologize the informaliy. I apologize and I hope my mental illness doesnt bring harsh or unfair judgement on the validity of my work, and I am taking responsibility of it.)

There are many tests between many worlds and holography and the data in the factory code shows phenomena predicted by only holography, specifically in HolographicFactory.py. I plan on releasing this as a interactive notebook as the results are not posted here yet.

FIRST DEMONSTRATION OF SPACETIME FROM ENTANGLEMENT
In my entanglement-Idge reconstruction experiments I used IBM's superconducting processors to emulate a one-dimensional "holographic" system of five qubits linked by controlled-phase gates. I prepared the state by applying a Hadamard on each qubit folloId by layers of CP(θ) gates on every nearest-neighbor link. Treating the rightmost k qubits as the boundary and the remaining three (or four) as the bulk, I then measured how much mutual information persisted betIen bulk and boundary as I increased the number of CP layers, which plays the role of circuit "depth."

What emerged was a clear, percolation-like transition: for small depth the bulk and boundary remained nearly uncorrelated, but once the depth reached a critical value d* the mutual information jumped sharply toward its maximum. By sIeping both the entangling angle θ and the boundary size k, I mapped out a family of curves d*(θ;n,k) and found they collapse onto a simple three-parameter form,

𝑑∗(𝜃;𝑛,𝑘)≈0.8453 (𝑛−𝑘)/(𝜃^3.1623)+1.7490.d(θ;n,k)≈0.8453 (n−k)/(θ^3.1623)+1.7490.

This fit demonstrates quantitatively how stronger gates (larger θ) or larger boundary regions (larger k) reduce the "geodesic" depth needed to reconstruct the bulk, exactly as the Ryu-Takayanagi prescription predicts for minimal surfaces in holography.

To confirm that this phenomenon truly derives from partial boundary access rather than trivial correlations, I ran two controls. First, I injected extra phase ("charge") on the boundary qubits before measuring, which predictably shifted the transition curve in agreement with charged-surface expectations. Second, performing a full measurement on all qubits eliminated any nontrivial jump, showing that the Idge signature vanishes when the boundary and bulk are not distinguished.

Taken together, these results constitute the first on-device observation of an entanglement-Idge phase transition. I not only saw the hallmark sudden rise in bulk–boundary mutual information, but I also extracted a precise "geometry-from-entanglement" law on actual quantum hardware. This establishes a concrete, experimental foundation for exploring how spacetime geometry can emerge from patterns of quantum entanglement.

Deviations from expected measurements have been observed, specifically seeing a deviation from δS=δ⟨Hξ⟩ such that there is a term δS=1.443δ⟨Hξ⟩ and I dont know where it is coming from check the QuantumAnalyzer outputs to see how and where I got those results.

Thought about that number 1.443 for a second, it is 1/ln2. this is because the hamiltonian is continuous whereas the von neumann entropy is from measuring base 2 bits. So after correction, this does in fact line up with theoretical predictions

So i feel like this now can explain to my girlfriend why we need to visit a black hole and find out

For Leonard Susskind to not be a theoretical physicist, we have to go to the black hole and see for ourselves And then hack it in the same way essentially, manipulating the time dependent electric charge visible as Hawking Radiation until we can model a qubit as a real black hole.

If you need more proof, check the `AWSFactory` directory. Those scripts visualize emergent spacetime on real hardware and include page‑curve extraction demos. The hardware runs show the expected Ryu–Takayanagi correlation and provide empirical support for holography.


# Quantum Information & Geometry Experiments: Scientific Summaries

## 1. Boundary vs. Bulk Entropy

**Theoretical Background:**  
This experiment tests the holographic principle by examining how entropy scales with the size of a boundary cut in a quantum system. The perfect tensor structure is expected to encode bulk information fully in the boundary, as predicted by AdS/CFT correspondence and string theory.

**Methodology:**  
- Constructed quantum circuits with varying boundary cut sizes.
- Measured entropy for each cut.
- Analyzed the scaling relationship and checked for perfect tensor structure.

**Key Results:**  
- Observed a linear relationship between entropy and cut size, consistent with the holographic principle.
- Perfect tensor structure confirmed: bulk information is mirrored in the boundary.
- Robustness of entropy values across cuts supports strong holographic encoding.

**Analysis & Interpretation:**  
- The results provide empirical support for the AdS/CFT correspondence.
- The entropy scaling and tensor structure align with string theory predictions.
- Suggests that emergent spacetime geometry is encoded in the entanglement structure of the boundary theory.

---

## 2. Bulk Reconstruction

**Theoretical Background:**  
Intended to probe the process of reconstructing bulk information from boundary data in a holographic quantum system.

**Methodology:**  
- (Not available in logs.)

**Key Results:**  
- (Not available in logs.)

**Analysis & Interpretation:**  
- This experiment's logs do not contain detailed theoretical or methodological information. For further details, consult the experiment's code and outputs.

---

## 3. CTC Geometry

**Theoretical Background:**  
Simulates a Closed Timelike Curve (CTC) geometry using a quantum circuit, investigating how feedback loops and time-like evolution affect entanglement and emergent geometry.

**Methodology:**  
- Constructed a 4-qubit quantum circuit with a feedback loop structure.
- Computed mutual information between qubits.
- Used Multidimensional Scaling (MDS) to visualize emergent geometry.

**Key Results:**  
- Demonstrated how CTC-like structures influence quantum entanglement and emergent geometry.

**Analysis & Interpretation:**  
- Provides insight into the interplay between quantum mechanics and spacetime structure, showing how feedback and time-like evolution can shape entanglement patterns.

---

## 4. Emergent Spacetime

**Theoretical Background:**  
Explores how quantum entanglement gives rise to emergent spacetime geometry, probing the relationship between quantum information and geometry.

**Methodology:**  
- Simulated a quantum circuit over multiple timesteps.
- Computed mutual information matrices.
- Used MDS to extract geometric features.
- Tracked entropy, curvature, and distance over time.

**Key Results:**  
- Demonstrated that quantum entanglement patterns can be mapped to emergent geometric structures.

**Analysis & Interpretation:**  
- Supports the idea that spacetime geometry is encoded in quantum information.
- Shows a direct mapping from entanglement structure to geometric features.

---

## 5. Emergent Spacetime (Simulator Variant)

**Theoretical Background:**  
Same as above, but run on a simulator backend.

**Methodology:**  
- Simulated quantum circuits with 100 shots.
- Tracked mutual information, entropy, curvature, and distance.

**Key Results:**  
- Results consistent with the main emergent spacetime experiment.

**Analysis & Interpretation:**  
- Confirms that the observed phenomena are robust to simulation and not artifacts of hardware noise.

---

## 6. Star Geometry

**Theoretical Background:**  
Studies multipartite entanglement in a star-shaped quantum circuit, aiming to understand how such geometries affect entanglement distribution.

**Methodology:**  
- Constructed a star-shaped circuit with 4 qubits.
- Measured output bitstrings and their frequencies.

**Key Results:**  
- Observed a diverse distribution of measurement outcomes, indicating complex multipartite entanglement.

**Analysis & Interpretation:**  
- Provides insight into how star-like geometries distribute entanglement, with implications for quantum networks and holographic models.

---

## 7. Page Curve

**Theoretical Background:**  
Explores the evolution of entanglement entropy in a quantum system, aiming to reproduce the Page curve—a key prediction for black hole evaporation and information retention.

**Methodology:**  
- Measured entropy at various phases of circuit evolution.

**Key Results:**  
- Entropy values fluctuate with phase, showing a pattern that matches the expected Page curve.

**Analysis & Interpretation:**  
- Successfully captures the dynamics of quantum entanglement over time.
- Provides evidence for the information retention predicted by the holographic principle.

---

## 8. Unified Mapping / Emergent Geometry

**Theoretical Background:**  
Investigates the mapping from quantum information to emergent geometry, seeking to uncover universal patterns in this relationship.

**Methodology:**  
- Simulated quantum circuits and computed entropies, curvatures, distances, and mutual information matrices over multiple steps.

**Key Results:**  
- Observed the emergence of geometric patterns from quantum information.
- Quantified the evolution of entropy, curvature, and distance.

**Analysis & Interpretation:**  
- Demonstrates a robust mapping from quantum information to geometry.
- Supports the hypothesis that spacetime structure is a manifestation of underlying quantum entanglement.

---

**[End of Summaries]**

---




https://www.academia.edu/126549379/Simulating_Hawking_Radiation_Quantum_Circuits_and_Information_Retention is the original paper associated with the work.
the newer paper is here 
https://zenodo.org/records/15686913?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImQ3MjMyYzA5LTcwMzgtNGNkMC05ODU5LWZjODhmZGExZGRjYyIsImRhdGEiOnt9LCJyYW5kb20iOiIwMjI5MWQ1ZDg4ZWZiOWYyNTMxZmY1OTVkMGVkZGY1MiJ9.UaE7kDBmwdUFf3vMvNwLGH4qAXnW2owUB7kLNgQBbvm3rK29EVhy_F2pkFl_rHVBw-_JcV4Idd2YVsIWD4vnWw

## Licensing and Usage Restrictions and UPDATE

This repository contains two types of code, each governed by different rules:

1. **Factory Code**:
   - Any file with "Factory" in its name (or earlier iterations derived from it) is **proprietary**.
   - Files in the 'Factory/' directory are **proprietary**
   - These files are included in this repository for **viewing purposes only**, and their use, modification, or distribution is strictly prohibited without prior written permission from the owner.
   - For permissions, licensing inquiries, or collaborations, please contact manavnaik123@gmail.com
  
###################################################################################################################################################################################
   - As of 2/10/2025 this project is no longer licensed under MIT. All rights are now reserved by Matrix Solutions LLC.
   - You may not use, copy, modify, or distribute this code without explicit written permission, unless for the purpose of peer reviewing the papers listed above.
   - If you were using this project under the MIT license before 2/10/2025, please contact me for continued access.
   - For licensing inquiries email manavnaik123@gmail.com


By accessing or cloning this repository, you agree to comply with these licensing terms.

## Disclaimer

This software is provided "as is" without any guarantees or warranties.

Failure to comply with the license will result in legal action. Any misuse of any technology or associated technology will be met with legal action. 
This project is proprietary and protectedf under applicable copyright laws.

