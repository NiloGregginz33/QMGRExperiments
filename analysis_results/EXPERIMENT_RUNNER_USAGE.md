# Experiment Runner Usage Guide

This document explains how to use the dynamic experiment runner script `run_experiment.py` to easily run any experiment in the codebase.

---

## Overview

The script `run_experiment.py` allows you to:
- Dynamically list all available experiments in `src/experiments/`
- Select and run any experiment or sub-experiment (including those in `run_simple_experiments.py`)
- Choose the quantum device (simulator or real hardware)
- Set the number of shots
- Use either an interactive menu or command-line arguments

---

## How to Use

### 1. Interactive Mode (Recommended)

Simply run:

```bash
python run_experiment.py
```

You will see a numbered list of all available experiments and sub-experiments, for example:

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

You will then be prompted to select the experiment by number. The script will run the selected experiment with default device (`simulator`) and shots (`1024`).

---

### 2. Command-Line Mode

You can also specify the experiment number and options directly:

```bash
python run_experiment.py --experiment 2 --device ionq --shots 500
```

- `--experiment N` : Run experiment number N (see the list printed by the script)
- `--device DEVICE` : Choose the quantum device (`simulator`, `ionq`, `rigetti`, `oqc`)
- `--shots N` : Number of shots (default: 1024)

---

## Options

| Option         | Description                                              | Default      |
| --------------|---------------------------------------------------------|--------------|
| --experiment  | Experiment number (see list)                             | (interactive)|
| --device      | Quantum device: `simulator`, `ionq`, `rigetti`, `oqc`   | simulator    |
| --shots       | Number of shots for quantum measurements                | 1024         |

---

## Examples

- **Run interactively:**
  ```bash
  python run_experiment.py
  ```

- **Run the 3rd experiment on the Rigetti device with 500 shots:**
  ```bash
  python run_experiment.py --experiment 3 --device rigetti --shots 500
  ```

- **Run the "Simple: Temporal" sub-experiment (see its number in the list):**
  ```bash
  python run_experiment.py --experiment 7
  ```

---

## Output

- All results, logs, and plots are saved in the `experiment_logs/` directory, organized by experiment and device.
- The script prints a summary and any errors to the console.

---

## Adding New Experiments

- Any new `.py` file added to `src/experiments/` (except those starting with `run_` or named `__init__.py`/`experiment_logger.py`) will automatically appear in the menu.
- Sub-experiments in `run_simple_experiments.py` are always included.

---

## Troubleshooting

- If you see "Invalid choice.", make sure you enter a valid experiment number.
- For device errors, check your AWS credentials and device availability.
- For more details, check the logs in `experiment_logs/`.

---

## Summary

The dynamic experiment runner makes it easy to:
- List and run any experiment in the codebase
- Choose hardware and configuration
- Scale as new experiments are added

For advanced batch runs, see also `run_all_experiments.py` and `run_hardware_experiments.py`. 