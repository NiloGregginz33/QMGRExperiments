# Experiment Logs Structure

All experiment outputs must be saved to the `experiment_logs/` directory at the project root. Each experiment run should create a new subdirectory named after the experiment and timestamp (e.g., `experiment_name_YYYYMMDD_HHMMSS/`).

## Each experiment log directory should contain:
- `summary.txt`: Plain text summary of the experiment
- `results.json`: Detailed results in JSON format
- Any relevant plots or data files (e.g., `.png` images)

This structure ensures all experiment outputs are organized, reproducible, and easy to review.

---

## Context from Previous Chats

- The `src/experiments/` folder contains experiment scripts.
- Experiment outputs are saved in `experiment_logs/` at the project root.
- Each experiment run creates a timestamped subdirectory in `experiment_logs/`.
- Typical files in each log directory: `summary.txt`, `results.json`, and plots/images.
- This structure was confirmed and requested by the user for consistency and reproducibility.
- The Memory Bank is used to document rules and project knowledge for future reference.
- If you need to update or expand this rule, refer to this README and the Memory Bank documentation.
