import json
import logging
import datetime
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_dir="experiment_logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Set up logging
        self.logger = logging.getLogger('quantum_experiments')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(f'{log_dir}/experiment_{timestamp}.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Results storage
        self.results = {}
        
    def log_experiment(self, experiment_name: str, results: Dict[str, Any], theory_comparison: Dict[str, Any]):
        """Log experiment results and comparison with theory"""
        self.logger.info(f"\n{'='*50}\nExperiment: {experiment_name}\n{'='*50}")
        
        # Log results
        self.logger.info("\nResults:")
        for key, value in results.items():
            self.logger.info(f"{key}: {value}")
            
        # Log theory comparison
        self.logger.info("\nTheory Comparison:")
        for key, value in theory_comparison.items():
            self.logger.info(f"{key}: {value}")
            
        # Store results
        self.results[experiment_name] = {
            'results': results,
            'theory_comparison': theory_comparison,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save to JSON
        with open(f'{self.log_dir}/results_{experiment_name}.json', 'w') as f:
            json.dump(self.results[experiment_name], f, indent=2)
            
    def log_implication(self, experiment_name: str, implication: str):
        """Log implications of experiment results"""
        self.logger.info(f"\nImplications for {experiment_name}:")
        self.logger.info(implication)
        
        # Add to results
        if experiment_name in self.results:
            self.results[experiment_name]['implications'] = implication
            with open(f'{self.log_dir}/results_{experiment_name}.json', 'w') as f:
                json.dump(self.results[experiment_name], f, indent=2)

class PhysicsExperimentLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results = []
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"experiment_logs/{experiment_name}_{self.timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        
    def log_result(self, result):
        """Log a single experiment result."""
        # Convert numpy arrays to lists for JSON serialization
        processed_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                processed_result[key] = value.tolist()
            else:
                processed_result[key] = value
                
        self.results.append(processed_result)
        
        # Save individual result
        result_file = os.path.join(self.log_dir, f"result_{len(self.results)}.json")
        with open(result_file, 'w') as f:
            json.dump(processed_result, f, indent=2)
            
    def finalize(self):
        """Finalize the experiment by computing averages and saving summary."""
        if not self.results:
            return
            
        # Compute averages
        avg_mi = np.mean([np.mean(r["mi_matrix"]) for r in self.results])
        avg_curvature = np.mean([r["curvature"] for r in self.results])
        
        # Create summary
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "num_results": len(self.results),
            "patterns": list(set(r["pattern"] for r in self.results)),
            "num_injections": list(set(r["num_injections"] for r in self.results)),
            "gap_cycles": list(set(r["gap_cycles"] for r in self.results)),
            "average_mi": float(avg_mi),
            "average_curvature": float(avg_curvature)
        }
        
        # Save summary as JSON
        summary_file = os.path.join(self.log_dir, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Generate markdown analysis
        self._generate_markdown_analysis(summary)
        
    def _generate_markdown_analysis(self, summary):
        """Generate a markdown file with analysis of the results."""
        md_file = os.path.join(self.log_dir, "analysis.md")
        
        with open(md_file, 'w') as f:
            f.write(f"# {self.experiment_name} Analysis\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("## Experiment Parameters\n")
            f.write(f"- Number of Results: {summary['num_results']}\n")
            f.write(f"- Patterns Tested: {', '.join(summary['patterns'])}\n")
            f.write(f"- Number of Injections: {summary['num_injections']}\n")
            f.write(f"- Gap Cycles: {summary['gap_cycles']}\n\n")
            
            f.write("## Results Summary\n")
            f.write(f"- Average Mutual Information: {summary['average_mi']:.4f}\n")
            f.write(f"- Average Curvature: {summary['average_curvature']:.4f}\n\n")
            
            f.write("## Analysis\n")
            f.write("### Mutual Information\n")
            if summary['average_mi'] > 0.1:
                f.write("- Strong quantum correlations observed between qubits\n")
                f.write("- Indicates significant entanglement in the system\n")
            else:
                f.write("- Weak quantum correlations between qubits\n")
                f.write("- System may need stronger entangling operations\n")
                
            f.write("\n### Curvature\n")
            if abs(summary['average_curvature']) > 0.1:
                f.write("- Significant curvature detected in emergent geometry\n")
                f.write("- Suggests non-trivial spacetime structure\n")
            else:
                f.write("- Near-flat emergent geometry\n")
                f.write("- May indicate need for stronger non-local interactions\n")
                
            f.write("\n### Recommendations\n")
            if summary['average_mi'] < 0.1:
                f.write("1. Increase number of entangling operations\n")
                f.write("2. Add non-local interactions (e.g., CZ gates between non-adjacent qubits)\n")
                f.write("3. Consider stronger charge injection patterns\n")
            elif abs(summary['average_curvature']) < 0.1:
                f.write("1. Add more complex entanglement patterns\n")
                f.write("2. Implement measurement-based feedback\n")
                f.write("3. Increase number of qubits for better geometry resolution\n")

    def log_parameters(self, params):
        """Log experiment parameters."""
        self.results["parameters"] = params
        with open(f"{self.log_dir}/results.json", 'w') as f:
            json.dump(self.results, f, indent=2)

    def write_physics_analysis(self, analysis):
        """Write physics analysis to markdown file."""
        with open(f"{self.log_dir}/physics_analysis.md", 'a') as f:
            f.write(f"## Analysis for {self.experiment_name}\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            f.write(analysis)
            f.write("\n\n---\n\n")

    def analyze_entanglement_geometry(self, entropy_data, mi_matrix):
        """Analyze entanglement and emergent geometry."""
        analysis = []
        
        # Analyze entropy scaling
        if len(entropy_data) > 1:
            scaling = np.polyfit(np.log(range(1, len(entropy_data) + 1)), entropy_data, 1)
            analysis.append(f"Entropy scaling exponent: {scaling[0]:.3f}")
            
            # Compare to theoretical predictions
            if abs(scaling[0] - 1) < 0.1:
                analysis.append("Entropy follows area law (S ∝ A)")
            elif abs(scaling[0] - 2) < 0.1:
                analysis.append("Entropy follows volume law (S ∝ V)")
        
        # Analyze mutual information structure
        if mi_matrix is not None:
            # Check for long-range correlations
            avg_mi = np.mean(mi_matrix)
            max_mi = np.max(mi_matrix)
            analysis.append(f"Average mutual information: {avg_mi:.3f}")
            analysis.append(f"Maximum mutual information: {max_mi:.3f}")
            
            # Check for geometric structure
            if max_mi > 0.5:
                analysis.append("Strong long-range correlations detected")
                analysis.append("Consistent with emergent geometric structure")
        
        return "\n".join(analysis)

    def analyze_phase_transitions(self, phase_data, order_parameter):
        """Analyze phase transitions and critical behavior."""
        analysis = []
        
        if len(phase_data) > 1:
            # Look for discontinuities in order parameter
            diff = np.diff(order_parameter)
            if np.any(np.abs(diff) > np.std(order_parameter)):
                analysis.append("Phase transition detected")
                
                # Estimate critical point
                crit_idx = np.argmax(np.abs(diff))
                crit_phase = phase_data[crit_idx]
                analysis.append(f"Estimated critical point at phase = {crit_phase:.3f}")
                
                # Analyze scaling behavior
                if len(order_parameter) > 2:
                    scaling = np.polyfit(np.log(np.abs(phase_data - crit_phase)), 
                                       np.log(np.abs(order_parameter)), 1)
                    analysis.append(f"Critical exponent: {scaling[0]:.3f}")
        
        return "\n".join(analysis) 