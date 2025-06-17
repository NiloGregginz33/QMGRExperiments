import os
import json
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, experiment_name):
        """Initialize the experiment logger with a specific experiment name."""
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("experiment_outputs", f"{experiment_name}_{self.timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler for detailed logs
        fh = logging.FileHandler(os.path.join(self.log_dir, f"{experiment_name}_log.txt"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Initialize documentation sections
        self.documentation = {
            "theoretical_background": "",
            "methodology": "",
            "parameters": {},
            "metrics": {},
            "analysis": "",
            "interpretation": ""
        }

    def log_theoretical_background(self, background):
        self.documentation["theoretical_background"] = background
        self.logger.info(f"Theoretical Background:\n{background}")

    def log_methodology(self, methodology):
        self.documentation["methodology"] = methodology
        self.logger.info(f"Methodology:\n{methodology}")

    def log_parameters(self, parameters):
        self.documentation["parameters"].update(parameters)
        self.logger.info(f"Parameters: {json.dumps(parameters, indent=2)}")

    def log_metrics(self, metrics):
        self.documentation["metrics"].update(metrics)
        self.logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    def log_analysis(self, analysis):
        self.documentation["analysis"] = analysis
        self.logger.info(f"Analysis:\n{analysis}")

    def log_interpretation(self, interpretation):
        self.documentation["interpretation"] = interpretation
        self.logger.info(f"Interpretation:\n{interpretation}")

    def save_plot(self, fig, name):
        fig.savefig(os.path.join(self.log_dir, f"{name}.png"))
        plt.close(fig)

    def save_documentation(self):
        with open(os.path.join(self.log_dir, "documentation.json"), 'w') as f:
            json.dump(self.documentation, f, indent=2)

    def save_raw_data(self, data, name):
        np.save(os.path.join(self.log_dir, f"{name}.npy"), data)

    def __del__(self):
        self.save_documentation() 