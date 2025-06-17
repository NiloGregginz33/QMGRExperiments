from experiment_logger import ExperimentLogger
import numpy as np
import matplotlib.pyplot as plt

def run_test_experiment():
    logger = ExperimentLogger("test_experiment")
    
    # Log theoretical background
    logger.log_theoretical_background("""
    This is a test experiment to verify the logging system.
    """)
    
    # Log methodology
    logger.log_methodology("""
    1. Generate random data
    2. Create a plot
    3. Save results
    """)
    
    # Generate some test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Log parameters
    logger.log_parameters({
        "num_points": 100,
        "x_range": [0, 10]
    })
    
    # Create and save plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Test Plot")
    logger.save_plot(fig, "test_plot")
    
    # Log metrics
    logger.log_metrics({
        "mean": float(np.mean(y)),
        "std": float(np.std(y))
    })
    
    # Log analysis
    logger.log_analysis("""
    The test data shows a sine wave pattern.
    """)
    
    # Log interpretation
    logger.log_interpretation("""
    This test confirms that the logging system is working correctly.
    """)

if __name__ == "__main__":
    run_test_experiment() 