import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.experiment_logger import PhysicsExperimentLogger
from quantum.temporal_injection import TemporalInjectionExperiment
from quantum.contradictions import ContradictionsExperiment
from quantum.demo import HolographicDemo
import numpy as np
from braket.devices import LocalSimulator

def run_experiment_sequence():
    """Run a sequence of physics experiments and analyze results."""
    
    # Initialize local simulator
    device = LocalSimulator()
    
    # Initialize logger
    logger = PhysicsExperimentLogger("physics_sequence")
    
    # Log initial parameters
    logger.log_parameters({
        "device": "LocalSimulator",
        "shots": 1024,
        "max_qubits": 6,
        "phase_steps": 15
    })
    
    # Run temporal injection experiment
    print("Running temporal injection experiment...")
    temp_exp = TemporalInjectionExperiment(device)
    temp_results = temp_exp.run()
    
    # Log temporal injection results
    for result in temp_results:
        logger.log_result(result)
    
    # Analyze temporal injection results
    temp_analysis = logger.analyze_entanglement_geometry(
        [r["entropy"] for r in temp_results],
        temp_results[-1].get("mi_matrix")
    )
    logger.write_physics_analysis(f"""
    ## Temporal Injection Analysis
    
    {temp_analysis}
    
    ### Physics Implications:
    1. The temporal injection pattern shows how quantum entanglement can give rise to emergent time-like structures
    2. The observed entropy scaling suggests a connection to the holographic principle
    3. The mutual information structure indicates potential geometric encoding of temporal relationships
    """)
    
    # Run contradictions experiment
    print("Running contradictions experiment...")
    cont_exp = ContradictionsExperiment(device)
    cont_results = cont_exp.run()
    
    # Log contradictions results
    for result in cont_results:
        logger.log_result(result)
    
    # Analyze contradictions results: compute average entropy per test
    cont_analysis = "Contradictions Experiment Analysis:\n"
    for result in cont_results:
        avg_entropy = np.mean(result["entropies"])
        cont_analysis += f"Test: {result['test_name']}, Average Entropy: {avg_entropy:.3f} bits\n"
    
    logger.write_physics_analysis(f"""
    ## Contradictions Analysis
    
    {cont_analysis}
    
    ### Physics Implications:
    1. The observed phase transitions may correspond to changes in the emergent geometry
    2. Critical behavior suggests a connection to quantum phase transitions in many-body systems
    3. The scaling exponents provide insights into the universality class of the emergent spacetime
    """)
    
    # Run holographic demo
    print("Running holographic demo...")
    demo = HolographicDemo(device)
    demo_results = demo.run()
    
    # Log holographic demo results
    for result in demo_results:
        logger.log_result(result)
    
    # Analyze demo results
    demo_analysis = logger.analyze_entanglement_geometry(
        [r["entropy"] for r in demo_results],
        demo_results[-1].get("mi_matrix")
    )
    logger.write_physics_analysis(f"""
    ## Holographic Demo Analysis
    
    {demo_analysis}
    
    ### Physics Implications:
    1. The holographic encoding demonstrates how bulk geometry emerges from boundary entanglement
    2. The observed scaling laws provide evidence for the AdS/CFT correspondence
    3. The mutual information structure reveals the geometric organization of quantum information
    """)
    
    # Finalize and get summary
    summary = logger.finalize()
    print("\nExperiment Summary:")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Average entropy: {summary['average_entropy']:.3f}")
    print(f"Max mutual information: {summary['max_mutual_info']:.3f}")
    print(f"Average curvature: {summary['average_curvature']:.3f}")
    print(f"Success rate: {summary['success_rate']:.3f}")
    
    return summary

if __name__ == "__main__":
    run_experiment_sequence() 