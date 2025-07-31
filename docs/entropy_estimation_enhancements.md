# Enhanced Entropy Estimation for Real Hardware Page Curve Observation

## Overview

This document describes the enhanced entropy estimation capabilities added to the custom curvature experiment, enabling robust Page curve observation on real quantum hardware using classical shadows and randomized measurements.

## Key Enhancements

### 1. Advanced Entropy Estimation Methods

The experiment now supports four entropy estimation methods:

#### Basic Method (`--entropy_method basic`)
- Standard measurement-based entropy computation
- Fastest method, suitable for simulators
- Limited accuracy on noisy hardware

#### Classical Shadow Tomography (`--entropy_method shadow`)
- Uses random Clifford circuits to estimate quantum state
- Robust against hardware noise
- Provides confidence intervals and error estimates
- Configurable with `--num_shadows` and `--shots_per_shadow`

#### Randomized Measurements (`--entropy_method random`)
- Measures in random bases to estimate entropy
- Good for characterizing subsystem properties
- Hardware-optimized circuit transpilation
- Configurable with `--num_bases` and `--shots_per_basis`

#### Hybrid Method (`--entropy_method hybrid`)
- Combines shadow tomography and randomized measurements
- Weighted average for improved accuracy
- Fallback to individual methods if one fails
- Most robust for real hardware

### 2. Hardware Optimization Features

#### Adaptive Shot Allocation
- Automatically adjusts shots based on system size
- Reduces shots for large systems to prevent timeouts
- Increases shots for small systems for better accuracy

#### Hardware-Aware Transpilation
- Uses backend-specific optimization
- Conservative optimization levels for shadow circuits
- Fallback transpilation if hardware optimization fails

#### Error Mitigation Integration
- Bootstrap sampling for confidence intervals
- Statistical error analysis
- Relative error estimation

### 3. Enhanced Page Curve Analysis

#### Comprehensive Data Collection
- Entropy estimates with confidence intervals
- Method-specific metadata
- Error analysis and statistics
- Hardware backend information

#### Advanced Visualization
- Error bars on Page curve plots
- Confidence interval visualization
- Method usage distribution
- Error analysis summary

## Usage Examples

### Basic Page Curve with Classical Shadows
```bash
python src/experiments/custom_curvature_experiment.py \
    --num_qubits 4 \
    --device ibm_brisbane \
    --page_curve \
    --entropy_method shadow \
    --num_shadows 50 \
    --shots_per_shadow 500
```

### Advanced Page Curve with Hybrid Method
```bash
python src/experiments/custom_curvature_experiment.py \
    --num_qubits 6 \
    --device ibm_brisbane \
    --page_curve \
    --entropy_method hybrid \
    --num_shadows 100 \
    --shots_per_shadow 500 \
    --num_bases 20 \
    --shots_per_basis 500 \
    --page_curve_timesteps 10
```

### Custom Radiation Ordering
```bash
python src/experiments/custom_curvature_experiment.py \
    --num_qubits 5 \
    --device ibm_brisbane \
    --page_curve \
    --entropy_method random \
    --radiation_ordering "0,1,2,3,4" \
    --num_bases 15 \
    --shots_per_basis 1000
```

## Command Line Arguments

### Entropy Estimation Parameters
- `--entropy_method`: Choose method (`basic`, `shadow`, `random`, `hybrid`)
- `--num_shadows`: Number of shadow samples (default: 50)
- `--shots_per_shadow`: Shots per shadow measurement (default: 500)
- `--num_bases`: Number of random measurement bases (default: 10)
- `--shots_per_basis`: Shots per random basis measurement (default: 500)

### Page Curve Parameters
- `--page_curve`: Enable Page curve analysis
- `--page_curve_timesteps`: Number of evaporation steps (default: 10)
- `--radiation_ordering`: Custom qubit evaporation sequence

## Output Files

### Enhanced Results JSON
The experiment now saves comprehensive entropy estimation data:

```json
{
  "page_curve_data": {
    "timesteps": [...],
    "radiation_entropies": [...],
    "radiation_entropy_metadata": [
      {
        "entropy": 1.234,
        "confidence_interval": [1.100, 1.368],
        "std_error": 0.068,
        "relative_error": 0.055,
        "method": "shadow",
        "success": true,
        "backend_name": "ibm_brisbane",
        "circuit_depth": 15
      }
    ]
  },
  "error_analysis": {
    "average_std_error": 0.045,
    "average_relative_error": 0.038,
    "method_distribution": {"shadow": 8, "random": 2},
    "successful_estimates": 10,
    "total_estimates": 10
  }
}
```

### Enhanced Plots
- **Page curve with error bars**: Shows entropy evolution with statistical uncertainty
- **Confidence intervals**: Visualizes 95% confidence intervals
- **Method distribution**: Shows which entropy estimation methods were used
- **Error analysis**: Displays average errors and method reliability

## Technical Details

### Classical Shadow Implementation
1. **Random Clifford Generation**: Creates random Clifford circuits for state tomography
2. **Hardware Optimization**: Transpiles circuits for specific backend constraints
3. **Bootstrap Sampling**: Uses resampling to estimate confidence intervals
4. **Density Matrix Reconstruction**: Estimates reduced density matrices for entropy calculation

### Randomized Measurement Implementation
1. **Random Basis Generation**: Creates random rotation circuits
2. **Adaptive Shot Allocation**: Adjusts shots based on subsystem size
3. **Statistical Analysis**: Computes entropy statistics across measurement bases
4. **Error Estimation**: Provides standard errors and confidence intervals

### Error Analysis
- **Bootstrap Confidence Intervals**: 95% confidence intervals using resampling
- **Standard Error Estimation**: Statistical uncertainty in entropy estimates
- **Relative Error Calculation**: Error relative to entropy magnitude
- **Method Reliability Assessment**: Tracks success rates of different methods

## Best Practices

### For Real Hardware
1. **Use hybrid method** for maximum robustness
2. **Start with moderate parameters** (50 shadows, 10 bases)
3. **Monitor error rates** and adjust parameters accordingly
4. **Use smaller systems** (4-6 qubits) for initial testing

### For Simulators
1. **Use basic method** for fastest execution
2. **Increase parameters** for higher accuracy
3. **Test with larger systems** to validate scalability

### Parameter Tuning
- **Increase `num_shadows`** for better accuracy (trade-off: longer runtime)
- **Increase `shots_per_shadow`** for lower statistical noise
- **Increase `num_bases`** for more robust randomized measurements
- **Use `radiation_ordering`** to control evaporation sequence

## Troubleshooting

### Common Issues
1. **High error rates**: Reduce system size or increase shots
2. **Timeouts**: Reduce number of shadows/bases or use smaller systems
3. **Failed estimates**: Check hardware availability and try hybrid method
4. **Poor convergence**: Increase parameters or use different entropy method

### Performance Optimization
1. **Adaptive parameters**: Let the system automatically adjust shots
2. **Hardware-specific settings**: Use backend-optimized transpilation
3. **Parallel execution**: Run multiple experiments simultaneously
4. **Caching**: Reuse shadow circuits when possible

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: Use ML to predict optimal parameters
2. **Advanced Error Mitigation**: Implement zero-noise extrapolation
3. **Real-time Monitoring**: Live error rate tracking during execution
4. **Automated Parameter Tuning**: Self-optimizing parameter selection

### Research Applications
1. **Quantum Error Correction**: Study entropy evolution in error-corrected systems
2. **Quantum Phase Transitions**: Observe entropy scaling near critical points
3. **Quantum Many-body Physics**: Analyze entanglement structure in complex systems
4. **Quantum Information Theory**: Test fundamental entropy bounds and relations

## Conclusion

The enhanced entropy estimation capabilities enable robust Page curve observation on real quantum hardware, providing:

- **Accurate entropy estimates** with quantified uncertainty
- **Hardware-optimized execution** for real quantum devices
- **Comprehensive error analysis** for scientific validation
- **Flexible parameter control** for different experimental needs

This implementation bridges the gap between theoretical Page curve predictions and experimental observation, enabling new insights into quantum gravity and holographic duality using current quantum hardware. 