# Custom Curvature Experiment Error Analysis

This directory contains comprehensive error analysis tools for the custom curvature experiment, designed to add error bars to all graphs and provide statistical analysis of uncertainties.

## Files

### Core Analysis Scripts

1. **`custom_curvature_error_analysis.py`** - Main error analysis class
   - Comprehensive error analysis for custom curvature experiments
   - Bootstrap error estimation for mutual information
   - Shot noise analysis
   - Statistical significance testing
   - Error propagation for derived quantities
   - Confidence intervals for all measurements
   - Enhanced visualizations with error bars

2. **`run_error_analysis.py`** - Simple runner script
   - Easy-to-use script to run error analysis on any results file
   - Command-line interface for quick analysis

## Features

### Error Analysis Capabilities

- **Bootstrap Error Estimation**: Uses 1000 bootstrap samples for robust error estimation
- **Shot Noise Analysis**: Calculates standard errors based on shot counts
- **Statistical Significance Testing**: Linear trend analysis and confidence intervals
- **Error Propagation**: Handles uncertainty in derived quantities
- **Confidence Intervals**: 95% confidence intervals for all measurements
- **Enhanced Visualizations**: All plots include error bars and confidence bands

### Generated Outputs

For each analysis, the following files are generated:

1. **`error_analysis_summary.txt`** - Comprehensive text report
   - Experiment parameters
   - Error analysis parameters
   - Mutual information analysis with statistics
   - Entropy analysis with error estimates
   - Statistical significance tests
   - Error sources and recommendations

2. **`error_analysis_results.json`** - Structured data output
   - All analysis results in JSON format
   - Raw data for further processing
   - Error estimates and confidence intervals

3. **`mi_evolution_with_errors.png`** - Mutual information plot
   - MI evolution over timesteps with error bars
   - Confidence interval shading
   - Statistical summary annotations

4. **`entropy_evolution_with_errors.png`** - Entropy plot
   - Entropy evolution over timesteps with error bars
   - Confidence interval shading
   - Statistical summary annotations

5. **`mi_distance_correlation_with_errors.png`** - Correlation plot
   - MI vs Distance correlation with error bars
   - Linear fit with R² and p-value
   - Error-weighted analysis

## Usage

### Quick Start

```bash
# Run error analysis on the latest experiment
python analysis/custom_curvature_error_analysis.py

# Run error analysis on a specific results file
python analysis/run_error_analysis.py path/to/results.json
```

### Example Commands

```bash
# Analyze a specific hardware run
python analysis/run_error_analysis.py experiment_logs/custom_curvature_experiment/older_results/results_n7_geomH_curv5_ibm_brisbane_DUXTVJ.json

# Analyze a simulator run
python analysis/run_error_analysis.py experiment_logs/custom_curvature_experiment/instance_20250726_185317/results_n3_geomS_curv4_simulator_DI2DQ6.json
```

### Programmatic Usage

```python
from analysis.custom_curvature_error_analysis import CustomCurvatureErrorAnalyzer

# Create analyzer
analyzer = CustomCurvatureErrorAnalyzer("path/to/results.json")

# Run complete analysis
analysis_data = analyzer.run_complete_analysis()

# Access specific analyses
mi_analysis = analyzer.analyze_mi_errors()
entropy_analysis = analyzer.analyze_entropy_errors()
distance_analysis = analyzer.analyze_distance_errors()
```

## Error Analysis Methodology

### Bootstrap Error Estimation

The analysis uses bootstrap resampling to estimate uncertainties:

1. **Resampling**: Creates 1000 bootstrap samples from the original data
2. **Statistical Analysis**: Calculates mean and standard deviation for each quantity
3. **Confidence Intervals**: Uses **percentile-based 95% confidence intervals** (more robust than normal approximation)
4. **Error Propagation**: Handles uncertainty in derived quantities
5. **Physical Constraint Enforcement**: Ensures entropy ≥ 0 and other theoretical bounds

### Physical Constraint Enforcement

**Critical Fix for Negative Entropy Values:**

- **Problem**: Traditional symmetric confidence intervals can produce negative lower bounds for entropy
- **Solution**: **Percentile-based bootstrap CIs** with physical constraint enforcement
- **Methodology**: 
  - Bootstrap samples are constrained to entropy ≥ 0 during generation
  - Confidence intervals use 2.5th and 97.5th percentiles (for 95% CI)
  - Lower bounds are clamped to 0.0 when necessary
  - Truncation events are documented and reported
- **Statistical Justification**: Negative bounds are artifacts of finite sampling, not physical reality
- **Impact**: Ensures all confidence intervals respect theoretical constraints

### Shot Noise Analysis

For quantum measurements, shot noise is a fundamental source of error:

- **Standard Error**: σ = √(p(1-p)/N) where p is probability and N is shot count
- **Recommendation**: Increase shot count to reduce shot noise
- **Current Analysis**: Uses 2000 shots per measurement

### Statistical Significance Testing

The analysis includes several statistical tests:

1. **Linear Trend Test**: Tests for significant trends over time
2. **R² Analysis**: Measures goodness of fit for correlations
3. **P-value Testing**: Determines statistical significance (α = 0.05)
4. **Coefficient of Variation**: Measures relative variability

## Error Sources and Recommendations

### Primary Error Sources

1. **Shot Noise**: Fundamental quantum measurement uncertainty
2. **Bootstrap Sampling**: Statistical uncertainty in error estimation
3. **Quantum Decoherence**: Hardware-specific errors from IBM devices
4. **Statistical Fluctuations**: Natural variation in quantum measurements

### Recommendations

1. **Increase Shot Count**: More shots reduce shot noise (current: 2000)
2. **Multiple Runs**: Consider averaging over multiple experiment runs
3. **Error Mitigation**: Apply quantum error correction techniques
4. **Calibration**: Regular device calibration for consistent results

## Output Interpretation

### Mutual Information Analysis

- **Mean MI**: Average mutual information across all timesteps
- **MI Error**: Standard deviation of MI values
- **MI Range**: [min, max] values across timesteps
- **Coefficient of Variation**: Relative variability (std/mean)

### Entropy Analysis

- **Mean Entropy**: Average entropy across all timesteps
- **Entropy Error**: Standard deviation of entropy values
- **Entropy Range**: [min, max] values across timesteps
- **Coefficient of Variation**: Relative variability

### Statistical Significance

- **Linear Trend**: Tests if MI/entropy changes systematically over time
- **R² Value**: Measures how well data fits a linear trend
- **P-value**: Probability of observing the trend by chance
- **Significance**: Whether the trend is statistically significant (p < 0.05)

## Example Results

### Sample Analysis Output

```
# Custom Curvature Experiment Error Analysis Report

## Experiment Parameters
- Number of qubits: 7
- Geometry: hyperbolic
- Curvature: 5.0
- Device: ibm_brisbane
- Shots: 2000
- Timesteps: 8

## Mutual Information Analysis
- Mean MI across timesteps: 0.100000 ± 0.000000
- Average MI error: 0.000000
- MI range: [0.100000, 0.100000]
- Coefficient of variation: 0.000

## Entropy Analysis
- Mean entropy across timesteps: 0.254375 ± 0.254420
- Average entropy error: 0.249939
- Entropy range: [0.000000, 0.520000]
- Coefficient of variation: 1.000

## Statistical Significance Tests
- Linear trend test: slope = 0.000000 ± 0.000000
- R² = 0.000, p-value = 1.000
- Trend significance: Not significant (α = 0.05)
```

## Troubleshooting

### Common Issues

1. **No counts data**: Some simulator runs don't include counts data
   - Solution: Use existing mutual information data instead
   - The script automatically falls back to available data

2. **Large file sizes**: Hardware runs can be very large
   - Solution: The script automatically selects the largest file for analysis
   - Consider using specific file paths for targeted analysis

3. **Unicode errors**: Windows encoding issues
   - Solution: All files are saved with UTF-8 encoding
   - The script handles encoding automatically

### Performance Tips

1. **Bootstrap samples**: Reduce `n_bootstrap` for faster analysis
2. **File selection**: Use specific file paths to avoid searching
3. **Output directory**: Specify custom output directory for organization

## Integration with Other Tools

The error analysis tools integrate with the existing analysis pipeline:

- **Input**: Standard custom curvature experiment results JSON files
- **Output**: Enhanced plots with error bars for publication
- **Data**: Structured JSON output for further analysis
- **Reports**: Comprehensive text summaries for documentation

## Future Enhancements

Planned improvements to the error analysis tools:

1. **Advanced Error Models**: More sophisticated quantum error models
2. **Cross-validation**: Validation of error estimates
3. **Automated Reporting**: Integration with experiment logging
4. **Real-time Analysis**: Live error analysis during experiments
5. **Comparative Analysis**: Compare errors across different experiments 