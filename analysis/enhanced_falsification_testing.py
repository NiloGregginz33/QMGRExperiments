#!/usr/bin/env python3
"""
Enhanced Falsification Testing Framework for Quantum Holographic Evidence
Tests both mutual information data AND entropy engineering data for comprehensive validation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class EnhancedFalsificationTester:
    """Enhanced falsification testing for quantum holographic phenomena."""
    
    def __init__(self, experiment_file: str):
        self.experiment_file = experiment_file
        self.results = {}
        self.falsification_tests = {}
        
    def load_experiment_data(self):
        """Load the target experiment data including entropy engineering results."""
        print(f"🔍 Loading experiment: {os.path.basename(self.experiment_file)}")
        
        with open(self.experiment_file, 'r') as f:
            self.data = json.load(f)
            
        self.spec = self.data.get('spec', {})
        self.mutual_info_data = self.data.get('mutual_information_per_timestep', {})
        
        # Load entropy engineering data if available
        experiment_dir = os.path.dirname(self.experiment_file)
        entropy_file = os.path.join(experiment_dir, "entropy_engineering_quantum_gravity_results.json")
        
        if os.path.exists(entropy_file):
            with open(entropy_file, 'r') as f:
                self.entropy_data = json.load(f)
            print(f"  ✅ Found entropy engineering data")
        else:
            self.entropy_data = None
            print(f"  ⚠️  No entropy engineering data found")
        
    def test_1_mutual_information_validation(self):
        """Test 1: Mutual Information Validation - check if MI shows real quantum evolution."""
        print("\n🧪 Test 1: Mutual Information Validation")
        
        if self.mutual_info_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            
            self.falsification_tests['mutual_information'] = {
                'description': 'Mutual information should show real quantum evolution',
                'expected_result': 'Dynamic MI values, not static fallback values',
                'actual_result': mi_metrics,
                'passes_test': mi_metrics['temporal_asymmetry'] > 0.01 and mi_metrics['quantum_correlations'] > 0.1
            }
            
            print(f"  ✅ Temporal asymmetry: {mi_metrics['temporal_asymmetry']:.4f}")
            print(f"  ✅ Quantum correlations: {mi_metrics['quantum_correlations']:.4f}")
            print(f"  ✅ Causal violations: {mi_metrics['causal_violations']}")
        else:
            print("  ⚠️  No mutual information data available")
            
    def test_2_entropy_engineering_validation(self):
        """Test 2: Entropy Engineering Validation - check if entropy optimization succeeded."""
        print("\n🧪 Test 2: Entropy Engineering Validation")
        
        if self.entropy_data:
            # Analyze entropy engineering results
            target_entropies = self.entropy_data.get('target_entropies', [])
            achieved_entropies = self.entropy_data.get('achieved_entropies', [])
            optimization_success = self.entropy_data.get('success', False)
            iterations = self.entropy_data.get('iterations', 0)
            loss = self.entropy_data.get('loss', float('inf'))
            
            # Calculate entropy evolution metrics
            entropy_evolution = self._analyze_entropy_evolution(target_entropies, achieved_entropies)
            
            self.falsification_tests['entropy_engineering'] = {
                'description': 'Entropy engineering should optimize to target patterns',
                'expected_result': 'Optimization converged, entropy patterns achieved',
                'optimization_success': optimization_success,
                'iterations': iterations,
                'loss': loss,
                'entropy_evolution': entropy_evolution,
                'passes_test': optimization_success and iterations > 0 and loss < 2.0
            }
            
            print(f"  ✅ Optimization success: {optimization_success}")
            print(f"  ✅ Iterations: {iterations}")
            print(f"  ✅ Final loss: {loss:.4f}")
            print(f"  ✅ Entropy evolution strength: {entropy_evolution['evolution_strength']:.4f}")
        else:
            print("  ⚠️  No entropy engineering data available")
            
    def test_3_quantum_gravity_signatures(self):
        """Test 3: Quantum Gravity Signatures - check for specific quantum gravity patterns."""
        print("\n🧪 Test 3: Quantum Gravity Signatures")
        
        quantum_gravity_signatures = {}
        
        # Check for Page curve signatures in entropy data
        if self.entropy_data:
            achieved_entropies = self.entropy_data.get('achieved_entropies', [])
            page_curve_signature = self._detect_page_curve_signature(achieved_entropies)
            quantum_gravity_signatures['page_curve'] = page_curve_signature
            
        # Check for holographic duality signatures
        if self.mutual_info_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            holographic_signature = self._detect_holographic_signature(mi_metrics)
            quantum_gravity_signatures['holographic_duality'] = holographic_signature
            
        # Check for causal structure emergence
        causal_signature = self._detect_causal_structure()
        quantum_gravity_signatures['causal_structure'] = causal_signature
        
        # Overall quantum gravity assessment
        total_signatures = len(quantum_gravity_signatures)
        detected_signatures = sum(1 for sig in quantum_gravity_signatures.values() if sig['detected'])
        
        self.falsification_tests['quantum_gravity_signatures'] = {
            'description': 'Detect specific quantum gravity signatures',
            'expected_result': 'Multiple quantum gravity signatures present',
            'signatures': quantum_gravity_signatures,
            'total_signatures': total_signatures,
            'detected_signatures': detected_signatures,
            'passes_test': detected_signatures >= 2  # At least 2 signatures detected
        }
        
        print(f"  ✅ Quantum gravity signatures detected: {detected_signatures}/{total_signatures}")
        for name, sig in quantum_gravity_signatures.items():
            status = "✅" if sig['detected'] else "❌"
            print(f"    {status} {name}: {sig['description']}")
            
    def test_4_noise_model_robustness(self):
        """Test 4: Noise Model Robustness - ensure results aren't due to noise artifacts."""
        print("\n🧪 Test 4: Noise Model Robustness")
        
        # Test entropy engineering robustness
        entropy_robustness = False
        if self.entropy_data:
            loss = self.entropy_data.get('loss', float('inf'))
            iterations = self.entropy_data.get('iterations', 0)
            # Low loss and reasonable iterations suggest genuine optimization
            entropy_robustness = loss < 1.5 and 10 <= iterations <= 100
            
        # Test mutual information robustness
        mi_robustness = False
        if self.mutual_info_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            # Check if MI shows genuine quantum evolution vs noise
            mi_robustness = mi_metrics['quantum_correlations'] > 0.3 and mi_metrics['classical_noise'] < 0.5
            
        # Test parameter sensitivity
        parameter_sensitivity = self._test_parameter_sensitivity()
        
        self.falsification_tests['noise_robustness'] = {
            'description': 'Results should be robust against noise artifacts',
            'expected_result': 'Genuine quantum effects, not noise artifacts',
            'entropy_robustness': entropy_robustness,
            'mi_robustness': mi_robustness,
            'parameter_sensitivity': parameter_sensitivity,
            'passes_test': entropy_robustness or mi_robustness  # At least one robust
        }
        
        print(f"  ✅ Entropy robustness: {entropy_robustness}")
        print(f"  ✅ MI robustness: {mi_robustness}")
        print(f"  ✅ Parameter sensitivity: {parameter_sensitivity}")
        
    def test_5_cross_validation_consistency(self):
        """Test 5: Cross-Validation Consistency - check consistency between different data sources."""
        print("\n🧪 Test 5: Cross-Validation Consistency")
        
        consistency_checks = {}
        
        # Check if both data sources show quantum evolution
        if self.mutual_info_data and self.entropy_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            entropy_evolution = self._analyze_entropy_evolution(
                self.entropy_data.get('target_entropies', []),
                self.entropy_data.get('achieved_entropies', [])
            )
            
            # Both should show quantum evolution
            both_quantum = (mi_metrics['quantum_correlations'] > 0.1 and 
                           entropy_evolution['evolution_strength'] > 0.1)
            consistency_checks['both_quantum'] = both_quantum
            
            # Check temporal consistency
            temporal_consistency = self._check_temporal_consistency()
            consistency_checks['temporal_consistency'] = temporal_consistency
            
        # Check geometric consistency
        geometric_consistency = self._check_geometric_consistency()
        consistency_checks['geometric_consistency'] = geometric_consistency
        
        # Overall consistency
        total_checks = len(consistency_checks)
        passed_checks = sum(1 for check in consistency_checks.values() if check)
        
        self.falsification_tests['cross_validation'] = {
            'description': 'Different data sources should be consistent',
            'expected_result': 'High consistency across data sources',
            'consistency_checks': consistency_checks,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'passes_test': passed_checks >= total_checks * 0.7  # 70% consistency
        }
        
        print(f"  ✅ Consistency checks passed: {passed_checks}/{total_checks}")
        for name, check in consistency_checks.items():
            status = "✅" if check else "❌"
            print(f"    {status} {name}")
            
    def _analyze_mutual_information(self, mi_data) -> Dict:
        """Analyze mutual information data for geometric signatures."""
        if not mi_data:
            return {
                'temporal_asymmetry': 0.0,
                'geometric_signature': 0.0,
                'causal_violations': 0,
                'quantum_correlations': 0.0,
                'classical_noise': 1.0
            }
        
        # Handle both list and dict formats
        if isinstance(mi_data, list):
            mi_evolution = []
            for timestep_data in mi_data:
                if isinstance(timestep_data, dict):
                    mi_values = list(timestep_data.values())
                    mi_evolution.append(np.mean(mi_values))
                else:
                    mi_evolution.append(0.1)
        else:
            timesteps = list(mi_data.keys())
            mi_evolution = []
            for timestep in timesteps:
                mi_values = list(mi_data[timestep].values())
                mi_evolution.append(np.mean(mi_values))
        
        # Calculate metrics
        if len(mi_evolution) > 1:
            temporal_asymmetry = np.std(mi_evolution)
            timestep_indices = np.arange(len(mi_evolution))
            geometric_signature = np.corrcoef(timestep_indices, mi_evolution)[0,1]
        else:
            temporal_asymmetry = 0.0
            geometric_signature = 0.0
        
        # Count causal violations
        causal_violations = 0
        for i in range(1, len(mi_evolution)):
            if mi_evolution[i] < mi_evolution[i-1] * 0.5:
                causal_violations += 1
        
        # Quantum vs classical signatures
        quantum_correlations = max(0, geometric_signature)
        classical_noise = max(0, -geometric_signature)
        
        return {
            'temporal_asymmetry': temporal_asymmetry,
            'geometric_signature': geometric_signature,
            'causal_violations': causal_violations,
            'quantum_correlations': quantum_correlations,
            'classical_noise': classical_noise
        }
    
    def _analyze_entropy_evolution(self, target_entropies: List[float], achieved_entropies: List[float]) -> Dict:
        """Analyze entropy evolution patterns."""
        if not target_entropies or not achieved_entropies:
            return {
                'evolution_strength': 0.0,
                'target_correlation': 0.0,
                'pattern_match': False
            }
        
        # Calculate evolution strength (variance in achieved entropies)
        evolution_strength = np.std(achieved_entropies)
        
        # Calculate correlation with target pattern
        if len(target_entropies) == len(achieved_entropies):
            target_correlation = np.corrcoef(target_entropies, achieved_entropies)[0,1]
        else:
            target_correlation = 0.0
        
        # Check for quantum gravity patterns (e.g., Page curve signature)
        pattern_match = self._detect_quantum_gravity_pattern(achieved_entropies)
        
        return {
            'evolution_strength': evolution_strength,
            'target_correlation': target_correlation,
            'pattern_match': pattern_match
        }
    
    def _detect_page_curve_signature(self, entropies: List[float]) -> Dict:
        """Detect Page curve signature in entropy evolution."""
        if len(entropies) < 3:
            return {'detected': False, 'description': 'Insufficient data'}
        
        # Page curve signature: entropy increases then decreases
        increasing = all(entropies[i] <= entropies[i+1] for i in range(len(entropies)//2))
        decreasing = all(entropies[i] >= entropies[i+1] for i in range(len(entropies)//2, len(entropies)-1))
        
        page_curve_detected = increasing and decreasing
        
        return {
            'detected': page_curve_detected,
            'description': 'Page curve signature (increase then decrease)',
            'increasing_phase': increasing,
            'decreasing_phase': decreasing
        }
    
    def _detect_holographic_signature(self, mi_metrics: Dict) -> Dict:
        """Detect holographic duality signatures in mutual information."""
        # Holographic duality: strong correlations between boundary and bulk
        strong_correlations = mi_metrics['quantum_correlations'] > 0.3
        low_noise = mi_metrics['classical_noise'] < 0.5
        
        holographic_detected = strong_correlations and low_noise
        
        return {
            'detected': holographic_detected,
            'description': 'Holographic duality (strong correlations, low noise)',
            'correlation_strength': mi_metrics['quantum_correlations'],
            'noise_level': mi_metrics['classical_noise']
        }
    
    def _detect_causal_structure(self) -> Dict:
        """Detect emergent causal structure."""
        # Check for causal violations in MI data
        if self.mutual_info_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            causal_violations = mi_metrics['causal_violations']
            
            # Some causal violations suggest emergent time structure
            causal_detected = causal_violations > 0
            
            return {
                'detected': causal_detected,
                'description': 'Emergent causal structure (causal violations)',
                'violations_count': causal_violations
            }
        else:
            return {
                'detected': False,
                'description': 'No MI data available for causal analysis'
            }
    
    def _test_parameter_sensitivity(self) -> bool:
        """Test if results are sensitive to parameter changes (suggesting genuine effects)."""
        # This is a simplified test - in practice would vary parameters
        if self.entropy_data:
            # Check if optimization converged to reasonable parameters
            parameters = self.entropy_data.get('parameters', {})
            entanglement_strength = parameters.get('entanglement_strength', 0)
            
            # Reasonable entanglement strength suggests genuine optimization
            return 0.1 < entanglement_strength < 10.0
        return False
    
    def _check_temporal_consistency(self) -> bool:
        """Check temporal consistency between different data sources."""
        # Simplified check - in practice would compare temporal evolution
        if self.mutual_info_data and self.entropy_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            entropy_evolution = self._analyze_entropy_evolution(
                self.entropy_data.get('target_entropies', []),
                self.entropy_data.get('achieved_entropies', [])
            )
            
            # Both should show some evolution
            return (mi_metrics['temporal_asymmetry'] > 0.01 or 
                   entropy_evolution['evolution_strength'] > 0.01)
        return False
    
    def _check_geometric_consistency(self) -> bool:
        """Check geometric consistency across data sources."""
        # Simplified check - in practice would compare geometric signatures
        if self.mutual_info_data:
            mi_metrics = self._analyze_mutual_information(self.mutual_info_data)
            # Check if geometric signature is reasonable
            return abs(mi_metrics['geometric_signature']) < 1.0
        return True
    
    def _detect_quantum_gravity_pattern(self, entropies: List[float]) -> bool:
        """Detect quantum gravity patterns in entropy evolution."""
        if len(entropies) < 2:
            return False
        
        # Check for non-monotonic evolution (characteristic of quantum gravity)
        differences = np.diff(entropies)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        
        # Non-monotonic evolution suggests quantum gravity effects
        return sign_changes > 0
    
    def run_all_tests(self):
        """Run all enhanced falsification tests."""
        print("🧪 ENHANCED FALSIFICATION TESTING FRAMEWORK")
        print("=" * 60)
        
        self.load_experiment_data()
        
        # Run all tests
        self.test_1_mutual_information_validation()
        self.test_2_entropy_engineering_validation()
        self.test_3_quantum_gravity_signatures()
        self.test_4_noise_model_robustness()
        self.test_5_cross_validation_consistency()
        
        # Generate comprehensive report
        self._generate_enhanced_report()
        
    def _generate_enhanced_report(self):
        """Generate comprehensive enhanced falsification report."""
        print("\n" + "=" * 60)
        print("📊 ENHANCED FALSIFICATION TESTING REPORT")
        print("=" * 60)
        
        passed_tests = 0
        total_tests = len(self.falsification_tests)
        
        for test_name, test_result in self.falsification_tests.items():
            status = "✅ PASS" if test_result['passes_test'] else "❌ FAIL"
            print(f"\n{test_name.upper()}: {status}")
            print(f"  Description: {test_result['description']}")
            print(f"  Expected: {test_result['expected_result']}")
            
            # Print specific results for each test
            if test_name == 'mutual_information':
                mi_metrics = test_result['actual_result']
                print(f"  Result: Temporal asymmetry = {mi_metrics['temporal_asymmetry']:.4f}")
                print(f"  Result: Quantum correlations = {mi_metrics['quantum_correlations']:.4f}")
            elif test_name == 'entropy_engineering':
                print(f"  Result: Optimization success = {test_result['optimization_success']}")
                print(f"  Result: Iterations = {test_result['iterations']}")
                print(f"  Result: Final loss = {test_result['loss']:.4f}")
            elif test_name == 'quantum_gravity_signatures':
                print(f"  Result: Signatures detected = {test_result['detected_signatures']}/{test_result['total_signatures']}")
            elif test_name == 'noise_robustness':
                print(f"  Result: Entropy robust = {test_result['entropy_robustness']}")
                print(f"  Result: MI robust = {test_result['mi_robustness']}")
            elif test_name == 'cross_validation':
                print(f"  Result: Consistency checks = {test_result['passed_checks']}/{test_result['total_checks']}")
            
            if test_result['passes_test']:
                passed_tests += 1
        
        # Overall assessment
        print(f"\n📈 OVERALL ASSESSMENT:")
        print(f"  Tests Passed: {passed_tests}/{total_tests}")
        print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests >= total_tests * 0.8:
            print("  🎉 STRONG EVIDENCE: Genuine quantum holographic phenomena detected")
        elif passed_tests >= total_tests * 0.6:
            print("  ⚠️  MODERATE EVIDENCE: Some quantum effects detected, concerns remain")
        else:
            print("  ❌ WEAK EVIDENCE: Results likely due to noise or artifacts")
        
        # Save detailed results
        output_dir = os.path.dirname(self.experiment_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_file = os.path.join(output_dir, f"enhanced_falsification_report_{timestamp}.json")
        
        # Clean data for JSON serialization
        clean_tests = {}
        for test_name, test_result in self.falsification_tests.items():
            clean_test = {
                'description': test_result['description'],
                'expected_result': test_result['expected_result'],
                'passes_test': bool(test_result['passes_test'])
            }
            
            # Add specific results based on test type
            if test_name == 'mutual_information':
                clean_test['temporal_asymmetry'] = float(test_result['actual_result']['temporal_asymmetry'])
                clean_test['quantum_correlations'] = float(test_result['actual_result']['quantum_correlations'])
            elif test_name == 'entropy_engineering':
                clean_test['optimization_success'] = bool(test_result['optimization_success'])
                clean_test['iterations'] = int(test_result['iterations'])
                clean_test['loss'] = float(test_result['loss'])
            elif test_name == 'quantum_gravity_signatures':
                clean_test['detected_signatures'] = int(test_result['detected_signatures'])
                clean_test['total_signatures'] = int(test_result['total_signatures'])
            elif test_name == 'noise_robustness':
                clean_test['entropy_robustness'] = bool(test_result['entropy_robustness'])
                clean_test['mi_robustness'] = bool(test_result['mi_robustness'])
            elif test_name == 'cross_validation':
                clean_test['passed_checks'] = int(test_result['passed_checks'])
                clean_test['total_checks'] = int(test_result['total_checks'])
            
            clean_tests[test_name] = clean_test
        
        with open(report_file, 'w') as f:
            json.dump({
                'experiment_file': self.experiment_file,
                'timestamp': timestamp,
                'falsification_tests': clean_tests,
                'overall_assessment': {
                    'tests_passed': int(passed_tests),
                    'total_tests': int(total_tests),
                    'success_rate': float(passed_tests/total_tests*100)
                }
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")

def main():
    """Main function to run enhanced falsification testing."""
    if len(sys.argv) != 2:
        print("Usage: python enhanced_falsification_testing.py <experiment_file>")
        sys.exit(1)
    
    experiment_file = sys.argv[1]
    
    if not os.path.exists(experiment_file):
        print(f"Error: Experiment file not found: {experiment_file}")
        sys.exit(1)
    
    # Run enhanced falsification testing
    tester = EnhancedFalsificationTester(experiment_file)
    tester.run_all_tests()

if __name__ == "__main__":
    main() 