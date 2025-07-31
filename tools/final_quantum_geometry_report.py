#!/usr/bin/env python3
"""
Final Quantum Geometry Report
============================

Synthesizes comprehensive analysis results to provide insights about:
1. Geometric locality in mutual information matrix
2. Graph structure and cluster hierarchy  
3. MDS visualization of bulk geometry
4. Ryu-Takayanagi consistency with dissimilarity scaling
5. Entanglement spectrum comparison to Haar states
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Any

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FinalQuantumGeometryReport:
    def __init__(self, analysis_results_file: str):
        """Initialize with comprehensive analysis results."""
        self.analysis_results_file = analysis_results_file
        self.results = self.load_results()
        
    def load_results(self) -> Dict:
        """Load comprehensive analysis results."""
        try:
            with open(self.analysis_results_file, 'r') as f:
                results = json.load(f)
            print(f"✅ Loaded analysis results from: {self.analysis_results_file}")
            return results
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return {}
    
    def analyze_mutual_information_geometric_locality(self) -> Dict:
        """Analyze geometric locality in mutual information matrix."""
        print("🔍 Analyzing Mutual Information Geometric Locality...")
        
        locality = self.results.get('geometric_locality', {})
        
        # Key findings
        nn_correlation = locality.get('nearest_neighbor_correlation', 0)
        locality_score = locality.get('locality_score', 0)
        distance_decay_rate = locality.get('distance_decay_rate', 0)
        nonlocal_fraction = locality.get('nonlocal_correlation_fraction', 0)
        
        analysis = {
            'nearest_neighbor_correlation': nn_correlation,
            'locality_score': locality_score,
            'distance_decay_rate': distance_decay_rate,
            'nonlocal_fraction': nonlocal_fraction,
            'interpretation': {}
        }
        
        # Interpret locality score
        if locality_score > 1.0:
            analysis['interpretation']['locality'] = "STRONG LOCALITY: High nearest-neighbor correlations relative to average"
        elif locality_score > 0.5:
            analysis['interpretation']['locality'] = "MODERATE LOCALITY: Some geometric structure present"
        else:
            analysis['interpretation']['locality'] = "WEAK LOCALITY: Primarily non-local correlations"
        
        # Interpret distance decay
        if distance_decay_rate > 0.1:
            analysis['interpretation']['distance_decay'] = "EXPONENTIAL DECAY: Strong distance-dependent correlations"
        else:
            analysis['interpretation']['distance_decay'] = "NO DECAY: Correlations not distance-dependent"
        
        # Interpret non-local fraction
        if nonlocal_fraction > 1.0:
            analysis['interpretation']['nonlocal'] = "HIGHLY NON-LOCAL: Extensive long-range correlations"
        elif nonlocal_fraction > 0.5:
            analysis['interpretation']['nonlocal'] = "MODERATELY NON-LOCAL: Some long-range structure"
        else:
            analysis['interpretation']['nonlocal'] = "PRIMARILY LOCAL: Limited long-range correlations"
        
        print(f"  📊 Nearest-neighbor correlation: {nn_correlation:.4f}")
        print(f"  📊 Locality score: {locality_score:.4f}")
        print(f"  📊 Distance decay rate: {distance_decay_rate:.4f}")
        print(f"  📊 Non-local fraction: {nonlocal_fraction:.4f}")
        
        return analysis
    
    def analyze_graph_structure_cluster_hierarchy(self) -> Dict:
        """Analyze graph structure and cluster hierarchy."""
        print("🔍 Analyzing Graph Structure and Cluster Hierarchy...")
        
        graph = self.results.get('graph_structure', {})
        
        # Key metrics
        density = graph.get('density', 0)
        clustering = graph.get('average_clustering', 0)
        modularity = graph.get('modularity', 0)
        num_communities = graph.get('num_communities', 1)
        community_sizes = graph.get('community_sizes', [])
        is_connected = graph.get('is_connected', False)
        
        analysis = {
            'density': density,
            'clustering': clustering,
            'modularity': modularity,
            'num_communities': num_communities,
            'community_sizes': community_sizes,
            'is_connected': is_connected,
            'interpretation': {}
        }
        
        # Interpret graph density
        if density > 0.5:
            analysis['interpretation']['density'] = "HIGH DENSITY: Dense connectivity, strong interactions"
        elif density > 0.2:
            analysis['interpretation']['density'] = "MODERATE DENSITY: Balanced connectivity"
        else:
            analysis['interpretation']['density'] = "LOW DENSITY: Sparse connectivity, weak interactions"
        
        # Interpret clustering
        if clustering > 0.3:
            analysis['interpretation']['clustering'] = "HIGH CLUSTERING: Strong local community structure"
        elif clustering > 0.1:
            analysis['interpretation']['clustering'] = "MODERATE CLUSTERING: Some local structure"
        else:
            analysis['interpretation']['clustering'] = "LOW CLUSTERING: No local community structure"
        
        # Interpret modularity
        if modularity > 0.3:
            analysis['interpretation']['modularity'] = "HIGH MODULARITY: Strong community separation"
        elif modularity > 0.1:
            analysis['interpretation']['modularity'] = "MODERATE MODULARITY: Some community structure"
        else:
            analysis['interpretation']['modularity'] = "LOW MODULARITY: No clear community structure"
        
        # Interpret connectivity
        if is_connected:
            analysis['interpretation']['connectivity'] = "CONNECTED: Single coherent structure"
        else:
            analysis['interpretation']['connectivity'] = f"DISCONNECTED: {num_communities} separate components"
        
        print(f"  📊 Graph density: {density:.4f}")
        print(f"  📊 Average clustering: {clustering:.4f}")
        print(f"  📊 Modularity: {modularity:.4f}")
        print(f"  📊 Number of communities: {num_communities}")
        print(f"  📊 Connected: {is_connected}")
        
        return analysis
    
    def analyze_mds_bulk_geometry(self) -> Dict:
        """Analyze MDS visualization of bulk geometry."""
        print("🔍 Analyzing MDS Bulk Geometry Visualization...")
        
        mds = self.results.get('mds_visualization', {})
        
        # Key metrics
        stress_2d = mds.get('stress_2d', 1.0)
        stress_3d = mds.get('stress_3d', 1.0)
        coords_2d = mds.get('coords_2d', [])
        coords_3d = mds.get('coords_3d', [])
        
        analysis = {
            'stress_2d': stress_2d,
            'stress_3d': stress_3d,
            'coords_2d': coords_2d,
            'coords_3d': coords_3d,
            'interpretation': {}
        }
        
        # Interpret MDS stress (lower is better)
        if stress_2d < 0.1:
            analysis['interpretation']['stress_2d'] = "EXCELLENT 2D EMBEDDING: Very low distortion"
        elif stress_2d < 0.3:
            analysis['interpretation']['stress_2d'] = "GOOD 2D EMBEDDING: Low distortion"
        elif stress_2d < 0.5:
            analysis['interpretation']['stress_2d'] = "FAIR 2D EMBEDDING: Moderate distortion"
        else:
            analysis['interpretation']['stress_2d'] = "POOR 2D EMBEDDING: High distortion"
        
        if stress_3d < 0.1:
            analysis['interpretation']['stress_3d'] = "EXCELLENT 3D EMBEDDING: Very low distortion"
        elif stress_3d < 0.3:
            analysis['interpretation']['stress_3d'] = "GOOD 3D EMBEDDING: Low distortion"
        elif stress_3d < 0.5:
            analysis['interpretation']['stress_3d'] = "FAIR 3D EMBEDDING: Moderate distortion"
        else:
            analysis['interpretation']['stress_3d'] = "POOR 3D EMBEDDING: High distortion"
        
        # Analyze geometric structure
        if coords_2d:
            coords_array = np.array(coords_2d)
            # Calculate geometric properties
            centroid = np.mean(coords_array, axis=0)
            distances_from_centroid = np.linalg.norm(coords_array - centroid, axis=1)
            radius = np.std(distances_from_centroid)
            
            analysis['geometric_properties'] = {
                'centroid': centroid.tolist(),
                'radius': float(radius),
                'spatial_distribution': 'concentrated' if radius < 0.3 else 'spread'
            }
            
            if radius < 0.3:
                analysis['interpretation']['geometry'] = "CONCENTRATED: Points clustered near center"
            else:
                analysis['interpretation']['geometry'] = "SPREAD: Points distributed across space"
        
        print(f"  📊 2D MDS stress: {stress_2d:.4f}")
        print(f"  📊 3D MDS stress: {stress_3d:.4f}")
        if 'geometric_properties' in analysis:
            print(f"  📊 Spatial radius: {analysis['geometric_properties']['radius']:.4f}")
        
        return analysis
    
    def analyze_ryu_takayanagi_consistency(self) -> Dict:
        """Analyze Ryu-Takayanagi consistency with dissimilarity scaling."""
        print("🔍 Analyzing Ryu-Takayanagi Consistency...")
        
        rt = self.results.get('ryu_takayanagi', {})
        
        # Key metrics
        scaling_exponent = rt.get('rt_scaling_exponent', 0)
        consistency_score = rt.get('rt_consistency_score', 0)
        area_law_violation = rt.get('area_law_violation', 0)
        power_law_r2 = rt.get('rt_power_law_r2', 0)
        
        analysis = {
            'scaling_exponent': scaling_exponent,
            'consistency_score': consistency_score,
            'area_law_violation': area_law_violation,
            'power_law_r2': power_law_r2,
            'interpretation': {}
        }
        
        # Interpret RT scaling exponent (should be ~1 for RT consistency)
        if abs(scaling_exponent - 1.0) < 0.2:
            analysis['interpretation']['scaling'] = "RT CONSISTENT: Scaling exponent close to 1"
        elif abs(scaling_exponent - 1.0) < 0.5:
            analysis['interpretation']['scaling'] = "MODERATELY RT CONSISTENT: Scaling exponent near 1"
        else:
            analysis['interpretation']['scaling'] = f"RT INCONSISTENT: Scaling exponent {scaling_exponent:.2f} far from 1"
        
        # Interpret consistency score
        if consistency_score > 0.8:
            analysis['interpretation']['consistency'] = "HIGH CONSISTENCY: Strong agreement with RT"
        elif consistency_score > 0.5:
            analysis['interpretation']['consistency'] = "MODERATE CONSISTENCY: Some agreement with RT"
        else:
            analysis['interpretation']['consistency'] = "LOW CONSISTENCY: Poor agreement with RT"
        
        # Interpret area law violation
        if area_law_violation > 0.1:
            analysis['interpretation']['area_law'] = "AREA LAW VIOLATED: Volume law scaling detected"
        elif area_law_violation > 0.05:
            analysis['interpretation']['area_law'] = "MODERATE VIOLATION: Some volume law behavior"
        else:
            analysis['interpretation']['area_law'] = "AREA LAW PRESERVED: Boundary scaling maintained"
        
        print(f"  📊 RT scaling exponent: {scaling_exponent:.4f}")
        print(f"  📊 RT consistency score: {consistency_score:.4f}")
        print(f"  📊 Area law violation: {area_law_violation:.4f}")
        print(f"  📊 Power law R²: {power_law_r2:.4f}")
        
        return analysis
    
    def analyze_entanglement_spectrum_haar_comparison(self) -> Dict:
        """Analyze entanglement spectrum and compare to Haar random states."""
        print("🔍 Analyzing Entanglement Spectrum vs Haar States...")
        
        ent = self.results.get('entanglement_spectrum', {})
        
        # Key metrics
        avg_entropy = ent.get('average_entanglement_entropy', 0)
        haar_deviation = ent.get('haar_deviation', 0)
        page_curve_r2 = ent.get('page_curve_r2', 0)
        expected_haar_entropy = ent.get('expected_haar_entropy', 0)
        spectral_gap = ent.get('spectral_gap', 0)
        spectrum_flatness = ent.get('spectrum_flatness', 0)
        
        analysis = {
            'avg_entropy': avg_entropy,
            'haar_deviation': haar_deviation,
            'page_curve_r2': page_curve_r2,
            'expected_haar_entropy': expected_haar_entropy,
            'spectral_gap': spectral_gap,
            'spectrum_flatness': spectrum_flatness,
            'interpretation': {}
        }
        
        # Interpret Haar deviation
        if haar_deviation < 0.1:
            analysis['interpretation']['haar'] = "HAAR-LIKE: Very close to random state behavior"
        elif haar_deviation < 0.3:
            analysis['interpretation']['haar'] = "NEAR-HAAR: Similar to random state behavior"
        elif haar_deviation < 0.5:
            analysis['interpretation']['haar'] = "MODERATELY HAAR: Some random state characteristics"
        else:
            analysis['interpretation']['haar'] = "NON-HAAR: Significantly different from random states"
        
        # Interpret Page curve fit
        if page_curve_r2 > 0.8:
            analysis['interpretation']['page_curve'] = "STRONG PAGE CURVE: Excellent fit to Page curve model"
        elif page_curve_r2 > 0.6:
            analysis['interpretation']['page_curve'] = "MODERATE PAGE CURVE: Good fit to Page curve model"
        elif page_curve_r2 > 0.4:
            analysis['interpretation']['page_curve'] = "WEAK PAGE CURVE: Some Page curve characteristics"
        else:
            analysis['interpretation']['page_curve'] = "NO PAGE CURVE: Poor fit to Page curve model"
        
        # Interpret spectral properties
        if spectral_gap > 0.5:
            analysis['interpretation']['spectral_gap'] = "LARGE GAP: Well-separated eigenvalues"
        else:
            analysis['interpretation']['spectral_gap'] = "SMALL GAP: Closely spaced eigenvalues"
        
        if spectrum_flatness < 0.5:
            analysis['interpretation']['flatness'] = "FLAT SPECTRUM: Uniform eigenvalue distribution"
        else:
            analysis['interpretation']['flatness'] = "PEAKED SPECTRUM: Non-uniform eigenvalue distribution"
        
        print(f"  📊 Average entanglement entropy: {avg_entropy:.4f}")
        print(f"  📊 Expected Haar entropy: {expected_haar_entropy:.4f}")
        print(f"  📊 Haar deviation: {haar_deviation:.4f}")
        print(f"  📊 Page curve R²: {page_curve_r2:.4f}")
        print(f"  📊 Spectral gap: {spectral_gap:.4f}")
        print(f"  📊 Spectrum flatness: {spectrum_flatness:.4f}")
        
        return analysis
    
    def generate_comprehensive_report(self, output_dir: str = None) -> Dict:
        """Generate comprehensive quantum geometry report."""
        print("🚀 Generating Final Quantum Geometry Report")
        print("=" * 60)
        
        if output_dir is None:
            output_dir = "final_quantum_geometry_report"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run all analyses
        analyses = {
            'mutual_information_locality': self.analyze_mutual_information_geometric_locality(),
            'graph_structure': self.analyze_graph_structure_cluster_hierarchy(),
            'mds_geometry': self.analyze_mds_bulk_geometry(),
            'ryu_takayanagi': self.analyze_ryu_takayanagi_consistency(),
            'entanglement_spectrum': self.analyze_entanglement_spectrum_haar_comparison()
        }
        
        # Create summary visualizations
        self.create_report_visualizations(analyses, output_dir)
        
        # Save comprehensive report
        self.save_comprehensive_report(analyses, output_dir)
        
        # Print executive summary
        self.print_executive_summary(analyses)
        
        return analyses
    
    def create_report_visualizations(self, analyses: Dict, output_dir: str):
        """Create visualizations for the final report."""
        print("📈 Creating Report Visualizations...")
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Mutual Information Locality
        mi_loc = analyses['mutual_information_locality']
        axes[0, 0].bar(['NN Corr', 'Locality Score', 'Non-local Frac'], 
                      [mi_loc['nearest_neighbor_correlation'],
                       mi_loc['locality_score'],
                       mi_loc['nonlocal_fraction']])
        axes[0, 0].set_title('Mutual Information Geometric Locality')
        axes[0, 0].set_ylabel('Value')
        
        # 2. Graph Structure
        graph = analyses['graph_structure']
        axes[0, 1].bar(['Density', 'Clustering', 'Modularity'], 
                      [graph['density'], graph['clustering'], graph['modularity']])
        axes[0, 1].set_title('Graph Structure Metrics')
        axes[0, 1].set_ylabel('Value')
        
        # 3. MDS Quality
        mds = analyses['mds_geometry']
        axes[0, 2].bar(['2D Stress', '3D Stress'], 
                      [mds['stress_2d'], mds['stress_3d']])
        axes[0, 2].set_title('MDS Visualization Quality')
        axes[0, 2].set_ylabel('Stress (lower is better)')
        
        # 4. Ryu-Takayanagi
        rt = analyses['ryu_takayanagi']
        axes[1, 0].bar(['RT Exponent', 'Consistency', 'Area Law Viol'], 
                      [rt['scaling_exponent'], rt['consistency_score'], rt['area_law_violation']])
        axes[1, 0].set_title('Ryu-Takayanagi Consistency')
        axes[1, 0].set_ylabel('Value')
        
        # 5. Entanglement Spectrum
        ent = analyses['entanglement_spectrum']
        axes[1, 1].bar(['Avg Entropy', 'Haar Dev', 'Page R²'], 
                      [ent['avg_entropy'], ent['haar_deviation'], ent['page_curve_r2']])
        axes[1, 1].set_title('Entanglement Spectrum Analysis')
        axes[1, 1].set_ylabel('Value')
        
        # 6. Overall Assessment
        overall_score = self.calculate_overall_quantum_geometry_score(analyses)
        axes[1, 2].bar(['Quantum Geometry Score'], [overall_score])
        axes[1, 2].set_title('Overall Quantum Geometry Assessment')
        axes[1, 2].set_ylabel('Score (0-1)')
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, 'quantum_geometry_comprehensive_dashboard.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  📈 Comprehensive dashboard saved: {plot_file}")
    
    def calculate_overall_quantum_geometry_score(self, analyses: Dict) -> float:
        """Calculate overall quantum geometry score."""
        score = 0.0
        count = 0
        
        # Locality score (normalized)
        mi_loc = analyses['mutual_information_locality']
        if 'locality_score' in mi_loc:
            score += min(mi_loc['locality_score'] / 2.0, 1.0)
            count += 1
        
        # Graph modularity
        graph = analyses['graph_structure']
        if 'modularity' in graph:
            score += max(0, graph['modularity'])
            count += 1
        
        # RT consistency
        rt = analyses['ryu_takayanagi']
        if 'consistency_score' in rt:
            score += max(0, rt['consistency_score'])
            count += 1
        
        # Entanglement spectrum (closer to Haar is better)
        ent = analyses['entanglement_spectrum']
        if 'haar_deviation' in ent:
            score += max(0, 1.0 - ent['haar_deviation'])
            count += 1
        
        # Page curve fit
        if 'page_curve_r2' in ent:
            score += ent['page_curve_r2']
            count += 1
        
        return score / count if count > 0 else 0.0
    
    def save_comprehensive_report(self, analyses: Dict, output_dir: str):
        """Save comprehensive quantum geometry report."""
        report_file = os.path.join(output_dir, 'quantum_geometry_comprehensive_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("QUANTUM GEOMETRY COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Executive Summary
            overall_score = self.calculate_overall_quantum_geometry_score(analyses)
            f.write("EXECUTIVE SUMMARY:\n")
            f.write(f"  Overall Quantum Geometry Score: {overall_score:.4f}\n\n")
            
            if overall_score > 0.8:
                f.write("  🎉 EXCELLENT: Strong evidence of emergent quantum geometry\n")
                f.write("  - High locality and modularity\n")
                f.write("  - Consistent with Ryu-Takayanagi\n")
                f.write("  - Entanglement spectrum close to Haar random\n")
            elif overall_score > 0.6:
                f.write("  ✅ GOOD: Moderate evidence of quantum geometry\n")
                f.write("  - Some geometric structure detected\n")
                f.write("  - May need parameter tuning\n")
            elif overall_score > 0.4:
                f.write("  ⚠️  FAIR: Weak evidence of quantum geometry\n")
                f.write("  - Limited geometric structure\n")
                f.write("  - Consider different circuit design\n")
            else:
                f.write("  ❌ POOR: Little evidence of quantum geometry\n")
                f.write("  - Primarily classical correlations\n")
                f.write("  - Significant redesign needed\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Detailed Analysis
            f.write("DETAILED ANALYSIS:\n\n")
            
            # 1. Mutual Information Geometric Locality
            f.write("1. MUTUAL INFORMATION GEOMETRIC LOCALITY:\n")
            mi_loc = analyses['mutual_information_locality']
            f.write(f"  - Nearest-neighbor correlation: {mi_loc['nearest_neighbor_correlation']:.4f}\n")
            f.write(f"  - Locality score: {mi_loc['locality_score']:.4f}\n")
            f.write(f"  - Distance decay rate: {mi_loc['distance_decay_rate']:.4f}\n")
            f.write(f"  - Non-local fraction: {mi_loc['nonlocal_fraction']:.4f}\n")
            f.write(f"  - Interpretation: {mi_loc['interpretation']['locality']}\n\n")
            
            # 2. Graph Structure and Cluster Hierarchy
            f.write("2. GRAPH STRUCTURE AND CLUSTER HIERARCHY:\n")
            graph = analyses['graph_structure']
            f.write(f"  - Graph density: {graph['density']:.4f}\n")
            f.write(f"  - Average clustering: {graph['clustering']:.4f}\n")
            f.write(f"  - Modularity: {graph['modularity']:.4f}\n")
            f.write(f"  - Number of communities: {graph['num_communities']}\n")
            f.write(f"  - Connected: {graph['is_connected']}\n")
            f.write(f"  - Interpretation: {graph['interpretation']['modularity']}\n\n")
            
            # 3. MDS Bulk Geometry Visualization
            f.write("3. MDS BULK GEOMETRY VISUALIZATION:\n")
            mds = analyses['mds_geometry']
            f.write(f"  - 2D MDS stress: {mds['stress_2d']:.4f}\n")
            f.write(f"  - 3D MDS stress: {mds['stress_3d']:.4f}\n")
            if 'geometric_properties' in mds:
                f.write(f"  - Spatial radius: {mds['geometric_properties']['radius']:.4f}\n")
            f.write(f"  - Interpretation: {mds['interpretation']['stress_2d']}\n\n")
            
            # 4. Ryu-Takayanagi Consistency
            f.write("4. RYU-TAKAYANAGI CONSISTENCY:\n")
            rt = analyses['ryu_takayanagi']
            f.write(f"  - RT scaling exponent: {rt['scaling_exponent']:.4f}\n")
            f.write(f"  - RT consistency score: {rt['consistency_score']:.4f}\n")
            f.write(f"  - Area law violation: {rt['area_law_violation']:.4f}\n")
            f.write(f"  - Power law R²: {rt['power_law_r2']:.4f}\n")
            f.write(f"  - Interpretation: {rt['interpretation']['scaling']}\n\n")
            
            # 5. Entanglement Spectrum vs Haar States
            f.write("5. ENTANGLEMENT SPECTRUM VS HAAR STATES:\n")
            ent = analyses['entanglement_spectrum']
            f.write(f"  - Average entanglement entropy: {ent['avg_entropy']:.4f}\n")
            f.write(f"  - Expected Haar entropy: {ent['expected_haar_entropy']:.4f}\n")
            f.write(f"  - Haar deviation: {ent['haar_deviation']:.4f}\n")
            f.write(f"  - Page curve R²: {ent['page_curve_r2']:.4f}\n")
            f.write(f"  - Spectral gap: {ent['spectral_gap']:.4f}\n")
            f.write(f"  - Interpretation: {ent['interpretation']['haar']}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            if overall_score > 0.8:
                f.write("  - Continue with current approach\n")
                f.write("  - Scale up to larger systems\n")
                f.write("  - Investigate specific geometric features\n")
            elif overall_score > 0.6:
                f.write("  - Optimize circuit parameters\n")
                f.write("  - Increase entanglement strength\n")
                f.write("  - Add more geometric structure\n")
            elif overall_score > 0.4:
                f.write("  - Redesign circuit architecture\n")
                f.write("  - Implement stronger geometric constraints\n")
                f.write("  - Consider different entanglement patterns\n")
            else:
                f.write("  - Fundamental redesign needed\n")
                f.write("  - Implement geometric locality constraints\n")
                f.write("  - Add holographic boundary conditions\n")
        
        print(f"📝 Comprehensive report saved: {report_file}")
    
    def print_executive_summary(self, analyses: Dict):
        """Print executive summary to console."""
        print("\n" + "=" * 60)
        print("FINAL QUANTUM GEOMETRY REPORT - EXECUTIVE SUMMARY")
        print("=" * 60)
        
        overall_score = self.calculate_overall_quantum_geometry_score(analyses)
        print(f"🎯 Overall Quantum Geometry Score: {overall_score:.4f}")
        
        print(f"\n📊 Key Findings:")
        
        # Mutual Information Locality
        mi_loc = analyses['mutual_information_locality']
        print(f"  🔍 Geometric Locality: {mi_loc['interpretation']['locality']}")
        
        # Graph Structure
        graph = analyses['graph_structure']
        print(f"  🕸️  Graph Structure: {graph['interpretation']['modularity']}")
        
        # MDS Geometry
        mds = analyses['mds_geometry']
        print(f"  📐 MDS Geometry: {mds['interpretation']['stress_2d']}")
        
        # Ryu-Takayanagi
        rt = analyses['ryu_takayanagi']
        print(f"  🌌 RT Consistency: {rt['interpretation']['scaling']}")
        
        # Entanglement Spectrum
        ent = analyses['entanglement_spectrum']
        print(f"  ⚛️  Entanglement: {ent['interpretation']['haar']}")
        
        print(f"\n🎯 Final Assessment:")
        if overall_score > 0.8:
            print(f"  🎉 EXCELLENT: Strong evidence of emergent quantum geometry!")
        elif overall_score > 0.6:
            print(f"  ✅ GOOD: Moderate evidence of quantum geometry")
        elif overall_score > 0.4:
            print(f"  ⚠️  FAIR: Weak evidence of quantum geometry")
        else:
            print(f"  ❌ POOR: Little evidence of quantum geometry")

def main():
    """Main function to generate final quantum geometry report."""
    if len(sys.argv) < 2:
        print("Usage: python final_quantum_geometry_report.py <analysis_results_file> [output_directory]")
        print("Example: python final_quantum_geometry_report.py experiment_logs/comprehensive_geometry_analysis/comprehensive_analysis_results.json")
        sys.exit(1)
    
    analysis_results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "final_quantum_geometry_report"
    
    # Create and run the report generator
    report_generator = FinalQuantumGeometryReport(analysis_results_file)
    
    try:
        analyses = report_generator.generate_comprehensive_report(output_dir)
        print(f"\n✅ Final report complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 