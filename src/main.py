#!/usr/bin/env python3
"""
ADSS 2.0 (Adaptive Diffusion Step Skipping 2.0) Experiment
Main script that orchestrates all experiments and saves results.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from preprocess import load_data, setup_experiment_environment
from train import create_models, DummyDiffusionModel
from evaluate import run_inference_benchmark, run_component_ablation, run_dynamic_analysis

def main():
    """Main experiment orchestration function."""
    print("=" * 60)
    print("ADSS 2.0 (Adaptive Diffusion Step Skipping 2.0) Experiment")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    device = setup_experiment_environment()
    print(f"Using device: {device}")
    print()
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             '.research', 'iteration1', 'images')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    print("Loading dataset...")
    dataset = load_data('CIFAR10')
    print(f"Dataset loaded: {len(dataset)} samples")
    print()
    
    print("Creating models...")
    baseline_model, adss_model = create_models(device)
    print("Models created successfully")
    print()
    
    all_results = {}
    
    print("=" * 50)
    print("EXPERIMENT 1: Inference Speed and Quality Benchmarking")
    print("=" * 50)
    
    results_exp1 = run_inference_benchmark(baseline_model, adss_model, dataset, output_dir)
    all_results['experiment_1'] = results_exp1
    
    print("\nExperiment 1 Results:")
    for method, metrics in results_exp1.items():
        print(f"  {method}:")
        print(f"    Average steps: {metrics['avg_steps']:.2f}")
        print(f"    Inference time: {metrics['inference_time']:.4f}s")
        print(f"    FID score: {metrics['FID']:.4f}")
    print()
    
    print("=" * 50)
    print("EXPERIMENT 2: Component Ablation Study")
    print("=" * 50)
    
    results_exp2 = run_component_ablation(adss_model, dataset, output_dir)
    all_results['experiment_2'] = results_exp2
    
    print("\nExperiment 2 Results:")
    for variant, metrics in results_exp2.items():
        print(f"  {variant}:")
        print(f"    Average steps: {metrics['avg_steps']:.2f}")
        print(f"    Inference time: {metrics['inference_time']:.4f}s")
        print(f"    FID score: {metrics['FID']:.4f}")
    print()
    
    print("=" * 50)
    print("EXPERIMENT 3: Dynamic Step Skipping Analysis")
    print("=" * 50)
    
    logs_exp3 = run_dynamic_analysis(adss_model, output_dir)
    all_results['experiment_3'] = {
        'num_samples_analyzed': len(logs_exp3),
        'sample_logs': logs_exp3[:3]  # Store first 3 samples for brevity
    }
    
    print(f"\nExperiment 3 completed: Analyzed {len(logs_exp3)} samples")
    print("Dynamic step skipping visualization saved")
    print()
    
    results_file = os.path.join(output_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")
    
    generate_summary_report(all_results, output_dir)
    
    print("=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    status_enum = "stopped"
    print(f"Status: {status_enum}")

def generate_summary_report(results, output_dir):
    """Generate a comprehensive summary report with visualizations."""
    print("Generating summary report...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ADSS 2.0 Experiment Summary', fontsize=16, fontweight='bold')
    
    exp1_results = results['experiment_1']
    methods = list(exp1_results.keys())
    avg_steps = [exp1_results[m]['avg_steps'] for m in methods]
    inference_times = [exp1_results[m]['inference_time'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0,0].bar(x - width/2, avg_steps, width, label='Avg Steps', alpha=0.8)
    axes[0,0].bar(x + width/2, [t*100 for t in inference_times], width, label='Time (×100s)', alpha=0.8)
    axes[0,0].set_xlabel('Method')
    axes[0,0].set_ylabel('Value')
    axes[0,0].set_title('Inference Performance Comparison')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(methods)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    fid_scores = [exp1_results[m]['FID'] for m in methods]
    axes[0,1].bar(methods, fid_scores, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,1].set_xlabel('Method')
    axes[0,1].set_ylabel('FID Score')
    axes[0,1].set_title('Image Quality Comparison (FID)')
    axes[0,1].grid(True, alpha=0.3)
    
    exp2_results = results['experiment_2']
    variants = list(exp2_results.keys())
    variant_steps = [exp2_results[v]['avg_steps'] for v in variants]
    variant_fids = [exp2_results[v]['FID'] for v in variants]
    
    axes[1,0].bar(variants, variant_steps, color=['green', 'orange', 'red'], alpha=0.8)
    axes[1,0].set_xlabel('ADSS Variant')
    axes[1,0].set_ylabel('Average Steps')
    axes[1,0].set_title('Component Ablation: Steps')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    all_methods = methods + variants
    all_steps = avg_steps + variant_steps
    all_fids = fid_scores + variant_fids
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (method, steps, fid) in enumerate(zip(all_methods, all_steps, all_fids)):
        axes[1,1].scatter(steps, fid, s=100, c=colors[i % len(colors)], 
                         alpha=0.7, label=method)
    
    axes[1,1].set_xlabel('Average Steps')
    axes[1,1].set_ylabel('FID Score')
    axes[1,1].set_title('Quality vs Efficiency Trade-off')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'experiment_summary.pdf')
    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    main()
