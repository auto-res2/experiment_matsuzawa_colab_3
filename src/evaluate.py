#!/usr/bin/env python3
"""
Evaluation module for ADSS 2.0 experiment.
Contains all three experiments: benchmarking, ablation, and dynamic analysis.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Any, Tuple
from train import DummyDiffusionModel, modify_model

def compute_FID(generated_images: List[torch.Tensor], real_images: List[torch.Tensor]) -> float:
    """
    Compute Fréchet Inception Distance between generated and real images.
    Simplified implementation for demonstration.
    
    Args:
        generated_images: List of generated image tensors
        real_images: List of real image tensors
        
    Returns:
        FID score
    """
    gen_mean = np.mean([img.mean().item() for img in generated_images])
    real_mean = np.mean([img.mean().item() for img in real_images])
    
    gen_std = np.std([img.std().item() for img in generated_images])
    real_std = np.std([img.std().item() for img in real_images])
    
    fid = abs(gen_mean - real_mean) * 100 + abs(gen_std - real_std) * 50
    return float(fid)

def compute_IS(generated_images: List[torch.Tensor]) -> float:
    """
    Compute Inception Score for generated images.
    Simplified implementation for demonstration.
    
    Args:
        generated_images: List of generated image tensors
        
    Returns:
        IS score
    """
    scores = []
    for img in generated_images:
        score = torch.softmax(torch.randn(1000), dim=0)
        entropy = -torch.sum(score * torch.log(score + 1e-8))
        scores.append(entropy.item())
    
    is_score = np.exp(np.mean(scores))
    return is_score

def run_inference_benchmark(baseline_model: DummyDiffusionModel, 
                          adss_model: DummyDiffusionModel,
                          dataset: List[torch.Tensor],
                          output_dir: str,
                          num_samples: int = 50) -> Dict[str, Dict[str, float]]:
    """
    Run Experiment 1: Inference Speed and Quality Benchmarking.
    
    Args:
        baseline_model: Baseline diffusion model
        adss_model: ADSS 2.0 model
        dataset: Dataset for quality comparison
        output_dir: Directory to save results
        num_samples: Number of samples to generate
        
    Returns:
        Dictionary of results for each method
    """
    print(f"Running inference benchmark with {num_samples} samples...")
    
    results = {}
    
    for method, model in [('baseline', baseline_model), ('ADSS_2.0', adss_model)]:
        print(f"  Testing {method}...")
        
        start_time = time.time()
        step_counts = []
        generated_images = []
        
        for i in range(num_samples):
            if i % 10 == 0:
                print(f"    Sample {i+1}/{num_samples}")
            
            image, steps_used = model.diffusion_generate(sample_input=None)
            generated_images.append(image)
            step_counts.append(steps_used)
        
        elapsed_time = time.time() - start_time
        fid_score = compute_FID(generated_images, dataset[:num_samples])
        is_score = compute_IS(generated_images)
        
        results[method] = {
            'avg_steps': np.mean(step_counts),
            'std_steps': np.std(step_counts),
            'inference_time': elapsed_time,
            'avg_time_per_sample': elapsed_time / num_samples,
            'FID': fid_score,
            'IS': is_score,
            'step_counts': step_counts
        }
        
        print(f"    {method} completed: {np.mean(step_counts):.2f} avg steps, "
              f"{elapsed_time:.3f}s total, FID: {fid_score:.3f}")
    
    create_benchmark_plots(results, output_dir)
    
    return results

def create_benchmark_plots(results: Dict[str, Dict[str, float]], output_dir: str):
    """Create visualization plots for benchmark results."""
    
    methods = list(results.keys())
    avg_steps = [results[m]['avg_steps'] for m in methods]
    inference_times = [results[m]['avg_time_per_sample'] for m in methods]
    fid_scores = [results[m]['FID'] for m in methods]
    is_scores = [results[m]['IS'] for m in methods]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ADSS 2.0 vs Baseline: Inference Benchmark', fontsize=14, fontweight='bold')
    
    bars1 = axes[0,0].bar(methods, avg_steps, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,0].set_ylabel('Average Steps')
    axes[0,0].set_title('Computational Efficiency')
    axes[0,0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars1, avg_steps):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{val:.1f}', ha='center', va='bottom')
    
    bars2 = axes[0,1].bar(methods, inference_times, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,1].set_ylabel('Time per Sample (s)')
    axes[0,1].set_title('Wall-clock Performance')
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, inference_times):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{val:.3f}', ha='center', va='bottom')
    
    bars3 = axes[1,0].bar(methods, fid_scores, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1,0].set_ylabel('FID Score (lower is better)')
    axes[1,0].set_title('Image Quality (FID)')
    axes[1,0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, fid_scores):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{val:.2f}', ha='center', va='bottom')
    
    baseline_steps = results['baseline']['step_counts']
    adss_steps = results['ADSS_2.0']['step_counts']
    
    axes[1,1].hist(baseline_steps, alpha=0.6, label='Baseline', bins=10, color='skyblue')
    axes[1,1].hist(adss_steps, alpha=0.6, label='ADSS 2.0', bins=10, color='lightcoral')
    axes[1,1].set_xlabel('Number of Steps')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Step Count Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'inference_benchmark.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Benchmark plot saved: {plot_path}")

def run_component_ablation(adss_model: DummyDiffusionModel,
                         dataset: List[torch.Tensor],
                         output_dir: str,
                         num_samples: int = 30) -> Dict[str, Dict[str, float]]:
    """
    Run Experiment 2: Component Ablation Study.
    
    Args:
        adss_model: Base ADSS 2.0 model
        dataset: Dataset for quality comparison
        output_dir: Directory to save results
        num_samples: Number of samples to generate
        
    Returns:
        Dictionary of results for each variant
    """
    print(f"Running component ablation study with {num_samples} samples...")
    
    variants = {
        'Full_ADSS': adss_model,
        'No_RL': modify_model(adss_model, disable='RL_controller'),
        'No_Predictor': modify_model(adss_model, disable='predictor_module')
    }
    
    results = {}
    
    for variant_name, variant_model in variants.items():
        print(f"  Testing {variant_name}...")
        
        start_time = time.time()
        step_counts = []
        generated_images = []
        
        for i in range(num_samples):
            if i % 10 == 0:
                print(f"    Sample {i+1}/{num_samples}")
            
            image, steps_used = variant_model.diffusion_generate(sample_input=None)
            generated_images.append(image)
            step_counts.append(steps_used)
        
        elapsed_time = time.time() - start_time
        fid_score = compute_FID(generated_images, dataset[:num_samples])
        is_score = compute_IS(generated_images)
        
        results[variant_name] = {
            'avg_steps': np.mean(step_counts),
            'std_steps': np.std(step_counts),
            'inference_time': elapsed_time,
            'avg_time_per_sample': elapsed_time / num_samples,
            'FID': fid_score,
            'IS': is_score,
            'step_counts': step_counts
        }
        
        print(f"    {variant_name} completed: {np.mean(step_counts):.2f} avg steps, "
              f"FID: {fid_score:.3f}")
    
    create_ablation_plots(results, output_dir)
    
    return results

def create_ablation_plots(results: Dict[str, Dict[str, float]], output_dir: str):
    """Create visualization plots for ablation study results."""
    
    variants = list(results.keys())
    avg_steps = [results[v]['avg_steps'] for v in variants]
    inference_times = [results[v]['avg_time_per_sample'] for v in variants]
    fid_scores = [results[v]['FID'] for v in variants]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('ADSS 2.0 Component Ablation Study', fontsize=14, fontweight='bold')
    
    colors = ['green', 'orange', 'red']
    
    bars1 = axes[0,0].bar(variants, avg_steps, color=colors, alpha=0.8)
    axes[0,0].set_ylabel('Average Steps')
    axes[0,0].set_title('Computational Efficiency by Component')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars1, avg_steps):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{val:.1f}', ha='center', va='bottom')
    
    bars2 = axes[0,1].bar(variants, inference_times, color=colors, alpha=0.8)
    axes[0,1].set_ylabel('Time per Sample (s)')
    axes[0,1].set_title('Wall-clock Performance by Component')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars2, inference_times):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{val:.3f}', ha='center', va='bottom')
    
    bars3 = axes[1,0].bar(variants, fid_scores, color=colors, alpha=0.8)
    axes[1,0].set_ylabel('FID Score (lower is better)')
    axes[1,0].set_title('Image Quality by Component')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    for bar, val in zip(bars3, fid_scores):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      f'{val:.2f}', ha='center', va='bottom')
    
    axes[1,1].scatter(avg_steps, fid_scores, c=colors, s=100, alpha=0.7)
    for i, variant in enumerate(variants):
        axes[1,1].annotate(variant, (avg_steps[i], fid_scores[i]), 
                          xytext=(5, 5), textcoords='offset points')
    axes[1,1].set_xlabel('Average Steps')
    axes[1,1].set_ylabel('FID Score')
    axes[1,1].set_title('Efficiency vs Quality Trade-off')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'component_ablation.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Ablation plot saved: {plot_path}")

def generate_with_logging(model: DummyDiffusionModel, log_actions: bool = True) -> Tuple[torch.Tensor, Dict[str, List]]:
    """
    Generate a single sample with detailed logging of internal decisions.
    
    Args:
        model: ADSS 2.0 model
        log_actions: Whether to log RL actions
        
    Returns:
        Tuple of (generated_image, logs_dict)
    """
    logs = {
        'timesteps': [],
        'residual_noise': [],
        'convergence_probs': [],
        'rl_actions': [],
        'action_names': []
    }
    
    action_names = ['Normal Step', 'High-order Skip', 'Early Exit']
    
    image = torch.randn(3, 32, 32, device=model.device)
    total_steps = 0
    max_timesteps = model.max_timesteps
    
    for t in range(max_timesteps):
        if not model.disable_predictor and model.predictor_module is not None:
            noise_estimate, convergence_prob = model.predictor_module(None)
        else:
            noise_estimate = max(0.1, 1.0 - (t / max_timesteps) + np.random.normal(0, 0.1))
            convergence_prob = min(0.9, t / max_timesteps + np.random.normal(0, 0.1))
        
        if log_actions and (not model.disable_rl and model.rl_controller is not None):
            action = model.rl_controller.select_action(
                noise_estimate, convergence_prob, t, max_timesteps)
        else:
            if convergence_prob > 0.7 and t > max_timesteps * 0.5:
                action = 2
            elif noise_estimate < 0.4:
                action = 1
            else:
                action = 0
        
        logs['timesteps'].append(t)
        logs['residual_noise'].append(noise_estimate)
        logs['convergence_probs'].append(convergence_prob)
        logs['rl_actions'].append(action)
        logs['action_names'].append(action_names[action])
        
        image, step_taken = model.diffusion_step(action)
        total_steps += step_taken
        
        if action == 2 or model.check_convergence(image):
            break
    
    return image, logs

def run_dynamic_analysis(adss_model: DummyDiffusionModel,
                        output_dir: str,
                        num_samples: int = 10) -> List[Dict[str, List]]:
    """
    Run Experiment 3: Dynamic Step Skipping Analysis and Visualization.
    
    Args:
        adss_model: ADSS 2.0 model
        output_dir: Directory to save results
        num_samples: Number of samples to analyze
        
    Returns:
        List of log dictionaries for each sample
    """
    print(f"Running dynamic step skipping analysis with {num_samples} samples...")
    
    sample_logs = []
    
    for i in range(num_samples):
        print(f"  Analyzing sample {i+1}/{num_samples}")
        
        _, logs = generate_with_logging(adss_model, log_actions=True)
        sample_logs.append(logs)
        
        print(f"    Sample {i+1}: {len(logs['timesteps'])} timesteps, "
              f"Actions: {logs['action_names']}")
    
    create_dynamic_analysis_plots(sample_logs, output_dir)
    
    return sample_logs

def create_dynamic_analysis_plots(sample_logs: List[Dict[str, List]], output_dir: str):
    """Create visualization plots for dynamic analysis results."""
    
    example_log = sample_logs[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ADSS 2.0 Dynamic Step Skipping Analysis', fontsize=14, fontweight='bold')
    
    axes[0,0].plot(example_log['timesteps'], example_log['residual_noise'], 
                   marker='o', linewidth=2, markersize=6, color='blue')
    axes[0,0].set_xlabel('Timestep')
    axes[0,0].set_ylabel('Residual Noise Estimate')
    axes[0,0].set_title('Local Residual Noise Evolution')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_ylim(0, 1)
    
    axes[0,1].plot(example_log['timesteps'], example_log['convergence_probs'], 
                   marker='s', linewidth=2, markersize=6, color='green')
    axes[0,1].set_xlabel('Timestep')
    axes[0,1].set_ylabel('Convergence Probability')
    axes[0,1].set_title('Convergence Probability Evolution')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_ylim(0, 1)
    
    action_colors = ['skyblue', 'orange', 'red']
    action_names = ['Normal Step', 'High-order Skip', 'Early Exit']
    
    for i, (timestep, action) in enumerate(zip(example_log['timesteps'], example_log['rl_actions'])):
        axes[1,0].bar(timestep, 1, color=action_colors[action], alpha=0.8, width=0.8)
    
    axes[1,0].set_xlabel('Timestep')
    axes[1,0].set_ylabel('Action Type')
    axes[1,0].set_title('RL Controller Actions over Time')
    axes[1,0].set_yticks([0.5])
    axes[1,0].set_yticklabels(['Actions'])
    
    legend_elements = [Rectangle((0,0),1,1, facecolor=action_colors[i], alpha=0.8, label=action_names[i]) 
                      for i in range(3)]
    axes[1,0].legend(handles=legend_elements, loc='upper right')
    axes[1,0].grid(True, alpha=0.3)
    
    all_actions = []
    for logs in sample_logs:
        all_actions.extend(logs['rl_actions'])
    
    action_counts = [all_actions.count(i) for i in range(3)]
    bars = axes[1,1].bar(action_names, action_counts, color=action_colors, alpha=0.8)
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Action Distribution Across All Samples')
    axes[1,1].grid(True, alpha=0.3)
    
    for bar, count in zip(bars, action_counts):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                      str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'dynamic_step_skipping.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Dynamic analysis plot saved: {plot_path}")
    
    create_step_efficiency_plot(sample_logs, output_dir)

def create_step_efficiency_plot(sample_logs: List[Dict[str, List]], output_dir: str):
    """Create a plot showing step efficiency across samples."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Step Efficiency Analysis', fontsize=14, fontweight='bold')
    
    sample_lengths = [len(logs['timesteps']) for logs in sample_logs]
    avg_noise_levels = [np.mean(logs['residual_noise']) for logs in sample_logs]
    
    axes[0].hist(sample_lengths, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Number of Timesteps')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Sample Lengths')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(np.mean(sample_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sample_lengths):.1f}')
    axes[0].legend()
    
    axes[1].scatter(avg_noise_levels, sample_lengths, alpha=0.7, s=60, color='coral')
    axes[1].set_xlabel('Average Noise Level')
    axes[1].set_ylabel('Number of Timesteps')
    axes[1].set_title('Noise Level vs Sample Complexity')
    axes[1].grid(True, alpha=0.3)
    
    z = np.polyfit(avg_noise_levels, sample_lengths, 1)
    p = np.poly1d(z)
    sorted_noise = sorted([float(x) for x in avg_noise_levels])
    axes[1].plot(sorted_noise, p(np.array(sorted_noise)), 
                "r--", alpha=0.8, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')
    axes[1].legend()
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'step_efficiency_analysis.pdf')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"  Step efficiency plot saved: {plot_path}")
