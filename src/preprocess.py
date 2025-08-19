#!/usr/bin/env python3
"""
Data preprocessing module for ADSS 2.0 experiment.
Handles dataset loading and environment setup.
"""

import os
import torch
import numpy as np
from typing import List, Tuple

def setup_experiment_environment() -> torch.device:
    """
    Setup the experiment environment and return the appropriate device.
    Optimized for NVIDIA Tesla T4 with 16GB VRAM.
    """
    print("Setting up experiment environment...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
            
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    return device

def load_data(dataset_name: str = 'CIFAR10') -> List[torch.Tensor]:
    """
    Load dataset for the experiment.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        List of image tensors
    """
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == 'CIFAR10':
        num_samples = 100  # Reduced for testing
        image_size = (3, 32, 32)
        
        dataset = []
        for i in range(num_samples):
            if i % 3 == 0:
                image = torch.randn(image_size) * 0.3
            elif i % 3 == 1:
                image = torch.randn(image_size) * 0.6
            else:
                image = torch.randn(image_size) * 1.0
            
            image = torch.clamp((image + 1) / 2, 0, 1)
            dataset.append(image)
        
        print(f"Generated {len(dataset)} synthetic {dataset_name} samples")
        
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    return dataset

def preprocess_batch(images: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Preprocess a batch of images for diffusion model input.
    
    Args:
        images: List of image tensors
        device: Target device
        
    Returns:
        Preprocessed batch tensor
    """
    batch = torch.stack(images).to(device)
    
    batch = batch * 2.0 - 1.0
    
    return batch

def create_noise_schedule(num_timesteps: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create noise schedule for diffusion process.
    
    Args:
        num_timesteps: Number of diffusion timesteps
        
    Returns:
        Tuple of (betas, alphas_cumprod)
    """
    beta_start = 0.0001
    beta_end = 0.02
    
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return betas, alphas_cumprod

def add_noise(images: torch.Tensor, timesteps: torch.Tensor, 
              alphas_cumprod: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to images according to diffusion schedule.
    
    Args:
        images: Clean images
        timesteps: Timestep for each image
        alphas_cumprod: Cumulative alpha values
        
    Returns:
        Tuple of (noisy_images, noise)
    """
    noise = torch.randn_like(images)
    
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    
    noisy_images = torch.sqrt(alpha_t) * images + torch.sqrt(1 - alpha_t) * noise
    
    return noisy_images, noise

def compute_sample_complexity(image: torch.Tensor) -> float:
    """
    Compute a complexity score for an image to simulate different convergence rates.
    
    Args:
        image: Input image tensor
        
    Returns:
        Complexity score (higher = more complex)
    """
    grad_x = torch.abs(image[:, :, 1:] - image[:, :, :-1])
    grad_y = torch.abs(image[:, 1:, :] - image[:, :-1, :])
    
    complexity = (grad_x.mean() + grad_y.mean()).item()
    return complexity
