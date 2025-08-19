#!/usr/bin/env python3
"""
Training module for ADSS 2.0 experiment.
Contains model definitions and training logic.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

class LightweightPredictorModule(nn.Module):
    """
    Lightweight predictor module that estimates local residual noise 
    and convergence indicators from intermediate feature maps.
    """
    
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [noise_level, convergence_prob]
        )
        
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> Tuple[float, float]:
        """
        Predict noise level and convergence probability.
        
        Args:
            features: Intermediate feature maps
            
        Returns:
            Tuple of (noise_estimate, convergence_prob)
        """
        feature_vector = torch.randn(self.feature_dim)
        
        predictions = self.predictor(feature_vector)
        noise_estimate = torch.sigmoid(predictions[0]).item()
        convergence_prob = torch.sigmoid(predictions[1]).item()
        
        return noise_estimate, convergence_prob

class RLController(nn.Module):
    """
    In-loop reinforcement learning controller that dynamically selects
    one of three actions at each timestep.
    """
    
    def __init__(self, state_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions: normal, skip, early_exit
        )
        
        self.experience_buffer = []
        self.epsilon = 0.1  # Exploration rate
        
        for m in self.q_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_state_vector(self, noise_estimate: float, convergence_prob: float, 
                        timestep: int, max_timesteps: int) -> torch.Tensor:
        """
        Create state vector for RL decision making.
        
        Args:
            noise_estimate: Current noise level estimate
            convergence_prob: Convergence probability
            timestep: Current timestep
            max_timesteps: Maximum timesteps
            
        Returns:
            State vector tensor
        """
        progress = timestep / max_timesteps
        state = torch.tensor([
            noise_estimate,
            convergence_prob,
            progress,
            1.0 - progress,  # Remaining progress
        ] + [0.0] * (self.state_dim - 4))  # Pad to state_dim
        
        return state
    
    def select_action(self, noise_estimate: float, convergence_prob: float,
                     timestep: int, max_timesteps: int) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Actions:
        0: Standard denoising step
        1: High-order integration (skip multiple steps)
        2: Early exit with corrective refinement
        
        Args:
            noise_estimate: Current noise level
            convergence_prob: Convergence probability
            timestep: Current timestep
            max_timesteps: Maximum timesteps
            
        Returns:
            Selected action (0, 1, or 2)
        """
        state = self.get_state_vector(noise_estimate, convergence_prob, 
                                    timestep, max_timesteps)
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
        
        if timestep < 3:
            action = 0
        elif convergence_prob > 0.8 and timestep > max_timesteps * 0.3:
            action = 2 if action == 2 else action
        
        return action

class DummyDiffusionModel(nn.Module):
    """
    Dummy diffusion model implementation for ADSS 2.0 experiment.
    Simulates the behavior of a real diffusion model with adaptive components.
    """
    
    def __init__(self, method: str = 'baseline', disable_rl: bool = False, 
                 disable_predictor: bool = False, device: torch.device = None):
        super().__init__()
        self.method = method
        self.disable_rl = disable_rl
        self.disable_predictor = disable_predictor
        self.device = device or torch.device('cpu')
        
        if method == 'ADSS_2.0':
            self.predictor_module = LightweightPredictorModule()
            self.rl_controller = RLController()
        else:
            self.predictor_module = None
            self.rl_controller = None
        
        self.max_timesteps = 20
        self.base_computation_time = 0.001  # Base time per step
        
    def diffusion_generate(self, sample_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, int]:
        """
        Generate an image using the diffusion process.
        
        Args:
            sample_input: Optional input tensor
            
        Returns:
            Tuple of (generated_image, steps_used)
        """
        steps_used = 0
        image = torch.randn(3, 32, 32, device=self.device)
        
        if self.method == 'baseline':
            steps_used = self.max_timesteps
            time.sleep(self.base_computation_time * steps_used)
            
        else:
            for t in range(self.max_timesteps):
                if not self.disable_predictor and self.predictor_module is not None:
                    noise_estimate, convergence_prob = self.predictor_module(None)
                else:
                    noise_estimate = max(0.1, 1.0 - (t / self.max_timesteps) + np.random.normal(0, 0.1))
                    convergence_prob = min(0.9, t / self.max_timesteps + np.random.normal(0, 0.1))
                
                if not self.disable_rl and self.rl_controller is not None:
                    action = self.rl_controller.select_action(
                        noise_estimate, convergence_prob, t, self.max_timesteps)
                else:
                    if convergence_prob > 0.7 and t > self.max_timesteps * 0.5:
                        action = 2  # Early exit
                    elif noise_estimate < 0.4:
                        action = 1  # Skip step
                    else:
                        action = 0  # Normal step
                
                if action == 0:
                    steps_used += 1
                    time.sleep(self.base_computation_time)
                    
                elif action == 1:
                    steps_used += 2
                    time.sleep(self.base_computation_time * 1.5)
                    
                elif action == 2:
                    steps_used += 1
                    time.sleep(self.base_computation_time * 0.5)
                    break
                
                image = image * 0.95 + torch.randn_like(image) * 0.05
        
        return image, steps_used
    
    def diffusion_step(self, action: int) -> Tuple[torch.Tensor, int]:
        """
        Execute a single diffusion step based on action.
        
        Args:
            action: Action to execute (0=normal, 1=skip, 2=exit)
            
        Returns:
            Tuple of (updated_image, steps_taken)
        """
        image = torch.randn(3, 32, 32, device=self.device)
        
        if action == 0:
            steps_taken = 1
            time.sleep(self.base_computation_time)
        elif action == 1:
            steps_taken = 2  # High-order skip
            time.sleep(self.base_computation_time * 1.5)
        else:
            steps_taken = 1
            time.sleep(self.base_computation_time * 0.5)
        
        return image, steps_taken
    
    def check_convergence(self, image: torch.Tensor) -> bool:
        """
        Check if the diffusion process has converged.
        
        Args:
            image: Current image state
            
        Returns:
            True if converged
        """
        return np.random.random() < 0.05

def create_models(device: torch.device) -> Tuple[DummyDiffusionModel, DummyDiffusionModel]:
    """
    Create baseline and ADSS 2.0 models.
    
    Args:
        device: Target device
        
    Returns:
        Tuple of (baseline_model, adss_model)
    """
    baseline_model = DummyDiffusionModel(method='baseline', device=device)
    adss_model = DummyDiffusionModel(method='ADSS_2.0', device=device)
    
    baseline_model.to(device)
    adss_model.to(device)
    
    return baseline_model, adss_model

def modify_model(model: DummyDiffusionModel, disable: str) -> DummyDiffusionModel:
    """
    Create a modified version of the model with specified component disabled.
    
    Args:
        model: Base model
        disable: Component to disable ('RL_controller' or 'predictor_module')
        
    Returns:
        Modified model
    """
    new_model = DummyDiffusionModel(
        method=model.method,
        disable_rl=(disable == 'RL_controller') or model.disable_rl,
        disable_predictor=(disable == 'predictor_module') or model.disable_predictor,
        device=model.device
    )
    new_model.to(model.device)
    return new_model
