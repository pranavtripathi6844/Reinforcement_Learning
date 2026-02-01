"""Implementation of SimOpt (Simulation Optimization) for adaptive domain randomization.
This implementation uses Bayesian optimization to find optimal randomization parameters.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import json
import os
from typing import Dict, Tuple, List, Callable
import torch

class SimOpt:
    def __init__(self, 
                 param_ranges: Dict[str, Tuple[float, float]],
                 n_initial_points: int = 5,
                 n_iterations: int = 20,
                 save_dir: str = "./simopt_results",
                 checkpoint_interval: int = 5,
                 resume_from_checkpoint: bool = False):
        """
        Initialize SimOpt optimizer.
        
        Args:
            param_ranges: Dictionary of parameter ranges for each body part
            n_initial_points: Number of initial random points to sample
            n_iterations: Number of optimization iterations
            save_dir: Directory to save optimization results
            checkpoint_interval: Save checkpoint every N trials (0 = no checkpoints)
            resume_from_checkpoint: If True, load progress from existing checkpoint
        """
        self.param_ranges = param_ranges
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.save_dir = save_dir
        self.checkpoint_interval = checkpoint_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize empty history
        self.history = {
            'iterations': [],
            'params': [],
            'rewards': [],
            'best_reward': float('-inf'),
            'best_params': None
        }
        
        # Initialize or load optimization history
        if resume_from_checkpoint:
            print("🔄 Resuming from checkpoint...")
            try:
                self.load_progress()
                if self.history['iterations']:
                    print(f"✅ Loaded {len(self.history['iterations'])} completed trials")
                    print(f"   Best reward so far: {self.history['best_reward']:.2f}")
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
                print("Starting from scratch...")
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10
        )
        
    def _sample_random_params(self) -> Dict[str, Tuple[float, float]]:
        """Sample random parameters within the specified ranges, ensuring safe, positive bounds."""
        params = {}
        safe_min = 0.5
        safe_max = 1.5
        for param, (min_val, max_val) in self.param_ranges.items():
            # Sample a random factor between min_val and max_val
            factor = np.random.uniform(min_val, max_val)
            lower = max(safe_min, 1.0 - factor)
            upper = min(safe_max, 1.0 + factor)
            # Ensure lower < upper
            if lower >= upper:
                lower = safe_min
                upper = safe_max
            params[param] = (lower, upper)
        return params
    
    def _update_params(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Update parameters using Bayesian optimization."""
        # Fit GP to current data
        self.gp.fit(X, y)
        
        # Generate candidate points
        n_candidates = 1000
        candidates = []
        for _ in range(n_candidates):
            candidate = self._sample_random_params()
            # Convert to feature vector
            feature_vector = self._params_to_vector(candidate)
            candidates.append(feature_vector)
        candidates = np.array(candidates)
        
        # Predict mean and uncertainty
        mean, std = self.gp.predict(candidates, return_std=True)
        
        # Use Upper Confidence Bound (UCB) acquisition function
        ucb = mean + 2.0 * std
        
        # Select best candidate
        best_idx = np.argmax(ucb)
        best_candidate = candidates[best_idx]
        
        # Convert back to parameter dictionary
        return self._vector_to_params(best_candidate)
    
    def _params_to_vector(self, params: Dict[str, Tuple[float, float]]) -> np.ndarray:
        """Convert parameter dictionary to feature vector."""
        vector = []
        for param in sorted(self.param_ranges.keys()):
            min_val, max_val = params[param]
            vector.extend([min_val, max_val])
        return np.array(vector)
    
    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Convert feature vector back to parameter dictionary."""
        params = {}
        for i, param in enumerate(sorted(self.param_ranges.keys())):
            idx = i * 2
            params[param] = (vector[idx], vector[idx + 1])
        return params
    
    def optimize(self, 
                train_fn: Callable[[Dict[str, Tuple[float, float]]], object],
                eval_fn: Callable[[object], float]) -> Dict[str, Tuple[float, float]]:
        """
        Main optimization loop.
        
        Args:
            train_fn: Function to train model with given parameters
            eval_fn: Function to evaluate model on target domain
            
        Returns:
            Best parameters found
        """
        print("Starting SimOpt optimization...")
        print(f"Initial parameter ranges: {self.param_ranges}")
        
        # Reconstruct X, y from checkpoint if resuming
        if len(self.history['iterations']) > 0:
            print(f"🔄 Resuming from {len(self.history['iterations'])} completed trials")
            X = [self._params_to_vector(p) for p in self.history['params']]
            y = self.history['rewards'].copy()
            completed_trials = len(self.history['iterations'])
        else:
            X = []
            y = []
            completed_trials = 0
        
        # Initial random sampling (skip already completed)
        for i in range(completed_trials, self.n_initial_points):
            print(f"\nInitial sampling iteration {i+1}/{self.n_initial_points}")
            params = self._sample_random_params()
            print(f"Sampled parameters: {params}")
            
            # Train and evaluate
            model = train_fn(params)
            reward = eval_fn(model)
            
            # Store results
            X.append(self._params_to_vector(params))
            y.append(reward)
            
            # Update history
            self.history['iterations'].append(i)
            self.history['params'].append(params)
            self.history['rewards'].append(reward)
            
            print(f"Reward: {reward:.2f}")
            
            # Update best parameters
            if reward > self.history['best_reward']:
                self.history['best_reward'] = reward
                self.history['best_params'] = params
                print(f"New best reward: {reward:.2f}")
                print(f"New best parameters: {params}")
            
            # Save checkpoint every N trials
            if self.checkpoint_interval > 0 and (i + 1) % self.checkpoint_interval == 0:
                print(f"💾 Saving checkpoint at trial {i+1}")
                self.save_progress()
        
        # Bayesian optimization loop
        X = np.array(X)
        y = np.array(y)
        
        for i in range(self.n_iterations):
            print(f"\nOptimization iteration {i+1}/{self.n_iterations}")
            
            # Update parameters
            params = self._update_params(X, y)
            print(f"Updated parameters: {params}")
            
            # Train and evaluate
            model = train_fn(params)
            reward = eval_fn(model)
            
            # Store results
            X = np.vstack([X, self._params_to_vector(params)])
            y = np.append(y, reward)
            
            # Update history
            iteration = self.n_initial_points + i
            self.history['iterations'].append(iteration)
            self.history['params'].append(params)
            self.history['rewards'].append(reward)
            
            print(f"Reward: {reward:.2f}")
            
            # Update best parameters
            if reward > self.history['best_reward']:
                self.history['best_reward'] = reward
                self.history['best_params'] = params
                print(f"New best reward: {reward:.2f}")
                print(f"New best parameters: {params}")
            
            # Save checkpoint every N trials
            trial_num = self.n_initial_points + i + 1
            if self.checkpoint_interval > 0 and trial_num % self.checkpoint_interval == 0:
                checkpoint_file = os.path.join(self.save_dir, f'checkpoint_trial_{trial_num}.json')
                print(f"💾 Saving checkpoint at trial {trial_num} to {checkpoint_file}")
                self.save_progress()
        
        return self.history['best_params']
    
    def save_progress(self):
        """Save optimization progress to file."""
        # Convert numpy arrays to lists for JSON serialization
        history = {
            'iterations': self.history['iterations'],
            'params': self.history['params'],
            'rewards': [float(r) for r in self.history['rewards']],
            'best_reward': float(self.history['best_reward']),
            'best_params': self.history['best_params']
        }
        
        # Save to file
        with open(os.path.join(self.save_dir, 'optimization_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
    
    def load_progress(self):
        """Load optimization progress from file."""
        history_file = os.path.join(self.save_dir, 'optimization_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.history = json.load(f) 