"""Train SAC agent using SimOpt (Simulation Optimization) for adaptive domain randomization"""
import argparse
import os
import torch
import gym
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import json
from env.custom_hopper import *
from simopt import SimOpt


class TargetEvalCallback(BaseCallback):
    """Custom callback with GPU-accelerated vectorized evaluation on target environment."""
    
    def __init__(self, eval_freq=10000, n_eval_episodes=50, save_path="./best_model_simopt", verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        os.makedirs(save_path, exist_ok=True)
        
        # Create vectorized target environment for GPU-accelerated evaluation
        # Use 8 parallel envs for evaluation (faster than sequential)
        n_eval_envs = min(8, n_eval_episodes)
        def make_target_env():
            return gym.make('CustomHopper-target-v0')
        
        self.eval_env = DummyVecEnv([make_target_env for _ in range(n_eval_envs)])
        self.n_eval_envs = n_eval_envs
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # GPU-accelerated vectorized evaluation
            episode_rewards = []
            episodes_completed = 0
            
            obs = self.eval_env.reset()
            episode_reward = np.zeros(self.n_eval_envs)
            
            while episodes_completed < self.n_eval_episodes:
                # Predict actions for all envs simultaneously (GPU accelerated)
                actions, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, _ = self.eval_env.step(actions)
                episode_reward += rewards
                
                # Check for completed episodes
                for i, done in enumerate(dones):
                    if done and episodes_completed < self.n_eval_episodes:
                        episode_rewards.append(episode_reward[i])
                        episode_reward[i] = 0
                        episodes_completed += 1
            
            mean_reward = np.mean(episode_rewards[:self.n_eval_episodes])
            
            if self.verbose > 0:
                print(f"Target eval: {mean_reward:.2f} ± {np.std(episode_rewards[:self.n_eval_episodes]):.2f}")
            
            # Save if best
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                model_path = os.path.join(self.save_path, "best_model.zip")
                self.model.save(model_path)
                if self.verbose > 0:
                    print(f"New best target reward: {mean_reward:.2f} - Model saved!")
        
        return True

def load_best_params(file_path):
    """Load best parameters from JSON file."""
    with open(file_path, 'r') as f:
        params = json.load(f)
    print(f"Loaded best parameters from {file_path}:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    return params

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent using SimOpt for adaptive domain randomization')
    parser.add_argument('--episodes', type=int, default=2000,
                      help='Number of training episodes (default: 2000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate (default: 0.0003)')
    parser.add_argument('--mass_variation', type=float, default=0.3,
                      help='Mass variation range for UDR (e.g., 0.3 for ±30%)')
    parser.add_argument('--n-initial-points', type=int, default=5,
                      help='Number of initial random points (default: 5)')
    parser.add_argument('--n-iterations', type=int, default=20,
                      help='Number of optimization iterations (default: 20)')
    parser.add_argument('--eval-episodes', type=int, default=50,
                      help='Number of episodes for evaluation (default: 50)')
    parser.add_argument('--load_best_params', type=str, default=None,
                      help='Load best parameters from file (e.g., best_sac_params.json)')
    parser.add_argument('--n_envs', type=int, default=16,
                      help='Number of parallel environments (default: 16)')
    return parser.parse_args()

def train_with_params(params, args, best_params=None):
    """Train model with specific randomization parameters."""
    # Auto-detect device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create training environment with specified parameters
    def make_env():
        return gym.make('CustomHopper-source-v0',
                       use_udr=True,
                       mass_ranges=params)
    
    # Create parallel environments for training
    train_env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    # Create target evaluation callback (evaluates and saves based on target performance)
    target_eval_callback = TargetEvalCallback(
        eval_freq=10000,  # Evaluate every 10k steps
        n_eval_episodes=args.eval_episodes,
        save_path="./best_model_simopt",
        verbose=1
    )

    # Initialize SAC agent with best params or defaults
    if best_params:
        print(f"Using Optuna-optimized parameters")
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=best_params.get('learning_rate', 3e-4),
            buffer_size=best_params.get('buffer_size', 1000000),
            batch_size=best_params.get('batch_size', 256),
            gamma=best_params.get('gamma', 0.99),
            tau=0.005,
            learning_starts=1000,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            tensorboard_log="./logs_simopt/",
            verbose=1,
            device=device,
            policy_kwargs={"net_arch": [256, 256]}
        )
    else:
        print(f"Using default SAC parameters")
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            optimize_memory_usage=False,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            tensorboard_log="./logs_simopt/",
            verbose=1,
            device=device,
            policy_kwargs={"net_arch": [256, 256]}
        )

    # Calculate total timesteps
    max_steps_per_episode = 500
    total_timesteps = args.episodes * max_steps_per_episode

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=target_eval_callback,
        log_interval=1000
    )

    return model

def evaluate_on_target(model, n_episodes=50):
    """Evaluate model on target environment."""
    # Create target environment
    env = gym.make('CustomHopper-target-v0')
    env = DummyVecEnv([lambda: env])

    # Evaluate
    episode_rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
        
        episode_rewards.append(total_reward)
    
    # Calculate mean reward
    mean_reward = np.mean(episode_rewards)
    return mean_reward

def main():
    args = parse_args()
    
    # Auto-detect device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load optimized parameters if provided
    best_params = None
    if args.load_best_params:
        best_params = load_best_params(args.load_best_params)
    
    # Create directories
    os.makedirs("./best_model_simopt", exist_ok=True)
    os.makedirs("./logs_simopt", exist_ok=True)
    
    print("Training with SimOpt adaptive domain randomization")
    print(f"Mass variation: ±{args.mass_variation*100:.0f}%")
    
    # Define mass ranges for SimOpt
    mass_ranges = {
        'thigh': (1-args.mass_variation, 1+args.mass_variation),
        'leg': (1-args.mass_variation, 1+args.mass_variation),
        'foot': (1-args.mass_variation, 1+args.mass_variation)
    }
    
    import sys
    
    # Initialize SimOpt
    simopt = SimOpt(
        param_ranges=mass_ranges,
        n_initial_points=args.n_initial_points,
        n_iterations=args.n_iterations,
        save_dir="./simopt_results",
        checkpoint_interval=1,  # Save checkpoint after EVERY trial
        resume_from_checkpoint=True  # Always try to resume
    )
    
    # Run optimization
    best_mass_ranges = simopt.optimize(
        train_fn=lambda params: train_with_params(params, args, best_params),
        eval_fn=lambda model: evaluate_on_target(model, args.eval_episodes)
    )
    
    print("\nOptimization complete!")
    print(f"Best mass ranges found: {best_mass_ranges}")
    
    # Train final model with best parameters
    print("\nTraining final model with best mass ranges...")
    final_model = train_with_params(best_mass_ranges, args, best_params)
    
    # Save final model
    final_model_name = "./best_model_simopt/simopt_model"
    print(f"Saving final model as {final_model_name}...")
    final_model.save(final_model_name)
    print("Final model saved successfully!")

if __name__ == '__main__':
    main() 