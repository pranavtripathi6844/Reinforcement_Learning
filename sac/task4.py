"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
import numpy as np
import argparse
import os
import json
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC agent using Stable Baselines3')
    parser.add_argument('--episodes', type=int, default=2000,
                      help='Number of training episodes (default: 2000)')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help='Learning rate (default: 0.0003)')
    parser.add_argument('--use_udr', action='store_true',
                      help='Enable Uniform Domain Randomization during training')
    parser.add_argument('--mass_variation', type=float, default=0.3,
                      help='Mass variation range for UDR (e.g., 0.3 for ±30%)')
    parser.add_argument('--load_best_params', type=str, default=None,
                      help='Load best parameters from file (e.g., best_sac_params.json)')
    return parser.parse_args()

def load_best_params(params_file):
    """Load best parameters from file"""
    if not os.path.exists(params_file):
        print(f"Warning: {params_file} not found. Using default parameters.")
        return {}
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    print(f"Loaded optimized parameters from {params_file}:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    return params

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Auto-detect device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load optimized parameters if provided
    best_params = {}
    if args.load_best_params:
        best_params = load_best_params(args.load_best_params)
    
    try:
        # Set up mass ranges for UDR if enabled
        mass_ranges = None
        if args.use_udr:
            mass_ranges = {
                'thigh': (1-args.mass_variation, 1+args.mass_variation),
                'leg': (1-args.mass_variation, 1+args.mass_variation),
                'foot': (1-args.mass_variation, 1+args.mass_variation)
            }

        # Function to create environment
        def make_env():
            return gym.make('CustomHopper-source-v0',
                           use_udr=args.use_udr,
                           mass_ranges=mass_ranges)

        # Create the training environment with 16 parallel processes
        n_envs = 16
        print(f"Creating {n_envs} parallel environments for training...")
        train_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

        # Create single evaluation environment
        eval_env = DummyVecEnv([make_env])

        print('State space:', train_env.observation_space)  # state-space
        print('Action space:', train_env.action_space)  # action-space
        # Access parameters through env_method for SubprocVecEnv compatibility
        try:
            params = train_env.env_method('get_parameters')[0]
            print('Dynamics parameters:', params)
        except Exception:
            print('Dynamics parameters: (Parallel env - see logs for details)')
        if args.use_udr:
            print('UDR enabled with mass variation: ±{:.0f}%'.format(args.mass_variation * 100))

        # Create model directory with UDR indication and mass variation
        if args.use_udr:
            model_dir = f"./best_model_udr_{int(args.mass_variation*100)}"
            log_dir = f"./logs_udr_{int(args.mass_variation*100)}/"
        else:
            model_dir = "./best_model"
            log_dir = "./logs/"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=10000,
            deterministic=True,
            render=False
        )

        # Extract policy_kwargs if present in best_params
        policy_kwargs = {}
        if 'net_arch' in best_params:
            policy_kwargs['net_arch'] = best_params.pop('net_arch')
            print(f"Using optimized network architecture: {policy_kwargs['net_arch']}")

        # Initialize SAC agent with optimized parameters if available
        if best_params:
            # Use optimized parameters
            model = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=policy_kwargs,
                device=device,
                **best_params
            )
            print("Using optimized hyperparameters")
        else:
            # Use original parameters (your existing code)
            model = SAC(
                "MlpPolicy",
                train_env,
                learning_rate=args.learning_rate,
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
                tensorboard_log="./logs/",
                verbose=1,
                device=device
            )

        # Calculate total timesteps based on command line argument
        max_steps_per_episode = 500
        total_timesteps = args.episodes * max_steps_per_episode

        print(f"Training for {args.episodes} episodes ({total_timesteps} timesteps)")
        if args.use_udr:
            print("Training with UDR - masses will be randomized each episode")

        # Train the agent
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=1000
        )

        # Save the final model with UDR indication
        if args.use_udr:
            model_name = os.path.join(model_dir, f"udr_model_{int(args.mass_variation*100)}")
        else:
            model_name = os.path.join(model_dir, "source_model")
        print(f"Saving final model as {model_name}...")
        model.save(model_name)
        print("Final model saved successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Clean up
        train_env.close()
        eval_env.close()

if __name__ == '__main__':
    main()