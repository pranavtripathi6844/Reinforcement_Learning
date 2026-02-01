#!/usr/bin/env python3
"""Test trained SAC models on Hopper environment (source or target)."""

import gym
from stable_baselines3 import SAC
import numpy as np
import argparse

# Import custom environment to register it
from env import custom_hopper

def test_model(model_path, n_episodes=50, env_name='CustomHopper-source-v0'):
    """
    Test a trained model.
    
    Args:
        model_path: Path to the model .zip file
        n_episodes: Number of evaluation episodes
        env_name: Environment name ('CustomHopper-source-v0' or 'CustomHopper-target-v0')
    """
    # Create environment
    print(f"Environment: {env_name}")
    env = gym.make(env_name)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path)
    
    # Run evaluation
    print(f"\nRunning {n_episodes} episodes...")
    rewards = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        
        rewards.append(ep_reward)
        
        # Print progress every 10 episodes
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}: {ep_reward:.2f}")
    
    # Print results
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min reward:  {np.min(rewards):.2f}")
    print(f"  Max reward:  {np.max(rewards):.2f}")
    print(f"{'='*60}\n")
    
    env.close()
    return mean_reward, std_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SAC model on CustomHopper')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model .zip file')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes to run')
    parser.add_argument('--env', type=str, default='CustomHopper-source-v0',
                        choices=['CustomHopper-source-v0', 'CustomHopper-target-v0'],
                        help='Environment: source (sim) or target (real)')
    
    args = parser.parse_args()
    
    test_model(args.model, args.episodes, args.env)
