import argparse
import gym
import numpy as np
import os
import sys
import torch
from stable_baselines3 import SAC

# Add root to path so we can import env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

def test_simopt(model_path, env_name='CustomHopper-target-v0', n_episodes=50):
    """
    Test a trained SimOpt model.
    """
    print(f"Testing model: {model_path}")
    print(f"Environment: {env_name}")
    
    if not os.path.exists(model_path):
        if not os.path.exists(model_path + ".zip"):
            print(f"Error: Model not found at {model_path}")
            return
            
    # Create env
    env = gym.make(env_name)
    
    # Load model
    model = SAC.load(model_path)
    
    rewards = []
    print(f"Running {n_episodes} evaluation episodes...")
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}/{n_episodes}: Reward = {total_reward:.2f}")
            
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print("\nResults:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}")
    
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--env', type=str, default='CustomHopper-target-v0', help='Environment to test on')
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    
    args = parser.parse_args()
    
    test_simopt(args.model, args.env, args.episodes)
