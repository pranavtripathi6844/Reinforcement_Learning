"""Train an RL agent on the OpenAI Gym Hopper environment using REINFORCE with baseline"""
import argparse
import torch
import gym
import numpy as np
import os
import csv
import time
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.custom_hopper import *
from reinforce.agent_reinforce import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='auto', type=str, help='network device [cpu, cuda, auto]')
    parser.add_argument('--baseline', default=20.0, type=float, help='Baseline value for REINFORCE')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--n-envs', default=16, type=int, help='Number of parallel environments')
    parser.add_argument('--log-file', default='reinforce_log.csv', type=str, help='CSV log file')
    parser.add_argument('--model-file', default='model_with_baseline.mdl', type=str, help='Output model file')

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Function to create environment
    def make_env():
        return gym.make('CustomHopper-source-v0')

    # Create parallel environments
    print(f"Creating {args.n_envs} parallel environments...")
    env = make_vec_env(make_env, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=device, baseline=args.baseline, lr=args.lr)

    # Initialize log file
    with open(args.log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'time'])

    total_episodes = 0
    start_time = time.time()
    
    print(f"Starting training for {args.n_episodes} episodes...")
    
    while total_episodes < args.n_episodes:
        states = env.reset()
        dones = np.zeros(args.n_envs, dtype=bool)
        episode_rewards = np.zeros(args.n_envs)
        
        # Buffer for the whole batch of episodes
        # REINFORCE typically updates after full trajectories
        all_dones = [False] * args.n_envs
        
        while not all(all_dones):
            actions, action_log_probs = agent.get_action(states)
            
            # Detach actions for gym
            actions_np = actions.detach().cpu().numpy()
            
            next_states, rewards, dones, infos = env.step(actions_np)
            
            # Store transition
            # We only store if the environment hasn't finished yet in this batch
            # However, SubprocVecEnv resets automatically, so we need to mask 
            # rewards for environments that already finished.
            mask = torch.tensor([not d for d in all_dones]).to(device).float()
            agent.store_outcome(action_log_probs, rewards, dones)
            
            episode_rewards += rewards * (1 - np.array(all_dones))
            
            # Track which environments finished
            for i in range(args.n_envs):
                if dones[i]:
                    all_dones[i] = True
            
            states = next_states

        # Update policy after one full episode in each parallel environment
        policy_loss = agent.update_policy()
        
        total_episodes += args.n_envs
        
        # Average reward for this batch
        avg_reward = np.mean(episode_rewards)
        elapsed_time = time.time() - start_time
        
        # Log results
        with open(args.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            # Log multiple entries if multiple episodes finished
            for i in range(args.n_envs):
                writer.writerow([total_episodes - args.n_envs + i + 1, episode_rewards[i], elapsed_time])

        if total_episodes % args.print_every < args.n_envs:
            print(f'Episode: {total_episodes} | Avg Return: {avg_reward:.2f} | Loss: {policy_loss:.4f} | Time: {elapsed_time:.1f}s')

    # Save final model
    torch.save(agent.policy.state_dict(), args.model_file)
    print(f"Model saved to {args.model_file}")
    env.close()


if __name__ == '__main__':
    main()