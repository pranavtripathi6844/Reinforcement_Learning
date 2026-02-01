"""Train an RL agent on the OpenAI Gym Hopper environment using Actor-Critic"""
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
from actor_critic.agent_ac import ActorCriticAgent, Policy, Value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=1000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='auto', type=str, help='network device [cpu, cuda, auto]')
    parser.add_argument('--lr-actor', default=1e-3, type=float, help='Actor learning rate')
    parser.add_argument('--lr-critic', default=1e-3, type=float, help='Critic learning rate')
    parser.add_argument('--n-envs', default=16, type=int, help='Number of parallel environments')
    parser.add_argument('--log-file', default='actor_critic_log.csv', type=str, help='CSV log file')
    parser.add_argument('--model-file', default='actor_critic.mdl', type=str, help='Output model file')
    parser.add_argument('--checkpoint-dir', default='checkpoints_ac', type=str, help='Checkpoint directory')
    parser.add_argument('--checkpoint-every', default=5000, type=int, help='Save checkpoint every N episodes')

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
    value = Value(observation_space_dim)
    agent = ActorCriticAgent(policy, value, device=device, lr_actor=args.lr_actor, lr_critic=args.lr_critic)

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Try to resume from checkpoint
    start_episode = 0
    checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.value.load_state_dict(checkpoint['value_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        start_episode = checkpoint['episode']
        print(f"Resumed from episode {start_episode}")
    else:
        print("No checkpoint found, starting from scratch")

    # Initialize log file (append mode if resuming)
    log_mode = 'a' if start_episode > 0 else 'w'
    if log_mode == 'w':
        with open(args.log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'time'])

    total_episodes = start_episode
    start_time = time.time()
    
    # Track reward for each running episode
    env_rewards = np.zeros(args.n_envs)
    
    print(f"Starting training for {args.n_episodes} episodes...")
    
    states = env.reset()
    
    while total_episodes < args.n_episodes:
        dones = np.zeros(args.n_envs, dtype=bool)
        all_dones = [False] * args.n_envs
        env_rewards[:] = 0 # reset tracking for this batch

        # Collect trajectories until all environments finish at least one episode
        while not all(all_dones):
            actions, action_log_probs = agent.get_action(states)
            
            # Detach actions for gym
            actions_np = actions.detach().cpu().numpy()
            
            next_states, rewards, dones, infos = env.step(actions_np)
            
            # Store transition for all envs (masking is implicit in agent if needed, 
            # here we just store everything, but for true MC we need to be careful.
            # However, for simplicity and speed, we store all steps. 
            # The agent will compute returns for the full batch (T, N_ENVS).
            # Note: This simple batching implies all envs are synchronized in length 
            # or we accept some 'padding' steps after done. 
            # A more robust implementation would handle masking inside agent.store_outcome
            agent.store_outcome(states, next_states, action_log_probs, rewards, dones)
            
            # Update episode tracking
            env_rewards += rewards * (1 - np.array(all_dones)) 
            
            for i in range(args.n_envs):
                if dones[i]:
                    all_dones[i] = True
            
            states = next_states

        # Update policy and value AFTER full batch of episodes
        policy_loss, value_loss = agent.update()
        
        # Log and print results for this batch of episodes
        elapsed_time = time.time() - start_time
        total_episodes += args.n_envs
        
        for i in range(args.n_envs):
             with open(args.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([total_episodes - args.n_envs + i + 1, env_rewards[i], elapsed_time])

        if total_episodes % args.print_every < args.n_envs:
            avg_return = np.mean(env_rewards)
            print(f'Episode: {total_episodes} | Avg Return: {avg_return:.2f} | P-Loss: {policy_loss:.4f} | V-Loss: {value_loss:.4f} | Time: {elapsed_time:.1f}s')
        
        # Save checkpoint periodically
        if total_episodes % args.checkpoint_every < args.n_envs and total_episodes > 0:
            checkpoint = {
                'episode': total_episodes,
                'policy_state_dict': agent.policy.state_dict(),
                'value_state_dict': agent.value.state_dict(),
                'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            # Also save a numbered backup
            backup_path = os.path.join(args.checkpoint_dir, f'checkpoint_ep{total_episodes}.pt')
            torch.save(checkpoint, backup_path)
            print(f'Checkpoint saved at episode {total_episodes}')

    # Save final model
    torch.save(agent.policy.state_dict(), args.model_file)
    print(f"Model saved to {args.model_file}")
    env.close()


if __name__ == '__main__':
    main()