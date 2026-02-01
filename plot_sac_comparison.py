
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import os
from env import custom_hopper  # Register custom envs

def evaluate(model_path, env_name, n_episodes=50):
    print(f"Evaluating {model_path} on {env_name}...")
    try:
        env = gym.make(env_name)
        model = SAC.load(model_path)
        rewards = []
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, _ = env.step(action)
                total_reward += r
            rewards.append(total_reward)
        env.close()
        return rewards
    except Exception as e:
        print(f"Error evaluating: {e}")
        return [0]*n_episodes

def plot_comparison(results, title, filename):
    plt.figure(figsize=(10, 6))
    
    # Calculate rolling mean for smoother lines
    window = 5
    
    for label, rewards in results.items():
        # Plot raw data faintly - REMOVED per user request
        # plt.plot(rewards, alpha=0.2)
        
        # Plot smoothed data
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), smoothed, label=f"{label} (Mean: {np.mean(rewards):.0f})", linewidth=2)
        
    plt.xlabel('Heading: Episode Number')
    plt.ylabel('Episodic Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    # Configuration 1: Standard Parameters
    print("\n--- Testing Standard Parameters ---")
    std_results = {}
    std_results['Source to Source'] = evaluate('best_model_default/source_model.zip', 'CustomHopper-source-v0')
    std_results['Source to Target'] = evaluate('best_model_default/source_model.zip', 'CustomHopper-target-v0')
    std_results['Target to Target'] = evaluate('best_model_target/target_model.zip', 'CustomHopper-target-v0')
    
    plot_comparison(std_results, 'Standard Parameters: Source vs Target Transfer', 'sac_standard_comparison.png')
    
    # Configuration 2: Optuna Parameters
    print("\n--- Testing Optuna Parameters ---")
    opt_results = {}
    opt_results['Source to Source'] = evaluate('best_model/source_model.zip', 'CustomHopper-source-v0')
    opt_results['Source to Target'] = evaluate('best_model/source_model.zip', 'CustomHopper-target-v0')
    opt_results['Target to Target'] = evaluate('best_model_target_optuna/target_model_optuna.zip', 'CustomHopper-target-v0')
    
    plot_comparison(opt_results, 'Optuna Parameters: Source vs Target Transfer', 'sac_optuna_comparison.png')

if __name__ == "__main__":
    main()
