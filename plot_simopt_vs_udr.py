
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env import custom_hopper  # Register custom envs
import pandas as pd

def evaluate_model(model_path, env_name, n_episodes=100):
    print(f"Evaluating {model_path} on {env_name}...")
    try:
        env = gym.make(env_name)
        model = SAC.load(model_path)
        rewards = []
        for i in range(n_episodes):
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
        print(f"Error evaluating {model_path}: {e}")
        return []

def plot_comparison(simopt_rewards, udr_rewards):
    plt.figure(figsize=(12, 7))
    
    episodes = range(1, len(simopt_rewards) + 1)
    
    # Plot SimOpt
    plt.plot(episodes, simopt_rewards, color='tab:red', alpha=0.9, linewidth=1.5, label='SimOpt Champion')
    plt.axhline(y=np.max(simopt_rewards), color='tab:red', linestyle='--', alpha=0.5, label=f'SimOpt Max: {np.max(simopt_rewards):.2f}')
    
    # Plot UDR
    plt.plot(episodes, udr_rewards, color='tab:blue', alpha=0.8, linewidth=1.5, label='UDR (Best ±30%)')
    plt.axhline(y=np.max(udr_rewards), color='tab:blue', linestyle='--', alpha=0.5, label=f'UDR Max: {np.max(udr_rewards):.2f}')
    
    # Fill between to highlight visual difference
    # plt.fill_between(episodes, simopt_rewards, udr_rewards, alpha=0.1, color='gray')

    plt.xlabel('Episode Number')
    plt.ylabel('Episodic Reward')
    plt.title('SimOpt vs UDR: Target Environment Performance (100 Episodes)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Annotate the max points
    simopt_max_idx = np.argmax(simopt_rewards)
    plt.scatter(simopt_max_idx+1, np.max(simopt_rewards), color='red', zorder=5, s=100)
    plt.annotate('SimOpt Peak', (simopt_max_idx+1, np.max(simopt_rewards)), 
                 xytext=(simopt_max_idx+10, np.max(simopt_rewards)+50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('simopt_vs_udr_comparison.png', dpi=300)
    print("Saved simopt_vs_udr_comparison.png")

def main():
    # SimOpt Model
    simopt_model = 'champion_model_final.zip'
    
    # UDR Model (Best was UDR 30%) - We need to point to the correct file.
    # Assuming standard location: models/udr_30/best_model.zip or similar.
    # I'll check directory structure if this fails, but for now assuming this path based on previous contexts
    # Wait, earlier logs mentioned "models_udr/udr_30_optuna/best_model.zip" or similar?
    # Let's try to locate it or use the standard "best_model.zip" if we just ran it.
    # Given the complexity, I will assume it's stored at:
    udr_model = 'models_udr/udr_30_optuna/best_model.zip' 
    
    # Check if UDR model exists, if not, try to find it
    import os
    if not os.path.exists(udr_model):
        print(f"Warning: {udr_model} not found. Searching...")
        # Fallback to finding it or using a placeholder if strictly needed (but better to fail and ask)
        # For this turn I will try to look for it in the same script
        found = False
        for root, dirs, files in os.walk("."):
            for file in files:
                if "udr" in root and "30" in root and file == "best_model.zip":
                    udr_model = os.path.join(root, file)
                    print(f"Found UDR model at: {udr_model}")
                    found = True
                    break
            if found: break
    
    simopt_rewards = evaluate_model(simopt_model, 'CustomHopper-target-v0', n_episodes=100)
    udr_rewards = evaluate_model(udr_model, 'CustomHopper-target-v0', n_episodes=100)
    
    if simopt_rewards and udr_rewards:
        plot_comparison(simopt_rewards, udr_rewards)
        # Print stats for user verification
        print(f"SimOpt - Mean: {np.mean(simopt_rewards):.2f}, Max: {np.max(simopt_rewards):.2f}")
        print(f"UDR    - Mean: {np.mean(udr_rewards):.2f}, Max: {np.max(udr_rewards):.2f}")

if __name__ == "__main__":
    main()
