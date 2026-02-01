
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env import custom_hopper  # Register custom envs
import os

def evaluate_model(model_path, env_name, n_episodes=50):
    print(f"Evaluating {model_path} on {env_name} for {n_episodes} episodes...")
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
    
    plt.xlabel('Episode Number')
    plt.ylabel('Episodic Reward')
    plt.title('SimOpt vs UDR: Target Environment Performance (50 Episodes)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Annotate the max points
    simopt_max_idx = np.argmax(simopt_rewards)
    plt.scatter(simopt_max_idx+1, np.max(simopt_rewards), color='red', zorder=5, s=100)
    plt.annotate('SimOpt Peak', (simopt_max_idx+1, np.max(simopt_rewards)), 
                 xytext=(simopt_max_idx+5, np.max(simopt_rewards)+50),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig('simopt_vs_udr_50_episodes.png', dpi=300)
    print("Saved simopt_vs_udr_50_episodes.png")

def main():
    # SimOpt Model
    simopt_model = 'champion_model_final.zip'
    
    # UDR Model (Best was UDR 30%)
    udr_model = 'best_model_udr_30/best_model.zip'
    
    if not os.path.exists(udr_model):
        print(f"Error: {udr_model} not found.")
        return

    simopt_rewards = evaluate_model(simopt_model, 'CustomHopper-target-v0', n_episodes=50)
    udr_rewards = evaluate_model(udr_model, 'CustomHopper-target-v0', n_episodes=50)
    
    if simopt_rewards and udr_rewards:
        plot_comparison(simopt_rewards, udr_rewards)
        print(f"SimOpt - Mean: {np.mean(simopt_rewards):.2f}, Max: {np.max(simopt_rewards):.2f}")
        print(f"UDR    - Mean: {np.mean(udr_rewards):.2f}, Max: {np.max(udr_rewards):.2f}")

if __name__ == "__main__":
    main()
