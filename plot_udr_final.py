
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env import custom_hopper  # Register custom envs
import os

def evaluate_model(model_path, env_name, n_episodes=100):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return 0, 0
        
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
        return np.mean(rewards), np.std(rewards)
    except Exception as e:
        print(f"Error evaluating: {e}")
        return 0, 0

def main():
    models = {
        'UDR 10%': 'best_model_udr_10/best_model.zip',
        'UDR 30%': 'best_model_udr_30/best_model.zip',
        'UDR 50%': 'best_model_udr_50/best_model.zip'
    }
    
    envs = {
        'Source': 'CustomHopper-source-v0',
        'Target': 'CustomHopper-target-v0'
    }
    
    results = {name: {'Source': 0, 'Target': 0} for name in models}
    errors = {name: {'Source': 0, 'Target': 0} for name in models}
    
    # Run Evaluations
    for model_name, model_path in models.items():
        for env_label, env_name in envs.items():
            mean, std = evaluate_model(model_path, env_name, n_episodes=100)
            results[model_name][env_label] = mean
            errors[model_name][env_label] = std
            print(f"{model_name} on {env_label}: {mean:.2f} +/- {std:.2f}")

    # Plotting Grouped Bar Chart
    labels = list(models.keys())
    source_means = [results[l]['Source'] for l in labels]
    target_means = [results[l]['Target'] for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, source_means, width, label='Source (Simulation)', color='skyblue', edgecolor='black')
    rects2 = ax.bar(x + width/2, target_means, width, label='Target (Real World)', color='salmon', edgecolor='black')
    
    ax.set_ylabel('Average Episodic Reward')
    ax.set_title('UDR Performance Comparison (100 Episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add labels on top bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('udr_final_comparison.png', dpi=300)
    print("Saved udr_final_comparison.png")

if __name__ == "__main__":
    main()
