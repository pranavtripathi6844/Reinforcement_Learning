
import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env import custom_hopper  # Register custom envs

def evaluate_and_plot(model_path, env_name, n_episodes=100, title="", filename="", raw_color="silver", smooth_color="blue"):
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
            if (i+1) % 10 == 0:
                print(f"Episode {i+1}/{n_episodes}: {total_reward:.2f}")
        env.close()
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Plot raw data
        plt.plot(range(1, n_episodes+1), rewards, alpha=0.6, color=raw_color, label='Episodic Reward')
        
        # Plot smoothed data (Rolling Mean)
        window = 5
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window, len(rewards)+1), smoothed, color=smooth_color, linewidth=2.5, linestyle='-', label=f'Moving Average (n={window})')
        
        plt.xlabel('Episode Number')
        plt.ylabel('Reward')
        plt.title(f"{title}\nMean: {np.mean(rewards):.2f} | Max: {np.max(rewards):.2f}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved {filename}")
        
    except Exception as e:
        print(f"Error evaluating: {e}")

def main():
    model_path = 'champion_model_final.zip'
    
    # Graph 1: SimOpt on Target Environment
    # Raw: Darker Yellow (DarkGoldenrod), Smooth: Green
    evaluate_and_plot(
        model_path, 
        'CustomHopper-target-v0', 
        n_episodes=100, 
        title='SimOpt Champion: Evaluation on Target Environment (Real World)',
        filename='simopt_target_eval.png',
        raw_color='darkgoldenrod', 
        smooth_color='green'
    )
    
    # Graph 2: SimOpt on Source Environment
    # Raw: Darker Yellow (DarkGoldenrod), Smooth: Green
    evaluate_and_plot(
        model_path, 
        'CustomHopper-source-v0', 
        n_episodes=100, 
        title='SimOpt Champion: Evaluation on Source Environment (Simulation)', 
        filename='simopt_source_eval.png',
        raw_color='darkgoldenrod', 
        smooth_color='green'
    )

if __name__ == "__main__":
    main()
