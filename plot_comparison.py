import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_reward_comparison(log_files, labels, output_file, window=100):
    plt.figure(figsize=(10, 6))
    
    for log_file, label in zip(log_files, labels):
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found. Skipping.")
            continue
            
        df = pd.read_csv(log_file)
        if len(df) < window:
            print(f"Warning: {log_file} has too few entries for window {window}. Plotting raw.")
            plt.plot(df['episode'], df['reward'], label=label, alpha=0.3)
        else:
            smoothed_reward = moving_average(df['reward'], window)
            # Adjust episodes to match smoothed length
            episodes = df['episode'].iloc[window-1:]
            plt.plot(episodes, smoothed_reward, label=label)
            
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Comparison of RL Algorithms: Learning Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    print(f"Reward comparison saved to {output_file}")

def plot_time_comparison(log_files, labels, output_file):
    times = []
    valid_labels = []
    
    for log_file, label in zip(log_files, labels):
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found. Skipping.")
            continue
            
        df = pd.read_csv(log_file)
        total_time = df['time'].iloc[-1]
        times.append(total_time)
        valid_labels.append(label)
        
    if not times:
        print("No valid data for time comparison.")
        return
        
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_labels, times, color=['skyblue', 'salmon', 'lightgreen'][:len(times)])
    
    plt.xlabel('Algorithm')
    plt.ylabel('Total Training Time (seconds)')
    plt.title('Comparison of Training Time (100k Episodes)')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}s', ha='center', va='bottom')
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    print(f"Time comparison saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', default=['reinforce_no_baseline.csv', 'reinforce_baseline.csv', 'actor_critic_log.csv'],
                        help='List of log files')
    parser.add_argument('--labels', nargs='+', default=['REINFORCE (No Baseline)', 'REINFORCE (Baseline)', 'Actor-Critic'],
                        help='Labels for the plot')
    parser.add_argument('--reward-out', default='reward_comparison.png', help='Output file for reward plot')
    parser.add_argument('--time-out', default='time_comparison.png', help='Output file for time plot')
    parser.add_argument('--window', type=int, default=500, help='Moving average window')
    
    args = parser.parse_args()
    
    plot_reward_comparison(args.logs, args.labels, args.reward_out, args.window)
    plot_time_comparison(args.logs, args.labels, args.time_out)
