
import matplotlib.pyplot as plt
import numpy as np

def plot_final_performance():
    # Data from Table 2 of Comparison Report
    algorithms = ['REINFORCE\n(No Baseline)', 'REINFORCE\n(Baseline)', 'Actor-Critic']
    rewards = [213.85, 368.26, 1009.72]
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # Salmon, Skyblue, Lightgreen

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, rewards, color=colors, edgecolor='black', alpha=0.8)

    # Add labels
    plt.ylabel('Average Episodic Reward (100 Test Episodes)')
    plt.title('Final Performance Comparison: Actor-Critic vs REINFORCE')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Improve layout
    plt.ylim(0, 1200)  # Give some headroom for labels
    plt.tight_layout()
    
    filename = 'final_performance_bar_chart.png'
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")

if __name__ == "__main__":
    plot_final_performance()
