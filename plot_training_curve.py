
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_log(log_file, output_file):
    print(f"Loading log file: {log_file}...")
    
    # Initialize EventAccumulator
    ea = EventAccumulator(log_file)
    ea.Reload()
    
    # Extract scalar data
    tags = ea.Tags()['scalars']
    print(f"Available tags: {tags}")
    
    if 'rollout/ep_rew_mean' not in tags:
        print("Error: 'rollout/ep_rew_mean' not found in logs!")
        # Fallback to similar tags if standard name missing
        candidates = [t for t in tags if 'rew' in t]
        if candidates:
            tag = candidates[0]
            print(f"Using alternative tag: {tag}")
        else:
            return
    else:
        tag = 'rollout/ep_rew_mean'

    # Get data
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    rewards = [e.value for e in events]
    
    # Truncate to 2000 episodes (1M steps) as requested
    max_steps = 1000000
    zipped = [(s, r) for s, r in zip(steps, rewards) if s <= max_steps]
    steps_trunc, rewards_trunc = zip(*zipped)
    
    # Convert steps to episodes (500 steps per episode)
    episodes = [s / 500 for s in steps_trunc]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards_trunc, label='Average Episodic Reward', color='blue', linewidth=2)
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.title('Training Progress (Best Parameters) - First 2000 Episodes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add a secondary x-axis for episodes? (Optional but nice)
    # 1M steps = 2000 episodes
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    # Log file for SAC_1
    log_dir = "logs/SAC_1"
    # Find the events file
    event_files = [f for f in os.listdir(log_dir) if "events.out.tfevents" in f]
    if event_files:
        log_path = os.path.join(log_dir, event_files[0])
        plot_tensorboard_log(log_path, "sac_training_progress.png")
    else:
        print("No event file found!")
