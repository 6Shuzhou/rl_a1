# run.py
import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent
import torch

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

def train(config_name, use_target_net=False, use_replay_buffer=False):
    env = gym.make("CartPole-v1")
    
    # Use best parameters determined by ablation study
    agent = QLearningAgent(
        env,
        use_target_net=use_target_net,
        use_replay_buffer=use_replay_buffer,
        hidden_dim=256,          # Best hidden layer dimension
        lr=0.01,                 # Best learning rate
        gamma=0.99,
        batch_size=128,
        buffer_size=10_000,
        tau=0.1,                # Target network soft update coefficient
        epsilon_decay=0.999,     # Best epsilon decay rate
        epsilon_min=0.01,
        update_interval=1        # Best update interval
    )
    
    episode_rewards = []
    avg_rewards = []
    step_count = 0
    episode = 0
    total_steps = 1_000_000
    max_steps_per_episode = 500

    while step_count < total_steps:
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        episode_steps = 0

        while not (terminated or truncated) and episode_steps < max_steps_per_episode:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Only train when update interval condition is met
            if step_count % agent.update_interval == 0:
                agent.train_step(state, action, reward, next_state, done)
                
            state = next_state
            total_reward += reward
            step_count += 1
            episode_steps += 1

            # Periodically print progress (every 10,000 steps)
            if step_count % 10_000 == 0:
                current_avg = np.mean(episode_rewards[-200:]) if episode_rewards else 0
                print(f"{config_name} | Step {step_count} | Ep {episode} | ε={agent.epsilon:.3f} | Avg Reward={current_avg:.1f}")

        # Update epsilon at the end of each episode
        agent.update_epsilon()

        # Record rewards and calculate moving average
        episode_rewards.append(total_reward)
        window_size = 200
        avg = np.mean(episode_rewards[-window_size:]) if episode >= window_size else np.mean(episode_rewards)
        avg_rewards.append(avg)
        episode += 1

    # Save training curve
    plt.figure(figsize=(12, 4))
    plt.plot(episode_rewards, alpha=0.2, label="Raw Reward")
    plt.plot(avg_rewards, linewidth=2, label="Smoothed (200-episode Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{config_name} - Training Performance (Best Params)")
    plt.legend()
    plt.savefig(f"{config_name}_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

    return avg_rewards

def compare_configurations():
    """Compare performance of four configurations (using best parameters)"""
    configurations = {
        # "Naive": {"use_target_net": False, "use_replay_buffer": False},
        "Only TN": {"use_target_net": True, "use_replay_buffer": False},
        "Only ER": {"use_target_net": False, "use_replay_buffer": True},
        "TN & ER": {"use_target_net": True, "use_replay_buffer": True}
    }

    results = {}
    for config_name, params in configurations.items():
        print(f"\n=== Training {config_name} with BEST PARAMS ===")
        avg_rewards = train(config_name, **params)
        results[config_name] = avg_rewards

    # Plot comparison curves
    plt.figure(figsize=(12, 6))
    for label, rewards in results.items():
        plt.plot(rewards, label=label, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward (200-episode Avg)")
    plt.title("Performance Comparison with Best Parameters\n"
              "[LR=0.01, Update=4, Hidden=256, ε-decay=0.999]")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    compare_configurations()