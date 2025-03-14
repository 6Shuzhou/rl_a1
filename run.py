import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent

def train(config_name, use_target_net=False, use_replay_buffer=False):
    env = gym.make("CartPole-v1")
    agent = QLearningAgent(
        env,
        use_target_net=use_target_net,
        use_replay_buffer=use_replay_buffer,
        hidden_dim=128,
        lr=0.001,
        gamma=0.99,
        batch_size=64,
        buffer_size=10_000,
        target_update_freq=100
    )
    episode_rewards = []
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
            agent.train_step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            episode_steps += 1

            agent.update_epsilon()

            if step_count % 5_000 == 0:
                avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                print(f"{config_name} - Step {step_count}, Episodes {episode}, ε={agent.epsilon:.3f}, Avg Reward (last 20): {avg_reward:.1f}")

        episode_rewards.append(total_reward)
        episode += 1

    return episode_rewards

def compare_configurations():
    configurations = {
        "Naive": {"use_target_net": False, "use_replay_buffer": False},
        "Only TN": {"use_target_net": True, "use_replay_buffer": False},
        "Only ER": {"use_target_net": False, "use_replay_buffer": True},
        "TN & ER": {"use_target_net": True, "use_replay_buffer": True}
    }

    results = {}
    for config_name, params in configurations.items():
        print(f"\nTraining {config_name} configuration...")
        rewards = train(config_name, **params)
        results[config_name] = rewards

    # 绘制对比曲线
    plt.figure(figsize=(12, 6))
    for label, rewards in results.items():
        plt.plot(rewards, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Performance Comparison of Different Configurations")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    compare_configurations()