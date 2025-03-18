# run.py
import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent
import torch

# 过滤警告
import warnings
warnings.filterwarnings("ignore")

def train(config_name, use_target_net=False, use_replay_buffer=False):
    env = gym.make("CartPole-v1")
    
    # 使用消融研究确定的最佳参数
    agent = QLearningAgent(
        env,
        use_target_net=use_target_net,
        use_replay_buffer=use_replay_buffer,
        hidden_dim=256,          # 最佳隐藏层维度
        lr=0.01,                 # 最佳学习率
        gamma=0.99,
        batch_size=128,
        buffer_size=10_000,
        tau=0.01,                # 目标网络软更新系数
        epsilon_decay=0.999,      # 最佳ε衰减率
        epsilon_min=0.01,
        update_interval=4         # 最佳更新间隔（每4步更新一次）
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
            
            # 只在更新间隔满足时执行训练
            if step_count % agent.update_interval == 0:
                agent.train_step(state, action, reward, next_state, done)
                
            state = next_state
            total_reward += reward
            step_count += 1
            episode_steps += 1

            # 定期打印进度（每10,000步）
            if step_count % 10_000 == 0:
                current_avg = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                print(f"{config_name} | Step {step_count} | Ep {episode} | ε={agent.epsilon:.3f} | Avg Reward={current_avg:.1f}")

        # 每个episode结束时更新ε
        agent.update_epsilon()

        # 记录奖励并计算滑动平均
        episode_rewards.append(total_reward)
        window_size = 20
        avg = np.mean(episode_rewards[-window_size:]) if episode >= window_size else np.mean(episode_rewards)
        avg_rewards.append(avg)
        episode += 1

    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.plot(episode_rewards, alpha=0.2, label="Raw Reward")
    plt.plot(avg_rewards, linewidth=2, label="Smoothed (20-episode Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{config_name} - Training Performance (Best Params)")
    plt.legend()
    plt.savefig(f"{config_name}_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

    return avg_rewards

def compare_configurations():
    """ 比较四种配置的性能（使用最佳参数）"""
    configurations = {
        # "Naive": {"use_target_net": False, "use_replay_buffer": False},
        "Only TN": {"use_target_net": True, "use_replay_buffer": False},
        # "Only ER": {"use_target_net": False, "use_replay_buffer": True},
        # "TN & ER": {"use_target_net": True, "use_replay_buffer": True}
    }

    results = {}
    for config_name, params in configurations.items():
        print(f"\n=== Training {config_name} with BEST PARAMS ===")
        avg_rewards = train(config_name, **params)
        results[config_name] = avg_rewards

    # 绘制对比曲线
    plt.figure(figsize=(12, 6))
    for label, rewards in results.items():
        plt.plot(rewards, label=label, linewidth=2)
    
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward (20-episode Avg)")
    plt.title("Performance Comparison with Best Parameters\n"
              "[LR=0.01, Update=4, Hidden=256, ε-decay=0.999]")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    compare_configurations()