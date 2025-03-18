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
    )
    
    episode_rewards = []   # 存储每个episode的原始奖励
    avg_rewards = []       # 存储滑动平均奖励（用于平滑曲线）
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

            

            # 定期打印进度（每5,000步）
            if step_count % 5_000 == 0:
                current_avg = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                print(f"{config_name} | Step {step_count} | Ep {episode} | ε={agent.epsilon:.3f} | Avg Reward={current_avg:.1f}")
        
        agent.update_epsilon()        
        # 记录奖励并计算滑动平均
        episode_rewards.append(total_reward)
        window_size = 20
        if episode < window_size:
            avg = np.mean(episode_rewards[:episode+1])  # 初始阶段用全部数据
        else:
            avg = np.mean(episode_rewards[-window_size:])
        avg_rewards.append(avg)
        episode += 1

    # 绘制当前配置的训练曲线（原始+平滑）
    plt.figure(figsize=(12, 4))
    plt.plot(episode_rewards, alpha=0.2, label="Raw Reward")
    plt.plot(avg_rewards, linewidth=2, label="Smoothed (20-episode Avg)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"{config_name} - Training Performance")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{config_name}_training_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    return avg_rewards  # 返回平滑后的奖励用于对比

def compare_configurations():
    """ 比较四种配置的性能 """
    configurations = {
        "Naive": {"use_target_net": False, "use_replay_buffer": False},
        # "Only TN": {"use_target_net": True, "use_replay_buffer": False},
        # "Only ER": {"use_target_net": False, "use_replay_buffer": True},
        # "TN & ER": {"use_target_net": True, "use_replay_buffer": True}
    }

    results = {}
    for config_name, params in configurations.items():
        print(f"\n=== Training {config_name} ===")
        avg_rewards = train(config_name, **params)
        results[config_name] = avg_rewards

    # 绘制所有配置的对比曲线（平滑版）
    plt.figure(figsize=(12, 6))
    for label, rewards in results.items():
        plt.plot(rewards, label=label, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward (20-episode Avg)")
    plt.title("Performance Comparison of Different Configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图像释放内存

if __name__ == "__main__":
    compare_configurations()