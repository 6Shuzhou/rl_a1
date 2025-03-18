# Deep Q-Learning for CartPole-v1


PyTorch implementation of DQN with ablation studies and component analysis for CartPole-v1 environment.

## Features

- **Algorithm Components**:
  - naive DQN
  - Target Network (TN)
  - Experience Replay (ER)
  - Combined TN+ER

- **Hyperparameter Optimization**:
  - Learning rate: {1e-4, 1e-3, 1e-2}
  - Update interval: {1, 4, 16} steps
  - Network width: {64, 128, 256} hidden units
  - ε-decay: {0.99, 0.999, 0.9999}

- **Training Features**:
  - 1M environment steps training
  - 200-episode moving average
  - Gradient clipping (max norm=1.0)
  - Periodic evaluation (50 episodes)


### File Structure


- agent.py               # QNetwork and QLearningAgent implementation
- TN_ER_run.py                 # Component comparison experiments
- ablation_study_naive.py      # Hyperparameter sensitivity analysis
- replay_buffer.py       # Experience replay implementation
- ablation_plots               # Generated performance plots
- ablation_results               # Training logs and metrics

