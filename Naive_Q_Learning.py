import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QLearningAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)  # Random action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_net(state_tensor)
                return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # 计算当前Q值（需要梯度）
        current_q = self.q_net(state_tensor)[action]  # 保持梯度
        
        # 计算目标Q值（不需要梯度）
        with torch.no_grad():  # 关键修改：只在此处禁用梯度
            next_q = torch.max(self.q_net(next_state_tensor))
            target_q = reward + (1 - float(done)) * self.gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ε衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# 修改后的训练循环
env = gym.make('CartPole-v1')
agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)

for episode in range(500):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Episode: {episode+1}, Reward: {total_reward}, ε: {agent.epsilon:.2f}")