import numpy as np
import torch
import torch.optim as optim
from model import QNetwork
from replay_buffer import ReplayBuffer
import torch.nn as nn

class QLearningAgent:
    def __init__(self, env, use_target_net=False, use_replay_buffer=False,
                 hidden_dim=128, lr=0.001, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.999995, epsilon_min=0.01, batch_size=64,
                 buffer_size=10_000, target_update_freq=100):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.use_target_net = use_target_net
        self.use_replay_buffer = use_replay_buffer
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 初始化网络
        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden_dim)
        self.target_q_net = QNetwork(self.state_dim, self.action_dim, hidden_dim) if use_target_net else None
        if self.target_q_net:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_q_net.eval()

        # 优化器和损失函数
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size) if use_replay_buffer else None

        # 探索参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # 训练步数计数器
        self.train_step_count = 0  # 新增计数器

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_net(state_tensor)
                return torch.argmax(q_values).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        if self.target_q_net:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        self.train_step_count += 1  # 每次训练增加计数器

        # 将经验添加到缓冲区
        if self.use_replay_buffer:
            self.replay_buffer.add(state, action, reward, next_state, done)
            if len(self.replay_buffer) < self.batch_size:
                return  # 缓冲区不足时跳过训练

        # 从缓冲区采样或使用单步经验
        if self.use_replay_buffer:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        else:
            states = np.array([state])
            actions = np.array([action])
            rewards = np.array([reward], dtype=np.float32)
            next_states = np.array([next_state])
            dones = np.array([done], dtype=np.bool_)

        # 转换为Tensor
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        # 计算当前Q值
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            if self.target_q_net:
                next_q = self.target_q_net(next_states).max(1)[0]
            else:
                next_q = self.q_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)

        # 计算损失并更新网络
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 条件更新目标网络（关键修改点）
        if self.use_target_net and (self.train_step_count % self.target_update_freq == 0):
            self.update_target_network()