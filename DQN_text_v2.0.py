import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 画图用

# ====================== 1. 环境（开启可视化） ======================
# render_mode="human" 会自动弹出窗口看到小车和杆子！
env = gym.make("CartPole-v1", render_mode="human")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# ====================== 2. 神经网络 ======================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ====================== 3. DQN 智能体 ======================
class DQNAgent:
    def __init__(self):
        self.q_net = QNet()
        self.target_net = QNet()
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 0.1
        self.batch_size = 32

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        state = torch.FloatTensor(np.array(state))
        with torch.no_grad():
            q = self.q_net(state)
        return q.argmax().item()

    def store(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch_idx = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_idx]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for s, a, r, s_n, d in batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_n)
            dones.append(d)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        q = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ====================== 4. 训练 + 画图 ======================
agent = DQNAgent()
reward_history = []  # 保存每一轮的奖励，用于画图

print("开始训练...\n")

for episode in range(300):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

    # 记录奖励
    reward_history.append(total_reward)

    if episode % 10 == 0:
        agent.update_target()
        print(f"轮次 {episode:3d} | 坚持步数: {total_reward}")

print("训练完成！")

# ====================== 5. 绘制训练曲线 ======================
plt.figure(figsize=(10, 4))
plt.plot(reward_history, label="Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN CartPole Training Curve")
plt.legend()
plt.grid(True)
plt.show()

env.close()
