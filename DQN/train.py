import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import time
import argparse
from datetime import datetime

# ====================== 命令行参数：指定模型继续训练 ======================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="指定要加载的模型文件名")
args = parser.parse_args()

# ====================== 路径 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "dqn_train.log")
model_dir = os.path.join(script_dir, "dqn_best_model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ====================== 日志 ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler(log_file, mode='a', encoding='utf-8')]
)
log = logging.getLogger()

# ====================== 环境 ======================
env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# ====================== 网络 ======================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ====================== DQN ======================
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
        state = torch.FloatTensor(state)
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
        batch = [self.memory[i] for i in np.random.choice(len(self.memory), self.batch_size)]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q = self.q_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ====================== 加载指定模型 ======================
    def load_specified_model(self, model_name):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"❌ 模型不存在：{model_path}")
            exit()
        self.q_net.load_state_dict(torch.load(model_path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        log.info(f"✅ 成功加载指定模型：{model_name}")

    # ====================== 保存新模型（不覆盖） ======================
    def save_new_model(self, episode, reward):
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dqn_ep{episode}_rew{int(reward)}_{time_str}.pth"
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.q_net.state_dict(), save_path)
        log.info(f"💾 新模型已保存：{model_name}")

# ====================== 训练 ======================
if __name__ == "__main__":
    start_time = time.time()
    agent = DQNAgent()

    # ========== 关键：如果传入 --model 则加载指定模型 ==========
    if args.model is not None:
        agent.load_specified_model(args.model)

    log.info("\n" + "="*70)
    log.info("🚀 训练开始")
    log.info("="*70)

    best_reward = -1

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

        if episode % 10 == 0:
            agent.update_target()

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_new_model(episode, total_reward)

        log.info(f"🔴 轮次 {episode:3d} | 得分: {int(total_reward):4d} | 最高: {int(best_reward)}")

    log.info("="*70)
    log.info(f"✅ 训练完成！总耗时: {time.time()-start_time:.2f}s")
    log.info("="*70)

    env.close()
    print("✅ 训练完成！")