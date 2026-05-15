import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import time
from datetime import datetime

# ====================== 路径设置（全部在脚本当前目录） ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, "dqn_train.log")
model_dir = os.path.join(script_dir, "dqn_best_model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ====================== 日志配置（带时间 + 追加 + 空行清爽） ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ]
)
log = logging.getLogger()

# ====================== 环境 ======================
env = gym.make("CartPole-v1")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# ====================== 神经网络 ======================
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

# ====================== DQN 智能体 ======================
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
        states, actions, rewards, next_states, dones = zip(*batch)

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

    # ====================== 保存模型：按时间+轮数命名，不覆盖 ======================
    def save_best_model(self, episode, reward):
        # 生成时间戳：年月日_时分秒
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 模型文件名：带轮数、奖励、时间
        model_name = f"dqn_ep{episode}_rew{int(reward)}_{time_str}.pth"
        save_path = os.path.join(model_dir, model_name)
        torch.save(self.q_net.state_dict(), save_path)
        log.info(f"✅ 新最优模型已保存: {model_name}")

    # 加载最近一个模型继续训练
    def load_latest_model(self):
        if not os.path.exists(model_dir):
            log.info("ℹ️ 无模型文件夹，从头开始训练")
            return False
        # 找出所有pth模型文件
        file_list = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not file_list:
            log.info("ℹ️ 无历史模型，从头开始训练")
            return False
        # 取最新修改的模型
        file_list.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
        latest_file = file_list[-1]
        latest_path = os.path.join(model_dir, latest_file)
        self.q_net.load_state_dict(torch.load(latest_path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        log.info(f"✅ 加载最新模型继续训练: {latest_file}")
        return True

# ====================== 训练主程序 ======================
if __name__ == "__main__":
    start_time = time.time()
    agent = DQNAgent()
    # 自动加载最新模型，不会覆盖旧模型
    agent.load_latest_model()

    log.info("")
    log.info("=" * 70)
    log.info("🚀 DQN CartPole 训练开始")
    log.info("=" * 70)

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

        # 只有创造新纪录才保存，且每次都是新文件，不覆盖
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_best_model(episode, total_reward)

        log.info("")
        log.info(f"🔴 轮次 {episode:3d} | 坚持步数: {int(total_reward):4d} | 历史最高: {int(best_reward)}")

    end_time = time.time()
    used_time = end_time - start_time

    log.info("")
    log.info("=" * 70)
    log.info(f"✅ 训练完成！总耗时: {used_time:.2f} 秒")
    log.info(f"🏆 训练最高奖励: {int(best_reward)}")
    log.info("=" * 70)
    log.info("")

    env.close()
    print(f"✅ 训练完成！所有模型保存在：\n{model_dir}")