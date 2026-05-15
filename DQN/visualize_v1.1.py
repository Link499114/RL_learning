import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import os
import argparse

# ====================== 路径 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "dqn_best_model")

# ====================== 网络结构 ======================
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

# ====================== 可视化函数 ======================
def visualize_model(model_name):
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型不存在：{model_path}")
        return

    env = gym.make("CartPole-v1", render_mode="human")
    net = QNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    print(f"\n✅ 正在运行模型：{model_name}")

    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                s_tensor = torch.FloatTensor(state)
                act = net(s_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"第 {episode+1} 轮得分：{int(total_reward)}")

    env.close()

# ====================== 命令行参数解析 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN 可视化工具")
    parser.add_argument("--model", type=str, required=True, help="输入模型文件名，例如：dqn_ep120_rew500_xxx.pth")
    
    args = parser.parse_args()
    visualize_model(args.model)