import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import os
import re

# ====================== 路径 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "dqn_best_model")

# ====================== 网络结构（必须和训练一致）======================
class QNet(nn.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ====================== 列出所有模型 ======================
def list_all_models():
    if not os.path.exists(model_dir):
        print("❌ 模型文件夹不存在")
        return []
    
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    files.sort()
    return files

# ====================== 解析模型分数 ======================
def get_model_score(filename):
    match = re.search(r"rew(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0

# ====================== 可视化测试 chosen model ======================
def visualize_model(model_name):
    model_path = os.path.join(model_dir, model_name)
    
    env = gym.make("CartPole-v1", render_mode="human")
    net = QNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    print(f"✅ 正在观看模型：{model_name}")
    print(f"🎯 模型分数：{get_model_score(model_name)}")

    for episode in range(5):  # 播放5轮
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

        print(f"本轮分数：{int(total_reward)}")

    env.close()

# ====================== 主程序 ======================
if __name__ == "__main__":
    models = list_all_models()

    if not models:
        print("❌ 没有模型可观看")
        exit()

    print("="*50)
    print("📺 可用模型列表（按分数从高到低）")
    print("="*50)

    # 按分数排序
    models_with_score = [(f, get_model_score(f)) for f in models]
    models_with_score.sort(key=lambda x: x[1], reverse=True)

    for i, (fname, score) in enumerate(models_with_score):
        print(f"{i:2d} | 分数 {score:3d} | {fname}")

    print("\n输入序号观看对应模型：")
    idx = int(input(">>> "))

    if 0 <= idx < len(models_with_score):
        chosen = models_with_score[idx][0]
        visualize_model(chosen)
    else:
        print("❌ 输入错误")