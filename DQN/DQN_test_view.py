import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import os

# ====================== 路径和训练脚本保持一致 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "dqn_best_model")
model_path = os.path.join(model_dir, "best_dqn.pth")

# ====================== 网络结构 和 训练时完全一样 ======================
n_states = 4
n_actions = 2

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

# ====================== 加载模型 + 可视化演示 ======================
def test_model():
    # 带画面渲染
    env = gym.make("CartPole-v1", render_mode="human")
    
    # 初始化网络并加载权重
    q_net = QNet()
    if os.path.exists(model_path):
        q_net.load_state_dict(torch.load(model_path))
        print("✅ 成功加载训练好的最优模型！")
    else:
        print("❌ 未找到模型文件，请先训练生成 best_dqn.pth")
        return

    q_net.eval()  # 评估模式，关闭dropout等

    # 循环演示
    for ep in range(10):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 不探索，只选最优动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.array(state))
                q_val = q_net(state_tensor)
                action = q_val.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"第 {ep+1} 轮 | 平衡步数: {int(total_reward)}")

    env.close()

if __name__ == "__main__":
    test_model()