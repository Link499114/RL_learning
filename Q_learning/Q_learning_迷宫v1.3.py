import numpy as np
import os

# ====================== 环境设定 ======================
# 0:起点 | 1:陷阱 | 2:安全路 | 3:安全路 | 4:终点
n_states = 5
actions = [0, 1]  # 0=去正确路, 1=去陷阱
goal_state = 4

# 超参数
alpha = 0.1
gamma = 0.99
epsilon = 0.1

# 获取【当前脚本所在的文件夹路径】
script_dir = os.path.dirname(os.path.abspath(__file__))
print("📂 脚本当前路径：", script_dir)

# 要创建的文件夹路径（在脚本同目录下）
folder_path = os.path.join(script_dir, "best_models")
npy_path = os.path.join(folder_path, "best_q_table.npy")
csv_path = os.path.join(folder_path, "best_q_table.csv")

# 如果存放模型的文件夹不存在-->创建
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# ====================== 加载模型 ======================
def load_best_model():
    if os.path.exists(npy_path):
        Q = np.load(npy_path)
        print("✅ 成功加载最优模型！继续训练...")
        return Q
    else:
        print("⚠️ 未找到模型，从头开始训练...")
        return np.zeros((n_states, len(actions)))

Q = load_best_model()
best_total_reward = -1

# ====================== 选择动作 ======================
def choose_action(state):
    if np.random.uniform() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state, :])

# ====================== 环境 ======================
def step(state, action):
    if state == 0:
        next_state = 2 if action == 0 else 1
    elif state == 1:
        next_state = 1
    elif state == 2:
        next_state = 3
    elif state == 3:
        next_state = 4
    else:
        next_state = 4

    if next_state == 1:
        reward = 1
    elif next_state == 4:
        reward = 1000
    else:
        reward = 0
    return next_state, reward

# ====================== Q更新 ======================
def update_Q(s, a, r, s_next):
    Q[s,a] = (1 - alpha) * Q[s,a] + alpha * (r + gamma * np.max(Q[s_next]))

# ====================== 保存最好模型 ======================
def save_best_model():
    np.save(npy_path, Q)
    np.savetxt(csv_path, Q, fmt="%.4f", delimiter=",")
    print("💾 新纪录！已保存最优模型（npy+csv）")

# ====================== 【训练 + 完整日志打印】 ======================
print("="*60)
print("🚀 开始训练！")
print("="*60)

for episode in range(10):
    s = 0
    total_reward = 0

    print(f"\n\n🔴 ========== 第 {episode} 轮开始 ==========")
    print(f"📍 初始状态 = {s}")

    for step_idx in range(20):
        a = choose_action(s)
        s_next, r = step(s, a)

        # =============== 【完整日志打印】每一步全部输出 ===============
        print(f"\n--- 第 {episode} 轮 | 步数 {step_idx} ---")
        print(f"当前状态 s: {s}")
        print(f"选择动作 a: {a}")
        print(f"下一状态 s': {s_next}")
        print(f"获得奖励 r: {r}")

        print("\n📊 更新前 Q 表:")
        print(np.round(Q, 2))

        # 更新 Q
        update_Q(s, a, r, s_next)
        total_reward += r

        print("\n✅ 更新后 Q 表:")
        print(np.round(Q, 2))

        s = s_next
        print(f"\n🎯 目标状态: {goal_state}")

        if s == goal_state:
            print(f"\n🎉 第 {episode} 轮到达终点！")
            break

    # =============== 每轮结束 ===============
    print(f"\n🏆 第 {episode} 轮总奖励: {total_reward}")

    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        save_best_model()

    print(f"🔥 当前历史最高奖励: {best_total_reward}")

# ====================== 最终结果 ======================
print("\n\n" + "="*60)
print("📋 训练完成！最终 Q 表：")
print(np.round(Q, 2))
print("="*60)

print("\n最终最优路径：")
s = 0
path = [s]
for _ in range(10):
    a = np.argmax(Q[s])
    s, _ = step(s, a)
    path.append(s)
    if s == goal_state:
        break
print("🧭 路径:", path)