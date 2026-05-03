# ====================== 在Q_learning_run_demo的基础上添加日志功能 ======================

import numpy as np
import os
import logging

# ====================== 自动获取脚本所在路径 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir,"train.log")

# ====================== 日志配置：全部写入文件，终端不显示 ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8")  # 日志保存到 train.log
    ]
)
log = logging.getLogger()

# ====================== 环境设定 ======================
n_states = 5
actions = [0, 1]
goal_state = 4

alpha = 0.1
gamma = 0.99
epsilon = 0.1

# ====================== 路径 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "best_models")
npy_path = os.path.join(folder_path, "best_q_table.npy")
csv_path = os.path.join(folder_path, "best_q_table.csv")

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# ====================== 加载模型 ======================
def load_best_model():
    if os.path.exists(npy_path):
        Q = np.load(npy_path)
        log.info("✅ 成功加载最优模型！继续训练...")
        return Q
    else:
        log.info("⚠️ 未找到模型，从头开始训练...")
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

# ====================== 保存模型 ======================
def save_best_model():
    np.save(npy_path, Q)
    np.savetxt(csv_path, Q, fmt="%.4f", delimiter=",")
    log.info("💾 新纪录！已保存最优模型（npy+csv）")

# ====================== 训练 ======================
log.info("="*60)
log.info("🚀 开始训练！")
log.info("="*60)

for episode in range(10):
    s = 0
    total_reward = 0

    log.info(f"\n\n🔴 ========== 第 {episode} 轮开始 ==========")
    log.info(f"📍 初始状态 = {s}")

    for step_idx in range(20):
        a = choose_action(s)
        s_next, r = step(s, a)

        log.info(f"\n--- 第 {episode} 轮 | 步数 {step_idx} ---")
        log.info(f"当前状态 s: {s}")
        log.info(f"选择动作 a: {a}")
        log.info(f"下一状态 s': {s_next}")
        log.info(f"获得奖励 r: {r}")

        log.info("\n📊 更新前 Q 表:")
        log.info(np.round(Q, 2))

        update_Q(s, a, r, s_next)
        total_reward += r

        log.info("\n✅ 更新后 Q 表:")
        log.info(np.round(Q, 2))

        s = s_next
        log.info(f"\n🎯 目标状态: {goal_state}")

        if s == goal_state:
            log.info(f"\n🎉 第 {episode} 轮到达终点！")
            break

    log.info(f"\n🏆 第 {episode} 轮总奖励: {total_reward}")

    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        save_best_model()

    log.info(f"🔥 当前历史最高奖励: {best_total_reward}")

# ====================== 最终结果 ======================
log.info("\n\n" + "="*60)
log.info("📋 训练完成！最终 Q 表：")
log.info(np.round(Q, 2))
log.info("="*60)

log.info("\n最终最优路径：")
s = 0
path = [s]
for _ in range(10):
    a = np.argmax(Q[s])
    s, _ = step(s, a)
    path.append(s)
    if s == goal_state:
        break
log.info(f"🧭 路径: {path}")

# 终端只输出一句话表示完成
print("✅ 训练完成！所有日志已保存到 train.log 文件")