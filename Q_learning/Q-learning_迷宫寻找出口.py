import numpy as np
import time

# ====================== 1. 环境设置 ======================
# 状态数量：0,1,2,3,4,5 共6个位置
n_states = 6
# 动作：0=左 1=右
actions = [0, 1]
# 目标状态
goal_state = 3

# ====================== 2. Q 表初始化 ======================
# Q表：[状态数, 动作数]，一开始全是0
Q = np.zeros((n_states, len(actions)))

# 学习参数
alpha = 0.1    # 学习率
gamma = 0.1    # 折扣因子（看重未来奖励）
epsilon = 0.1  # 探索概率（10%概率随机走，防止死脑筋）

# ====================== 3. 选择动作 ======================
def choose_action(state):
    # 10% 概率随机探索
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(actions)
    # 90% 概率选Q值最大的动作（利用）
    else:
        return np.argmax(Q[state, :])

# ====================== 4. 环境交互：走一步 ======================
def step(state, action):
    # 向右走
    if action == 1:
        next_state = state + 1
    # 向左走
    else:
        next_state = state - 1

    # 不能走出边界
    next_state = max(0, min(next_state, n_states - 1))

    # 奖励：到终点+100，其他-1
    if next_state == goal_state:
        reward = 100
    else:
        reward = -1

    return next_state, reward

# ====================== 5. Q-Learning 更新公式 ======================
def update_Q(state, action, reward, next_state):
    # 核心公式！
    old_q = Q[state, action]
    # 下一个状态最大Q值
    next_max_q = np.max(Q[next_state, :])
    # 更新
    new_q = (1 - alpha) * old_q + alpha * (reward + gamma * next_max_q)
    Q[state, action] = new_q

# ====================== 6. 开始训练 ======================
print("开始训练 Q-Learning...\n")

for episode in range(30):  # 训练30轮
    state = 0  # 每轮从起点0开始
    total_reward = 0

    while True:
        # 选动作
        action = choose_action(state)
        # 走一步
        next_state, reward = step(state, action)
        # 更新Q表
        update_Q(state, action, reward, next_state)

        total_reward += reward
        state = next_state

        # 到达终点结束本轮
        if state == goal_state:
            break

    print(f"轮次 {episode:2d} | 总奖励: {total_reward:3d}")
    time.sleep(0.1)

# ====================== 7. 训练完成，看学到的Q表 ======================
print("\n训练完成！最终 Q 表：")
print(Q)
print("\n每一行是一个状态，每一列是动作（左/右）的价值")

# ====================== 8. 演示最优路径 ======================
print("\n智能体走最优路径：")
state = 0
path = [state]
while state != goal_state:
    action = np.argmax(Q[state, :])
    state, _ = step(state, action)
    path.append(state)
print("行走路径:", path)
