import numpy as np

# ====================== 环境设定 ======================
# 0:起点 | 1:陷阱 | 2:安全路 | 3:安全路 | 4:终点
n_states = 5
actions = [0, 1]  # 0=去正确路, 1=去陷阱
goal_state = 4

# Q表
Q = np.zeros((n_states, len(actions)))

# 超参数
alpha = 0.1
gamma = 0.1    # 👈 测试 0.99 / 0.1
epsilon = 0.1

# ====================== 选择动作 ======================
def choose_action(state):
    if np.random.uniform() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state, :])

# ====================== 环境：正确逻辑！ ======================
def step(state, action):
    # 起点：关键修正！
    if state == 0:
        if action == 0:
            next_state = 2   # ✅ 动作0 → 正确路线
        else:
            next_state = 1   # ❌ 动作1 → 陷阱

    elif state == 1:
        next_state = 1  # 陷阱卡死

    elif state == 2:
        next_state = 3  # 自动向前

    elif state == 3:
        next_state = 4  # 自动到终点

    else:
        next_state = 4

    # 奖励：陷阱只给小奖励，终点给超级奖励
    if next_state == 1:
        reward = 1       # 极小奖励，不诱惑
    elif next_state == 4:
        reward = 1000    # 巨大奖励
    else:
        reward = 0

    return next_state, reward

# ====================== Q更新 ======================
def update_Q(s, a, r, s_next):
    Q[s,a] = (1 - alpha) * Q[s,a] + alpha * (r + gamma * np.max(Q[s_next]))

# ====================== 训练 ======================
print("开始训练...")
for episode in range(1000):
    s = 0
    for _ in range(20):
        a = choose_action(s)
        s_next, r = step(s, a)
        update_Q(s, a, r, s_next)
        s = s_next
        if s == goal_state:
            break

# ====================== 测试结果 ======================
print("\n===== Q 表 =====")
print(np.round(Q, 2))

print("\n最终路径：")
s = 0
path = [s]
for _ in range(10):
    a = np.argmax(Q[s])
    s, _ = step(s, a)
    path.append(s)
    if s == goal_state:
        break
print("路径:", path)
