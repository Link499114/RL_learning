import numpy as np
import os

# ====================== 环境设定 ======================
# 0:起点 | 1:陷阱 | 2:安全路 | 3:安全路 | 4:终点
n_states = 5
actions = [0, 1]  # 0=去正确路, 1=去陷阱
goal_state = 4


# 超参数
alpha = 0.1
gamma = 0.99  # 👈 测试 0.99 / 0.1
epsilon = 0.1

# 获取【当前脚本所在的文件夹路径】
script_dir = os.path.dirname(os.path.abspath(__file__))
print("脚本当前路径：", script_dir)

# 要创建的文件夹路径（在脚本同目录下）
folder_path = os.path.join(script_dir, "best_models")

# --------------------- 拼接完整保存路径，避免报错 ------------------------
npy_path = os.path.join(folder_path, "best_q_table.npy")  # 科学计算用
csv_path = os.path.join(folder_path, "best_q_table.csv")  # Excel查看用

#如果存放模型的文件夹不存在-->创建存放模型的文件夹
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


# ====================== ✅ 加载之前训练好的最好模型 ======================
def load_best_model():
    if os.path.exists(npy_path):
        Q = np.load(npy_path)
        print("✅ 成功加载 最优模型！继续训练...")
        return Q
    else:
        print("⚠️  未找到模型，从头开始训练...")
        return np.zeros((n_states, len(actions)))
    


# Q表
Q = load_best_model()

# ====================== 保存最好模型 ======================

# 记录历史最好奖励（初始设为 -1）
best_total_reward = -1


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


# ====================== 保存：最好模型（npy + csv） ======================

def save_best_model():
    # 保存 .npy （给模型加载）
    np.save(npy_path, Q)
    
    # 保存 .csv （给你 Excel 打开看）
    np.savetxt(csv_path, Q, fmt="%.4f", delimiter=",")
    
    print("✅ 新纪录！已保存最好模型（npy + csv）")


# ====================== 训练 ======================
print("开始训练...")
for episode in range(10):
    s = 0
    total_reward = 0  # 记录本局总奖励  
    print("\n轮数：",episode)
    for _ in range(20):
        a = choose_action(s)
        s_next, r = step(s, a)

        print("\nchoose action:",s,a)   #选择状态和动作
        print("s_next,r:",s_next,r)     #下一个状态和奖励
        print("Q_old:")                 
        print(np.round(Q, 2))           #未更新的Q表

        update_Q(s, a, r, s_next)
        total_reward += r  # 累计奖励

        
        print("Q_update:")             
        print(np.round(Q, 2))           #更新后的Q表

        s = s_next
        
        print("goal_state:",goal_state) #目标状态
    
    
        if s == goal_state:         
            break

    # ====================== 判断是否为最好模型 ======================
    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        save_best_model()  # 保存 npy + csv

    print("第", episode, "奖励：", total_reward)
    print(f"第 {episode} 轮 | 当前最高奖励: {best_total_reward}")


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
