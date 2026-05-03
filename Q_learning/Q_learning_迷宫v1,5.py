# ======= 在v1.4的基础上添加日志运行时间，日志追加功能========
import numpy as np
import os
import logging
from datetime import datetime

# ========== 1. 固定路径：全部保存在当前脚本文件夹 ==========
script_dir = os.path.dirname(os.path.abspath(__file__))

# 日志文件完整路径
log_file = os.path.join(script_dir, "train_append.log")
# 模型文件夹
folder_path = os.path.join(script_dir, "best_models")
npy_path = os.path.join(folder_path, "best_q_table.npy")
csv_path = os.path.join(folder_path, "best_q_table.csv")



# ========== 2. 日志配置：带时间 + 追加不覆盖 + 仅写入文件 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8')
    ]
)
log = logging.getLogger()


# 创建模型文件夹 + 警告/错误处理
if not os.path.exists(folder_path):
    try:
        os.mkdir(folder_path)
        log.info(f"模型文件夹创建成功：{folder_path}")
    except Exception as e:
        log.error(f"模型文件夹创建失败：{str(e)}")
else:
    log.info(f"模型文件夹已存在：{folder_path}")

# ========== 3. 环境与超参数 ==========
n_states = 5
actions = [0, 1]
goal_state = 4

alpha = 0.1
gamma = 0.99
epsilon = 0.1

# 参数警告
if epsilon > 0.3:
    log.warning(f"探索率 epsilon = {epsilon} 过高，可能导致收敛变慢")
if alpha > 0.5:
    log.warning(f"学习率 alpha = {alpha} 过大，训练可能不稳定")

# ========== 4. 加载历史最优模型 ==========
def load_best_model():
    try:
        if os.path.exists(npy_path):
            Q = np.load(npy_path)
            log.info("成功加载最优模型，继续训练")
            return Q
        else:
            log.info("未找到历史模型，从头开始训练")
            return np.zeros((n_states, len(actions)))
    except Exception as e:
        log.error(f"加载模型失败：{str(e)}")
        return np.zeros((n_states, len(actions)))

Q = load_best_model()
best_total_reward = -1

# ========== 5. 工具函数 ==========
def choose_action(state):
    if np.random.uniform() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state, :])

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

def update_Q(s, a, r, s_next):
    try:
        Q[s,a] = (1 - alpha) * Q[s,a] + alpha * (r + gamma * np.max(Q[s_next]))
    except Exception as e:
        log.error(f"Q表更新错误：s={s}, a={a}, 错误信息={str(e)}")

def save_best_model():
    try:
        np.save(npy_path, Q)
        np.savetxt(csv_path, Q, fmt="%.4f", delimiter=",")
        log.info("出现新最高奖励，已保存最优模型(npy+csv)")
    except Exception as e:
        log.error(f"最优模型保存失败：{str(e)}")

# ========== 6. 开始训练 ==========
log.info("-" * 60)
log.info("===== 启动新一轮训练 =====")

for episode in range(10):
    s = 0
    total_reward = 0

    log.info(f"---------- 第 {episode} 轮 ----------")

    for step_idx in range(20):
        a = choose_action(s)
        s_next, r = step(s, a)

        log.info(f"当前状态:{s}  选择动作:{a}  下一状态:{s_next}  奖励:{r}")
        log.info(f"更新前Q表:\n{np.round(Q, 2)}")

        update_Q(s, a, r, s_next)
        total_reward += r

        log.info(f"更新后Q表:\n{np.round(Q, 2)}")

        s = s_next
        if s == goal_state:
            log.info("本轮成功到达终点")
            break

    # 训练警告
    if total_reward == 0:
        log.warning(f"第 {episode} 轮总奖励为 0，训练效果差！")

    log.info(f"本轮总奖励: {total_reward}")
    if total_reward >= best_total_reward:
        best_total_reward = total_reward
        save_best_model()
    log.info(f"当前历史最高奖励: {best_total_reward}")

# ========== 7. 训练结束 & 最终路径 ==========
log.info("===== 本轮训练结束 =====")
log.info(f"最终Q表:\n{np.round(Q, 2)}")

try:
    s = 0
    path = [s]
    for _ in range(10):
        a = np.argmax(Q[s])
        s, _ = step(s, a)
        path.append(s)
        if s == goal_state:
            break
    log.info(f"最终最优路径: {path}")
except Exception as e:
    log.error(f"生成最优路径失败：{str(e)}")

# 终端极简提示
print(f"✅ 训练完成，带时间日志已保存至：{log_file}")
log.info(f"训练全部完成！日志文件路径：{log_file}")