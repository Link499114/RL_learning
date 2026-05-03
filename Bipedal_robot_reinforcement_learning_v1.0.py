# 导入库
import gymnasium as gym
from stable_baselines3 import PPO

# ======================
# 1. 创建双足机器人环境
# ======================
# BipedalWalker-v3 是标准简单双足机器人仿真环境
env = gym.make("BipedalWalker-v3", render_mode="human")  # render_mode="human" 显示画面

# ======================
# 2. 用PPO算法训练
# ======================
# PPO 是最适合机器人控制的强化学习算法之一
model = PPO(
    "MlpPolicy",       # 神经网络策略
    env,               # 双足机器人环境
    verbose=1,         # 打印训练日志
    learning_rate=3e-4,
    n_steps=2048,
)

# 开始训练（10万步，新手足够看到效果）
print("开始训练双足机器人...")
model.learn(total_timesteps=100000)

# 保存训练好的模型
model.save("bipedal_robot_model")

# ======================
# 3. 测试训练结果
# ======================
print("测试训练好的机器人...")
obs, _ = env.reset()

while True:
    # 模型预测动作
    action, _states = model.predict(obs, deterministic=True)
    
    # 执行动作
    obs, reward, done, truncated, info = env.step(action)
    
    # 摔倒或结束 => 重置
    if done or truncated:
        obs, _ = env.reset()
