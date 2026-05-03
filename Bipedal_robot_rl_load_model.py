from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")
model = PPO.load("bipedal_robot_model")

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
