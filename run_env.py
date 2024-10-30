from env_file import UR5eEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = UR5eEnv(render_mode="human")

check_env(env, warn=True)

# Initialize the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)  # Adjust timesteps as needed

# Save the trained model
model.save("ppo_ur10e_reaching")

# To visualize the trained model, you can use:
# model = PPO.load("ppo_ur10e_reaching")
# obs = env.reset()
# for _ in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()

env.close()
