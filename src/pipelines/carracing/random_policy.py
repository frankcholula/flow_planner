import gymnasium as gym
import minari
from src.conf.environment import CarRacingConfig

config = CarRacingConfig()
LEVEL = "simple"
DATASET_NAME = f"{config.category}/{config.env_name}/{LEVEL}-v0"
TOTAL_EPS = 250

print(f"Preparing to collect {TOTAL_EPS} episodes for dataset: {DATASET_NAME}")

env = gym.make(config.env_name, render_mode="rgb_array", continuous=True)
data_collector = minari.DataCollector(env)

print(f"Collecting data...")
obs, info = data_collector.reset()
episodes_collected = 0
while episodes_collected < TOTAL_EPS:
    action = data_collector.action_space.sample()
    obs, reward, terminated, truncated, info = data_collector.step(action)
    if terminated or truncated:
        episodes_collected += 1
        print(f"Episodes collected: {episodes_collected}/{TOTAL_EPS}")
        obs, info = data_collector.reset()

print("\nData collection complete.")

dataset = data_collector.create_dataset(
    dataset_id=DATASET_NAME,
    algorithm_name="Random Policy",
    code_permalink="https://github.com/frankcholula/flow_planner",
    author="Frank Lu",
    author_email="lu.phrank@gmail.com",
    description=f"VAE training dataset for {config.env_name} using random policy",
    eval_env=data_collector,
)

print(f"\nSuccessfully created and saved dataset:")
print(dataset)

data_collector.close()
