import gymnasium as gym
import minari
from src.conf.environment import CarRacingConfig
from tqdm import tqdm

config = CarRacingConfig()
LEVEL = "simple"
DATASET_NAME = f"{config.category}/{config.env_name}/{LEVEL}-v0"
TOTAL_EPS = 250

print(f"Preparing to collect {TOTAL_EPS} episodes for dataset: {DATASET_NAME}")

env = gym.make(config.env_name, render_mode="rgb_array", continuous=True)
data_collector = minari.DataCollector(env)

print(f"Collecting data...")
for _ in tqdm(range(TOTAL_EPS), desc="Collecting episodes"):
    obs, _ = data_collector.reset()

    while True:
        action = data_collector.action_space.sample()
        obs, reward, terminated, truncated, info = data_collector.step(action)

        if terminated or truncated:
            break
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

print(f"\nSuccessfully created and saved dataset: {DATASET_NAME}")
data_collector.close()
