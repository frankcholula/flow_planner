from dataclasses import dataclass


@dataclass
class LunarLanderConfig:
    policy_type: str = "MlpPolicy"
    total_timesteps: int = int(10e5)
    env_name: str = "LunarLander-v3"
    horizon: int = 100
    obs_dim: int = 8
    action_dim: int = 2
    dataset_name: str = "LunarLanderContinuous-v3/ppo-1000-deterministic-v1"
