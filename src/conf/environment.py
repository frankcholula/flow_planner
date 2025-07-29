from dataclasses import dataclass


@dataclass
class LunarLanderConfig:
    policy_type: str = "MlpPolicy"
    total_timesteps: int = int(10e5)
    env_name: str = "LunarLander-v3"
    dataset_name: str = "LunarLanderContinuous-v3/ppo-1000-deterministic-v1"
    obs_dim: int = 8
    action_dim: int = 2

class HopperConfig:
    env_name: str = "Hopperm-v5"
    dataset_name: str = "mujoco/hopper/expert-v0"
    obs_dim: int = 11
    action_dim: int = 3
