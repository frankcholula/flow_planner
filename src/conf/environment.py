from dataclasses import dataclass


@dataclass
class LunarLanderConfig:
    env_name: str = "LunarLander-v3"
    dataset_name: str = "Box2D/LunarLanderContinuous-v3/expert-v0"
    obs_dim: int = 8
    action_dim: int = 2
    category: str = "Box2D"


@dataclass
class CarRacingConfig:
    env_name: str = "CarRacing-v3"
    dataset_name: str = "Box2D/CarRacing-v3/latent-v0"
    obs_dim: int = 16
    action_dim: int = 3
    category: str = "Box2D"


@dataclass
class BipedalWalkerConfig:
    env_name: str = "BipedalWalker-v3"
    dataset_name: str = "Box2D/BipedalWalker-v3/expert-v0"
    obs_dim: int = 24
    action_dim: int = 4
    category: str = "Box2D"


@dataclass
class HopperConfig:
    env_name: str = "Hopper-v5"
    dataset_name: str = "mujoco/hopper/expert-v0"
    obs_dim: int = 11
    action_dim: int = 3
    category: str = "MuJoCo"


@dataclass
class Walker2dConfig:
    env_name: str = "Walker2d-v5"
    dataset_name: str = "mujoco/walker2d/expert-v0"
    obs_dim: int = 17
    action_dim: int = 6
    category: str = "MuJoCo"
