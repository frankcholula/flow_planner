from tqdm.auto import tqdm
from minari import DataCollector

from stable_baselines3 import PPO, A2C, TD3
from src.utils.args import parse_agent_args
from rl_zoo3.utils import (
    get_latest_run_id,
    get_saved_hyperparams,
    get_wrapper_class,
    get_model_path,
)

import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
)

import os
import torch

ALGOS = {"ppo": PPO, "ppo_lstm": PPO, "a2c": A2C, "td3": TD3}


def make_env(env_name, hyperparams):
    env_kwargs = {"render_mode": "rgb_array"}
    if env_name in ["LunarLanderContinuous-v3", "CarRacing-v3"]:
        env_kwargs.update({"continuous": True})
    env = gym.make(env_name, **env_kwargs)
    data_collector = DataCollector(env)
    if "env_wrapper" in hyperparams:
        print("Wrapping environment with custom wrappers")
        for wrapper_config in hyperparams["env_wrapper"]:
            wrapper_name = list(wrapper_config.keys())[0]
            # some janky replacement because of the gymnasium implementation of FrameStack
            if "grayscaleobservation" in wrapper_name.lower():
                wrapper_config[wrapper_name]["keep_dim"] = False
        wrapper_class = get_wrapper_class(hyperparams=hyperparams, key="env_wrapper")
        env = wrapper_class(data_collector)

    if "frame_stack" in hyperparams:
        print("Wrapping environment with FrameStackObservation")
        n_stack = hyperparams["frame_stack"]
        env = FrameStackObservation(env, n_stack)
    return env, data_collector


def generate_dataset(args):
    torch.manual_seed(args.seed)
    # load hyperparams
    latest_run_id = get_latest_run_id(log_path="logs", env_name=args.env)
    _, model_path, log_path = get_model_path(
        exp_id=latest_run_id, folder="logs", algo=args.algo, env_name=args.env
    )
    stats_path = os.path.join(log_path, f"{args.env}")
    hyperparams, _ = get_saved_hyperparams(stats_path=stats_path, test_mode=True)
    minari_env, data_collector = make_env(args.env, hyperparams)
    agent = ALGOS[args.algo].load(model_path, env=minari_env)
    for i in tqdm(range(args.total_episodes), desc="Collecting episodes"):
        obs, _ = minari_env.reset()
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = minari_env.step(action)
            if terminated or truncated:
                break

    dataset = data_collector.create_dataset(
        dataset_id=f"Box2D/{args.env}/{args.level}-v{args.version}",
        algorithm_name=args.algo,
        code_permalink="https://github.com/frankcholula/flow_planner",
        author="Frank Lu",
        author_email="lu.phrank@gmail.com",
        description=f"Behavioral cloning dataset for {args.env} using {args.algo}",
        eval_env=minari_env,
    )


def main():
    args = parse_agent_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
