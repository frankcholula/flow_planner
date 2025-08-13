from tqdm.auto import tqdm
from minari import DataCollector
from stable_baselines3 import PPO, A2C, TD3
from src.utils.args import parse_agent_args
from rl_zoo3.utils import (
    get_latest_run_id,
    get_saved_hyperparams,
    get_wrapper_class,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import gymnasium as gym
import torch
from pprint import pprint

ALGO_DICT = {"ppo": PPO, "a2c": A2C, "td3": TD3}


def generate_dataset(args):
    torch.manual_seed(args.seed)

    # load hyperparams
    log_path = os.path.join("logs", args.algo)
    latest_run_id = get_latest_run_id(log_path=log_path, env_name=args.env)
    model_path = os.path.join(log_path, f"{args.env}_{latest_run_id}/{args.env}.zip")
    stats_path = os.path.join(log_path, f"{args.env}_{latest_run_id}/{args.env}")
    hyperparams, _ = get_saved_hyperparams(stats_path=stats_path, test_mode=True)
    pprint(hyperparams)


    minari_env = gym.make(args.env, **hyperparams.get("env_kwargs", {}))
    if "env_wrapper" in hyperparams:
        wrapper_class = get_wrapper_class(hyperparams=hyperparams, key="env_wrapper")
        minari_env = wrapper_class(minari_env)
    if "frame_stack" in hyperparams:
        n_stack = hyperparams["frame_stack"]
        minari_env = gym.wrappers.FrameStackObservation(minari_env, n_stack)
    
    env = DataCollector(minari_env)
    agent = ALGO_DICT[args.algo].load(model_path)
    for i in tqdm(range(args.total_episodes)):
        obs, _ = env.reset()
        while True:
            action, _ = agent.predict(obs)
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    # dataset = env.create_dataset(
    #     dataset_id=f"Box2D/{args.env}/{args.level}-v{args.version}",
    #     algorithm_name=args.algo,
    #     code_permalink="https://github.com/frankcholula/flow_planner",
    #     author="Frank Lu",
    #     author_email="lu.phrank@gmail.com",
    #     description=f"Behavioral cloning dataset for {args.env} using {args.algo}",
    #     eval_env=args.env,
    # )


def main():
    args = parse_agent_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
