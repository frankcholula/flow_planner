import rl_zoo3
from tqdm.auto import tqdm
from minari import DataCollector
from stable_baselines3 import PPO, A2C, TD3
from src.utils.args import parse_agent_args
from rl_zoo3.utils import (
    get_latest_run_id,
    get_saved_hyperparams,
    get_wrapper_class,
    get_model_path,
    create_test_env,
)
from rl_zoo3.wrappers import FrameSkip, YAMLCompatResizeObservation

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation

import os
import torch
from pprint import pprint

ALGOS = {"ppo": PPO, "ppo_lstm": PPO, "a2c": A2C, "td3": TD3}


def generate_dataset(args):
    torch.manual_seed(args.seed)

    # load hyperparams
    latest_run_id = get_latest_run_id(log_path="logs", env_name=args.env)
    _, model_path, log_path = get_model_path(
        exp_id=latest_run_id, folder="logs", algo=args.algo, env_name=args.env
    )
    stats_path = os.path.join(log_path, f"{args.env}")
    hyperparams, _ = get_saved_hyperparams(stats_path=stats_path, test_mode=True)
    pprint(hyperparams)

    # create vec env
    # vec_env = create_test_env(
    #     env_id=args.env,
    #     n_envs=1,
    #     stats_path=stats_path,
    #     seed=args.seed,
    #     log_dir=None,
    #     should_render=False,
    #     hyperparams=hyperparams,
    # )
    # minari_env = vec_env.envs[0]
    # # print(minari_env.observation_space.shape)
    # minari_env = gym.make(args.env, **hyperparams.get("env_kwargs", {}))

    # print("Before:", minari_env.observation_space.shape)
    # if "env_wrapper" in hyperparams:
    #     wrapper_class = get_wrapper_class(hyperparams=hyperparams, key="env_wrapper")
    #     minari_env = wrapper_class(minari_env)
    #     print("During:", minari_env.observation_space.shape)

    # if "frame_stack" in hyperparams:
    #     n_stack = hyperparams["frame_stack"]
    #     minari_env = FrameStackObservation(minari_env, n_stack)
# 
    # print("After:", minari_env.observation_space.shape)


    minari_env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
    minari_env = FrameSkip(minari_env, skip=2)
    minari_env = ResizeObservation(minari_env, shape=(64, 64))
    minari_env = GrayscaleObservation(minari_env, keep_dim=False)
    minari_env = FrameStackObservation(minari_env, 2)

    print("Final:", minari_env.observation_space.shape)  

    # agent = ALGOS[args.algo].load(model_path, env=minari_env)
    agent = PPO.load(model_path, env=minari_env)
    print("Agent observation space:", agent.observation_space.shape)
    env = DataCollector(minari_env)

    for i in tqdm(range(args.total_episodes)):
        obs, _ = env.reset()
        print("Observation shape:", obs.shape)
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            print("Action shape:", action.shape)
            try:
                obs, rew, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            except Exception as e:
                print(action)
                print(rew)
                print(terminated)
                print(truncated)
                print(info)
                raise e

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
