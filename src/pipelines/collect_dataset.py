from tqdm.auto import tqdm
from minari import DataCollector
from stable_baselines3 import PPO, A2C, TD3
from src.utils.args import parse_agent_args
from rl_zoo3.utils import get_latest_run_id
import os
import gymnasium as gym
import torch

ALGO_DICT = {"ppo": PPO, "a2c": A2C, "td3": TD3}


def generate_dataset(args):
    torch.manual_seed(args.seed)

    log_path = os.path.join("logs", args.algo)
    latest_run_id = get_latest_run_id(log_path=log_path, env_name=args.env)
    model_path = os.path.join(log_path, f"{args.env}_{latest_run_id}/{args.env}.zip")
    agent = ALGO_DICT[args.algo].load(model_path)
    env_kwargs = {"rgb_array": True}
    if args.env == "LunarLanderContinuous-v3":
        env_kwargs.update({"continuous": True})
    elif args.env == "BipedalWalkerHardcore-v3":
        env_kwargs.update({"hardcore": True})
    env = DataCollector(gym.make(id=args.env, **env_kwargs))

    for i in tqdm(range(args.total_episodes)):
        obs, _ = env.reset()
        while True:
            action, _ = agent.predict(obs)
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    dataset = env.create_dataset(
        dataset_id=f"Box2D/{args.env}/{args.algo}-v{args.version}",
        algorithm_name=args.algo,
        code_permalink="https://github.com/frankcholula/flow_planner",
        author="Frank Lu",
        author_email="lu.phrank@gmail.com",
        description=f"Behavioral cloning dataset for {args.env} using {args.algo}",
        eval_env=args.env,
    )


def main():
    args = parse_agent_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
