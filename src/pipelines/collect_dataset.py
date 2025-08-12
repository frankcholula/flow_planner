from tqdm.auto import tqdm
from minari import DataCollector
from stable_baselines3 import PPO, A2C, TD3
import gymnasium as gym
from src.utils.args import parse_agent_args

ALGO_DICT = {"ppo": PPO, "a2c": A2C, "td3": TD3}


def generate_dataset(args):
    if args.env == "LunarLanderContinous-v3":
        env = DataCollector(gym.make(args.env, continuous=True))
    elif args.env == "BipedalWalker-v3":
        env = DataCollector(gym.make(args.env))
    elif args.env == "CarRacing-v3":
        env = DataCollector(gym.make(args.env))
    elif args.env == "BipedalWalkerHardcore-v3":
        env = DataCollector(gym.make(args.env), hardcore=True)
    path = f"logs/{args.env}/best_model"

    agent = ALGO_DICT[args.algo].load(path)

    for i in tqdm(range(args.total_episodes)):
        obs, _ = env.reset(seed=args.seeds)
        while True:
            action, _ = agent.predict(obs)
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    dataset = env.create_dataset(
        dataset_id=f"Box2D/{args.env}/{args.algo}-v{args.version}",
        algorithm_name=args.algo,
        code_permalink="https://github.com/frankcholula/flow_planner/blob/src/pipelines/collect_dataset.py",
        author="Frank Lu",
        author_email="lu.phrank@gmail.com",
        description=f"Behavioral cloning dataset for {args.envenvironmenvent} using {args.algo}",
        eval_env=args.env,
    )


def main():
    args = parse_agent_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
