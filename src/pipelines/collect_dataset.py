from tqdm.auto import tqdm
from minari import DataCollector
from minari.serialization import serialize_space, deserialize_space

from stable_baselines3 import PPO, A2C, TD3
from src.utils.args import parse_agent_args
from rl_zoo3.utils import (
    get_latest_run_id,
    get_saved_hyperparams,
    get_wrapper_class,
    get_model_path,
    create_test_env,
)

import gymnasium as gym
from gymnasium.wrappers import (
    FrameStackObservation,
    TransformObservation,
)

import os
import torch
from pprint import pprint
import numpy as np
ALGOS = {"ppo": PPO, "ppo_lstm": PPO, "a2c": A2C, "td3": TD3}


def generate_dataset(args):
    torch.manual_seed(args.seed)

    # load hyperparams
    # latest_run_id = get_latest_run_id(log_path="logs", env_name=args.env)
    latest_run_id = 3
    _, model_path, log_path = get_model_path(
        exp_id=latest_run_id, folder="logs", algo=args.algo, env_name=args.env
    )
    stats_path = os.path.join(log_path, f"{args.env}")
    hyperparams, _ = get_saved_hyperparams(stats_path=stats_path, test_mode=True)
    minari_env = gym.make(
        args.env,
        **hyperparams.get("env_kwargs", {}),
        render_mode="rgb_array",
        continuous=True,
        max_episode_steps=100
    )
    if "env_wrapper" in hyperparams:
        for wrapper_config in hyperparams["env_wrapper"]:
            wrapper_name = list(wrapper_config.keys())[0]
            # some janky replacement because of the gymnasium implementation of FrameStack
            if "grayscaleobservation" in wrapper_name.lower():
                wrapper_config[wrapper_name]["keep_dim"] = False
        wrapper_class = get_wrapper_class(hyperparams=hyperparams, key="env_wrapper")
        minari_env = wrapper_class(minari_env)

    if "frame_stack" in hyperparams:
        n_stack = hyperparams["frame_stack"]
        minari_env = FrameStackObservation(minari_env, n_stack)
        # TODO: This is to avoid the stupid image compression by minari
        new_obs_space = gym.spaces.Box(
            low=0, high=255, shape=minari_env.observation_space.shape, dtype=np.float32
        )
        minari_env = TransformObservation(
            env=minari_env,
            func=lambda obs: obs.astype(np.float32),
            observation_space=new_obs_space,
        )

    agent = ALGOS[args.algo].load(model_path)
    env = DataCollector(minari_env)
    print(env.observation_space)
    print(env.action_space.shape)

    for i in tqdm(range(args.total_episodes)):
        obs, _ = env.reset()
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished.")
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


def generate_raw_dataset_with_collector(args):
    # --- 1. Find the local expert model and its configuration ---

    # latest_run_id = get_latest_run_id(log_path="logs", env_name=args.env)
    latest_run_id = 1
    _, model_path, log_path = get_model_path(
        exp_id=latest_run_id, folder="logs", algo=args.algo, env_name=args.env
    )
    stats_path = os.path.join(log_path, f"{args.env}")
    hyperparams, _ = get_saved_hyperparams(stats_path=stats_path, test_mode=True)
    
    # --- 2. Create the EXPERT'S environment (for decision making) ---
    # This environment perfectly replicates the training conditions (wrappers, normalization, etc.)
    # We will use this to process observations to get the correct agent action.
    expert_env = create_test_env(
        env_id=args.env,
        n_envs=1,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=None,
        should_render=False,
        hyperparams=hyperparams,
    )

    # --- 3. Create the RAW environment and wrap with Minari (for data collection) ---
    # This is the environment we will step through and record.
    raw_env = gym.make(args.env, render_mode="rgb_array")
    collector_env = DataCollector(raw_env)

    # --- 4. Load the agent ---
    agent = ALGOS[args.algo].load(model_path, env=expert_env)
    # fake_agent = ALGOS[args.algo].load(model_path, env=collector_env)
    # # --- 5. Run the simulation loop ---
    print(f"Generating {args.total_episodes} expert episodes...")

    print(f"Raw observation space: {collector_env.observation_space.shape}")
    print(f"Processed observation space: {expert_env.observation_space.shape}")
    
    
    for _ in tqdm(range(args.total_episodes)):
        processed_obs = expert_env.reset()
        obs, _ = collector_env.reset(seed=args.seed)
        while True:
            # generate action from expert env taking in processed obs
            agent_action, _ = agent.predict(processed_obs, deterministic=True)
            processed_obs, rew, dones, info = expert_env.step(agent_action)
            obs, rew, terminated, truncated, info = collector_env.step(agent_action.squeeze(0))
            if terminated or truncated:
                print("Episode finished.")
                break


    #     while not (terminated or truncated):
    #         # A) Process the raw observation using the expert's environment
    #         processed_obs = expert_env.normalize_obs(raw_obs)
            
    #         # B) Get the expert's action based on the processed view
    #         action, _ = agent.predict(processed_obs, deterministic=True)
            
    #         # C) Step the Minari-wrapped environment using this action.
    #         # Minari will record the pair: (raw observation, expert action)
    #         obs, reward, terminated, truncated, info = collector_env.step(action[0])

    #     raw_obs, _ = collector_env.reset()
    #     processed_obs = expert_env.reset()

    # --- 6. Create and save the Minari dataset ---
    # dataset = collector_env.create_dataset(
        # algorithm_name=args.algo,
        # author=args.organization
    # )
    
    # collector_env.close()
    # expert_env.close()

def main():
    args = parse_agent_args()
    # generate_dataset(args)
    generate_raw_dataset_with_collector(args)

if __name__ == "__main__":
    main()
