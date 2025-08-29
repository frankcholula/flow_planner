import torch
from src.pipelines.sampling import (
    generate_trajectory,
    generate_diffusion_trajectory,
)
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver
from src.pipelines.lunarlander.visualizers import visualize_trajectories as lvt
from src.pipelines.lunarlander.visualizers import plot_reward_histogram
from src.utils.loggers import EpisodeTimer
import numpy as np
import wandb
import matplotlib.pyplot as plt
import random


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


class WrappedConditionalModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, **extras):
        # using time embedding
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        return self.model(x, t, c)


def evaluate_open_loop_diffusion(
    env, model, noise_scheduler, stats, input_dim, args, logger=None, dataset=None
):
    model.eval()
    cond_dict = None

    if args.condition_on == "start_obs":
        start_observation, _ = env.reset()
        cond_dict = {"start_obs": torch.from_numpy(start_observation).to(args.device)}

    elif args.condition_on == "start_obs_waypoint":
        if dataset is None:
            raise ValueError("Dataset must be provided for waypoint evaluation.")

        valid_episode_found = False
        while not valid_episode_found:
            episode = dataset[random.choice(range(len(dataset)))]
            if len(episode.observations) > args.horizon:
                valid_episode_found = True

        start_observation = episode.observations[0]
        goal_observation = episode.observations[args.horizon - 1]

        print(
            f"Start obs (x,y): ({start_observation[0]:.4f}, {start_observation[1]:.4f})"
        )
        print(
            f"Goal obs (x,y):  ({goal_observation[0]:.4f}, {goal_observation[1]:.4f})"
        )
        cond_dict = {
            "start_obs_goal": (
                torch.from_numpy(start_observation).to(args.device),
                torch.from_numpy(goal_observation).to(args.device),
            )
        }

    elif args.condition_on == "start_obs_goal":
        start_observation, _ = env.reset()
        goal_observation = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32
        ).to(args.device)
        cond_dict = {
            "start_obs_goal": (
                torch.from_numpy(start_observation).to(args.device),
                goal_observation,
            )
        }

    trajectory_fn = lambda: generate_diffusion_trajectory(
        model=model,
        noise_scheduler=noise_scheduler,
        stats=stats,
        args=args,
        input_dim=input_dim,
        condition=cond_dict,
    )

    fig, ax = lvt(trajectory_fn=trajectory_fn, num_trajectories=5)
    if logger is not None and fig is not None:
        logger.log({"diffusion_trajectory_plot": wandb.Image(fig)})

    model.train()
    return fig, ax


def evaluate_open_loop(env, model, stats, input_dim, args, logger=None):
    model.eval()
    cond_dict = None
    if args.condition_on == "start_obs":
        start_observation, _ = env.reset()
        cond_dict = {"start_obs": torch.from_numpy(start_observation)}
    elif args.condition_on == "start_obs_goal":
        start_observation, _ = env.reset()
        if args.environment == "LunarLander-v3":
            goal_observation = torch.tensor(
                [0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32
            )
        elif (
            args.environment == "BipedalWalker-v3" or args.environment == "CarRacing-v3"
        ):
            goal_observation = torch.from_numpy(start_observation)
        else:
            raise ValueError(f"Unknown environment: {args.environment}")
        cond_dict = {
            args.condition_on: (torch.from_numpy(start_observation), goal_observation)
        }
    elif args.condition_on == "reward":
        # TODO: implement reward conditioning
        pass
    if cond_dict is not None:
        wrapped_vf = WrappedConditionalModel(model)
    else:
        wrapped_vf = WrappedModel(model)
    T = torch.linspace(0, 1, 10)
    solver = ODESolver(velocity_model=wrapped_vf)
    trajectory_fn = lambda: generate_trajectory(
        stats=stats,
        solver=solver,
        T=T,
        input_dim=input_dim,
        horizon=args.horizon,
        condition=cond_dict,
        solver_method=args.solver_method,
        batch_size=args.inference_batch_size,
        step_size=args.step_size,
        return_intermediates=False,
        model_target=args.model_target,
    )
    fig, ax = lvt(trajectory_fn=trajectory_fn, num_trajectories=5)
    if logger is not None:
        logger.log({"trajectory plot": wandb.Image(fig)})
    model.train()
    return fig, ax


def evaluate_policy_mpc(
    env,
    planner_fn,
    num_episodes,
    condition_type: str,
    args,
    goal_obs=None,
    max_episode_length=300,
    replan_freq=1,
    render=False,
    visualize=True,
    dataset=None,
):
    rewards = []
    print(
        f"\n--- Starting MPC Evaluation (Condition Type: {condition_type}) and Replan Frequency: {replan_freq} ---"
    )

    timed_planner_fn = EpisodeTimer(planner_fn)
    for eps in range(num_episodes):
        obs, _ = env.reset()
        total_rew = 0
        actions_plan = None
        plan_start_step = -1

        reference_episode = None
        if condition_type == "start_obs_waypoint":
            if dataset is None:
                raise ValueError("Dataset must be provided for waypoint MPC.")
            # Find an episode that is long enough to serve as a reference
            valid_episode_found = False
            while not valid_episode_found:
                reference_episode = dataset[random.choice(range(len(dataset)))]
                if (
                    len(reference_episode.observations) > 1
                ):  # Just need more than 1 step
                    valid_episode_found = True

        for t in range(max_episode_length):
            if render:
                env.render()

            if t % replan_freq == 0:
                start_obs_tensor = torch.from_numpy(obs).to(
                    args.device
                )  # Ensure tensor is on correct device
                cond_dict = {}

                if condition_type == "start_obs":
                    cond_dict = {"start_obs": start_obs_tensor}

                elif condition_type == "start_obs_waypoint":
                    waypoint_index = t + args.horizon - 1
                    # Clamp the index to the end of the reference trajectory if we go past it
                    if waypoint_index >= len(reference_episode.observations):
                        waypoint_index = len(reference_episode.observations) - 1

                    waypoint_obs = torch.from_numpy(
                        reference_episode.observations[waypoint_index]
                    ).to(args.device)
                    cond_dict = {"start_obs_goal": (start_obs_tensor, waypoint_obs)}

                elif condition_type == "start_obs_goal":
                    if goal_obs is None:
                        raise ValueError("goal_obs must be provided for start_obs_goal")
                    cond_dict = {
                        "start_obs_goal": (start_obs_tensor, goal_obs.to(args.device))
                    }

                elif condition_type == "unconditional":
                    cond_dict = None
                else:
                    raise ValueError(f"Unknown condition type: {condition_type}")

                _, actions_plan = timed_planner_fn(cond_dict)
                plan_start_step = t

            action_index_in_plan = t - plan_start_step
            if actions_plan is None or action_index_in_plan >= len(actions_plan):
                break

            action_to_take = actions_plan[action_index_in_plan].cpu().numpy()

            obs, rew, terminated, truncated, info = env.step(action_to_take)
            total_rew += rew

            if terminated or truncated:
                break

        rewards.append(total_rew)
        print(
            f"Episode {eps + 1}/{num_episodes} finished. Total Reward: {total_rew:.2f}"
        )

        timed_planner_fn.report_average_time()
        timed_planner_fn.reset()

    avg_model_reward = np.mean(rewards)
    std_model_reward = np.std(rewards)
    print(
        f"Average MPC Reward over {num_episodes} episodes: {avg_model_reward:.2f} +/- {std_model_reward:.2f}"
    )

    fig = None
    if visualize:
        fig = plot_reward_histogram(
            rewards, title=f"Reward Distribution (Replan Freq: {replan_freq})"
        )
        if fig:
            plt.close(fig)

    return rewards, fig
