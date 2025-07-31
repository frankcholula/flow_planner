import torch
from src.pipelines.sampling import generate_trajectory
from flow_matching.utils import ModelWrapper
from flow_matching.solver import ODESolver
from src.pipelines.lunarlander.visualizers import visualize_trajectories as lvt
import wandb

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)


class WrappedConditionalModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, **extras):
        return self.model(x, t, c)


def evaluate_open_loop(env, model, stats, input_dim, args, logger=None):
    if args.condition_on == "start_obs":
        start_observation, _ = env.reset()
        condition_dict = {"start_obs": torch.from_numpy(start_observation)}
    elif args.condition_on == "start_obs_goal":
        start_observation, _ = env.reset()
        goal_observation = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.float32)
        condition_dict = {
            args.condition_on: (torch.from_numpy(start_observation), goal_observation)
        }
    elif args.condition_on == "reward":
        # TODO: implement reward conditioning
        pass
    wrapped_vf = WrappedConditionalModel(model)
    T = torch.linspace(0, 1, 10)
    solver = ODESolver(velocity_model=wrapped_vf)
    trajectory_fn = lambda: generate_trajectory(
        stats=stats,
        solver=solver,
        T=T,
        input_dim=input_dim,
        horizon=args.horizon,
        condition=condition_dict if args.condition_on else None,
        solver_method=args.solver_method,
        batch_size=args.inference_batch_size,
        step_size=args.step_size,
        return_intermediates=False,
    )
    fig, ax = lvt(trajectory_fn=trajectory_fn, num_trajectories=5)
    if logger is not None:
        logger.log({"trajectory plot": wandb.Image(fig)})
    return fig, ax


def evaluate_policy_mpc(
    env, planner_fn, num_episodes, max_episode_length=300, replan_freq=1, render=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rewards = []
    print(
        "\n--- Starting MPC Evaluation (Replanning frequency: {} steps) ---".format(
            replan_freq
        )
    )
    for eps in range(num_episodes):
        obs, _ = env.reset()
        total_rew = 0

        for t in range(max_episode_length):
            if render:
                env.render()

            if t % replan_freq == 0:
                start_obs_tensor = torch.from_numpy(obs).to(device)
                _, actions_plan = planner_fn(start_obs_tensor)

            action_index_in_plan = t % replan_freq
            action_to_take = actions_plan[action_index_in_plan].cpu().numpy()

            obs, rew, terminated, truncated, info = env.step(action_to_take)
            total_rew += rew

            if terminated or truncated:
                break

        rewards.append(total_rew)
        print(
            f"Episode {eps + 1}/{num_episodes} finished. Total Reward: {total_rew:.2f}"
        )
    return rewards
