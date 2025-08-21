import torch


def unnormalize_trajectory(
    chunk, stats, horizon, obs_dim, action_dim, model_target="obs_act"
):
    obs_mean, obs_std = stats["obs_mean"].to(chunk.device), stats["obs_std"].to(
        chunk.device
    )
    act_mean, act_std = stats["act_mean"].to(chunk.device), stats["act_std"].to(
        chunk.device
    )

    obs, act = None, None

    if model_target == "obs_act":
        reshaped = chunk.reshape(horizon, obs_dim + action_dim)
        norm_obs = reshaped[:, :obs_dim]
        norm_act = reshaped[:, obs_dim:]
        obs = norm_obs * obs_std + obs_mean
        act = norm_act * act_std + act_mean
    elif model_target == "obs_only":
        reshaped = chunk.reshape(horizon, obs_dim)
        obs = reshaped * obs_std + obs_mean
        act = None
    elif model_target == "act_only":
        reshaped = chunk.reshape(horizon, action_dim)
        act = reshaped * act_std + act_mean
        obs = None

    return obs, act


@torch.no_grad()
def generate_trajectory(
    stats,
    solver,
    T,
    input_dim: int,
    horizon: int = 100,
    condition: dict = None,
    solver_method: str = "midpoint",
    batch_size: int = 1,
    step_size: float = 0.05,
    return_intermediates: bool = False,
    model_target: str = "obs_act",
):
    # infer obs and action dim from stats
    obs_dim = stats["obs_mean"].shape[0]
    action_dim = stats["act_mean"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "reward" in condition:
        rew_mean = stats["rew_mean"].to(device)
        rew_std = stats["rew_std"].to(device)
        norm_c = (
            torch.tensor([condition["reward"]], device=device) - rew_mean
        ) / rew_std
        c_tensor = norm_c.view(1, 1).expand(batch_size, -1)

    elif "start_obs" in condition:
        obs_mean = stats["obs_mean"].to(device)
        obs_std = stats["obs_std"].to(device)
        start_obs_tensor = condition["start_obs"].float().to(device)
        norm_c = (start_obs_tensor - obs_mean) / obs_std
        c_tensor = norm_c.unsqueeze(0).expand(batch_size, -1)
    elif "start_obs_goal" in condition:
        obs_mean = stats["obs_mean"].to(device)
        obs_std = stats["obs_std"].to(device)
        start_obs, goal_obs = condition["start_obs_goal"]
        start_obs_tensor = start_obs.float().to(device)
        goal_obs_tensor = goal_obs.float().to(device)
        norm_start = (start_obs_tensor - obs_mean) / obs_std
        norm_goal = (goal_obs_tensor - obs_mean) / obs_std
        norm_c = torch.cat([norm_start, norm_goal])
        c_tensor = norm_c.unsqueeze(0).expand(batch_size, -1)
    else:
        c_tensor = None
        raise ValueError("Condition type not recognized.")

    x_init = torch.randn((batch_size, input_dim), dtype=torch.float32, device=device)

    solver_kwargs = {
        "time_grid": T.to(device),
        "x_init": x_init,
        "c": c_tensor,
        "method": solver_method,
        "step_size": step_size,
        "return_intermediates": return_intermediates,
    }
    sol = solver.sample(**solver_kwargs)
    obs, act = unnormalize_trajectory(
        sol[0].flatten().detach(),
        stats,
        horizon,
        obs_dim,
        action_dim,
        model_target=model_target,
    )
    return obs, act


@torch.no_grad()
def generate_diffusion_trajectory(
    model, noise_scheduler, stats, args, input_dim, condition: dict = None
):
    c_tensor = None
    device = args.device
    batch_size = args.inference_batch_size
    # Check if a condition was provided at all
    if condition is not None:
        if "reward" in condition:
            rew_mean = stats["rew_mean"].to(device)
            rew_std = stats["rew_std"].to(device)
            norm_c = (
                torch.tensor([condition["reward"]], device=device) - rew_mean
            ) / rew_std
            c_tensor = norm_c.view(1, 1).expand(batch_size, -1)

        elif "start_obs" in condition:
            obs_mean = stats["obs_mean"].to(device)
            obs_std = stats["obs_std"].to(device)
            start_obs_tensor = condition["start_obs"].float().to(device)
            norm_c = (start_obs_tensor - obs_mean) / obs_std
            c_tensor = norm_c.unsqueeze(0).expand(batch_size, -1)

        elif "start_obs_goal" in condition:
            obs_mean = stats["obs_mean"].to(device)
            obs_std = stats["obs_std"].to(device)

            start_obs, goal_obs = condition["start_obs_goal"]
            start_obs_tensor = start_obs.float().to(device)
            goal_obs_tensor = goal_obs.float().to(device)
            norm_start = (start_obs_tensor - obs_mean) / obs_std
            norm_goal = (goal_obs_tensor - obs_mean) / obs_std
            norm_c = torch.cat([norm_start, norm_goal])
            c_tensor = norm_c.unsqueeze(0).expand(batch_size, -1)

        else:
            raise ValueError(
                f"Condition type in dict not recognized: {condition.keys()}"
            )

    # Reverse diffusion
    sample = torch.randn((batch_size, input_dim), device=device)
    noise_scheduler.set_timesteps(args.num_inference_steps)

    for t in noise_scheduler.timesteps:
        timestep_tensor = t.repeat(batch_size).to(device)

        # The model receives the final, processed c_tensor
        predicted_noise = model(sample, timestep_tensor, c=c_tensor)
        sample = noise_scheduler.step(predicted_noise, t, sample).prev_sample

    obs_dim = stats["obs_mean"].shape[0]
    action_dim = stats["act_mean"].shape[0]

    obs, act = unnormalize_trajectory(
        chunk=sample,
        stats=stats,
        horizon=args.horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        model_target=args.model_target,
    )
    return obs, act
