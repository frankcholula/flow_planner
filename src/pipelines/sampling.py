import torch


def unnormalize_trajectory(
    chunk, stats, horizon, obs_dim, action_dim, model_target="obs_act",
):
    """Unnormalize a (potentially batched) trajectory."""

    obs_mean, obs_std = stats["obs_mean"].to(chunk.device), stats["obs_std"].to(
        chunk.device
    )
    act_mean, act_std = stats["act_mean"].to(chunk.device), stats["act_std"].to(
        chunk.device
    )

    # bring chunk to shape (batch, horizon, dim)
    if chunk.dim() == 1:
        batch = 1
        reshaped = chunk.view(1, horizon, -1)
    elif chunk.dim() == 2:
        batch = chunk.shape[0]
        reshaped = chunk.view(batch, horizon, -1)
    elif chunk.dim() == 3:
        batch = chunk.shape[0]
        reshaped = chunk
    else:
        raise ValueError("Unsupported chunk shape")

    obs, act = None, None
    if model_target == "obs_act":
        norm_obs = reshaped[:, :, :obs_dim]
        norm_act = reshaped[:, :, obs_dim:]
        obs = norm_obs * obs_std + obs_mean
        act = norm_act * act_std + act_mean
    elif model_target == "obs_only":
        reshaped = reshaped.view(batch, horizon, obs_dim)
        obs = reshaped * obs_std + obs_mean
        act = None
    elif model_target == "act_only":
        reshaped = reshaped.view(batch, horizon, action_dim)
        act = reshaped * act_std + act_mean
        obs = None

    if batch == 1:
        obs = obs.squeeze(0) if obs is not None else None
        act = act.squeeze(0) if act is not None else None

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
    c_tensor = None
    obs_dim = stats["obs_mean"].shape[0]
    action_dim = stats["act_mean"].shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if condition:
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
            raise ValueError(f"Condition type not recognized: {condition.keys()}")

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
    norm_start, norm_goal = None, None
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
            norm_start = norm_c  # For fixing the start point only

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

    obs_dim = stats["obs_mean"].shape[0]
    action_dim = stats["act_mean"].shape[0]

    # Reverse diffusion
    sample = torch.randn(
        (batch_size, args.horizon, obs_dim + action_dim), device=device
    )

    # Pre-fill start/goal observations if provided
    mask = torch.ones_like(sample)
    fixed = torch.zeros_like(sample)
    if norm_start is not None:
        start_expand = norm_start.view(1, -1).expand(batch_size, -1)
        sample[:, 0, :obs_dim] = start_expand
        mask[:, 0, :obs_dim] = 0
        fixed[:, 0, :obs_dim] = start_expand
    if norm_goal is not None:
        goal_expand = norm_goal.view(1, -1).expand(batch_size, -1)
        sample[:, -1, :obs_dim] = goal_expand
        mask[:, -1, :obs_dim] = 0
        fixed[:, -1, :obs_dim] = goal_expand

    noise_scheduler.set_timesteps(args.num_inference_steps)
    sample_flat = sample.reshape(batch_size, -1)

    for t in noise_scheduler.timesteps:
        if args.cfg and c_tensor is not None:
            model_batch_size = 2 * batch_size
        else:
            model_batch_size = batch_size
        timestep_tensor = t.repeat(model_batch_size).to(device)
        if args.cfg and c_tensor is not None:
            # clever way to bypass 2 stage
            latent_model_input = torch.cat([sample_flat] * 2)
            null_condition = torch.zeros_like(c_tensor)
            c_double = torch.cat([null_condition, c_tensor])

            noise_pred_double = model(latent_model_input, timestep_tensor, c=c_double)
            noise_pred_uncond, noise_pred_cond = noise_pred_double.chunk(2)
            predicted_noise = noise_pred_uncond + args.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
        else:
            if c_tensor is not None:
                predicted_noise = model(sample_flat, timestep_tensor, c=c_tensor)
            else:
                predicted_noise = model(sample_flat, timestep_tensor)
        sample_flat = noise_scheduler.step(predicted_noise, t, sample_flat).prev_sample

        # Reshape and re-apply the fixed endpoints
        sample = sample_flat.view(batch_size, args.horizon, obs_dim + action_dim)
        sample = sample * mask + fixed * (1 - mask)
        sample_flat = sample.view(batch_size, -1)

    obs, act = unnormalize_trajectory(
        chunk=sample,
        stats=stats,
        horizon=args.horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        model_target=args.model_target,
    )

    # some debugging for the first sample
    if obs is not None:
        start = obs[0, 0, :2] if obs.dim() == 3 else obs[0, :2]
        end = obs[0, -1, :2] if obs.dim() == 3 else obs[-1, :2]
        print(f"Start (x,y): ({start[0].item():.4f}, {start[1].item():.4f})")
        print(f"End   (x,y): ({end[0].item():.4f}, {end[1].item():.4f})")
        print("------------------------------")

    return obs, act
