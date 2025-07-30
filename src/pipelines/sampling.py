import torch


def unnormalize_trajectory(chunk, stats, horizon, obs_dim, action_dim):
    obs_mean, obs_std = stats["obs_mean"].to(chunk.device), stats["obs_std"].to(
        chunk.device
    )
    act_mean, act_std = stats["act_mean"].to(chunk.device), stats["act_std"].to(
        chunk.device
    )
    reshaped = chunk.reshape(horizon, obs_dim + action_dim)
    norm_obs = reshaped[:, :obs_dim]
    norm_act = reshaped[:, obs_dim:]
    obs = norm_obs * obs_std + obs_mean
    act = norm_act * act_std + act_mean
    return obs, act


def generate_trajectory(
    stats, solver, T, input_dim, args, horizon, condition: dict, batch_size: int = 1
):
    # infer obs and action dim from stats
    obs_dim = stats["obs_mean"].shape[0]
    action_dim = stats["act_mean"].shape[0]

    if "reward" in condition:
        rew_mean = stats["rew_mean"].to(args.device)
        rew_std = stats["rew_std"].to(args.device)
        norm_c = (
            torch.tensor([condition["reward"]], device=args.device) - rew_mean
        ) / rew_std
        c_tensor = norm_c.view(1, 1).expand(batch_size, -1)

    elif "start_obs" in condition:
        obs_mean = stats["obs_mean"].to(args.device)
        obs_std = stats["obs_std"].to(args.device)
        start_obs_tensor = condition["start_obs"].float().to(args.device)
        norm_c = (start_obs_tensor - obs_mean) / obs_std
        c_tensor = norm_c.unsqueeze(0).expand(batch_size, -1)
    else:
        raise ValueError("Condition dictionary must contain 'reward' or 'start_obs'")

    x_init = torch.randn(
        (batch_size, input_dim), dtype=torch.float32, device=args.device
    )

    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        c=c_tensor,
        method="midpoint",
        step_size=0.05,
        return_intermediates=False,
    )
    obs, act = unnormalize_trajectory(
        sol[0].flatten().detach(), stats, horizon, obs_dim, action_dim
    )
    return obs, act
