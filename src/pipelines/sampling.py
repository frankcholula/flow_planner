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


def generate_trajectory(stats, solver, condition: dict, batch_size: int = 1):
    """
    Generates a trajectory based on a provided condition dictionary.

    Args:
        stats (dict): The dataset statistics for normalization.
        condition (dict): A dictionary specifying the condition,
                          e.g., {'reward': 250.0} or {'start_obs': tensor}.
        batch_size (int): The number of trajectories to generate.
    """
    # Prepare the condition tensor `c` based on the dictionary keys
    if "reward" in condition:
        rew_mean = stats["rew_mean"].to(device)
        rew_std = stats["rew_std"].to(device)
        norm_c = (
            torch.tensor([condition["reward"]], device=device) - rew_mean
        ) / rew_std
        c_tensor = norm_c.view(1, 1).expand(batch_size, -1)  # Shape: (batch_size, 1)

    elif "start_obs" in condition:
        obs_mean = stats["obs_mean"].to(device)
        obs_std = stats["obs_std"].to(device)
        # Ensure the input start_obs is a tensor on the correct device
        start_obs_tensor = condition["start_obs"].to(device)
        norm_c = (start_obs_tensor - obs_mean) / obs_std
        # Add a batch dimension and expand if needed
        c_tensor = norm_c.unsqueeze(0).expand(batch_size, -1)  # Shape: (batch_size, 8)

    else:
        raise ValueError("Condition dictionary must contain 'reward' or 'start_obs'")

    # Generate the initial noise vector
    x_init = torch.randn((batch_size, input_dim), dtype=torch.float32, device=device)

    # Sample from the solver, passing the prepared condition `c`
    sol = solver.sample(
        time_grid=T,
        x_init=x_init,
        c=c_tensor,
        method="midpoint",
        step_size=0.05,
        return_intermediates=False,  # Only need the final result for inference
    )
    # sol has shape (batch_size, input_dim)

    # Un-normalize the first trajectory in the batch
    obs, act = unnormalize_trajectory(
        sol[0].flatten().detach(), stats, horizon, obs_dim, action_dim
    )
    return obs, act
