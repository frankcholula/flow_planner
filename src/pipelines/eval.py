import torch

def evaluate_policy_mpc(
    env, planner_fn,
    num_episodes, max_episode_length=300, replan_freq=1, render=False
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
        action_plan = None

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