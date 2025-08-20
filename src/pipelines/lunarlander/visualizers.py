import matplotlib.pyplot as plt
import numpy as np


def visualize_trajectories(trajectory_fn, num_trajectories=5):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Trajectory Visualization")
    for _ in range(num_trajectories):
        colors = ["red", "orange", "yellow", "green", "blue"]
        obs, act = trajectory_fn()
        visualize_chunk(ax, obs, color=colors[_ % len(colors)], mode="line")
    return fig, ax


def visualize_chunk(
    ax, chunk, color, x_limits=(-0.6, 0.6), y_limits=(-0.2, 1.6), mode="line"
):
    ax.fill_between(
        [-0.2, 0.2], -0.02, 0, color="gold", alpha=0.8, zorder=1, label="Landing Pad"
    )
    x = chunk[:, 0].cpu().numpy()
    y = chunk[:, 1].cpu().numpy()
    if mode == "line":
        ax.plot(x, y, linestyle="-", color=color, alpha=0.7)
    elif mode == "scatter":
        ax.scatter(x, y, color=color, alpha=0.7)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(True)


def visualize_dataset(dataset):
    episode_lengths = []
    aggregated_rewards = []

    for eps in dataset.episode_indices:
        episode = dataset[eps]
        episode_lengths.append(episode.observations.shape[0])
        aggregated_rewards.append(np.sum(episode.rewards))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(episode_lengths, bins=50, edgecolor="black")
    plt.title("Distribution of Episode Lengths")
    plt.xlabel("Episode Length")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(aggregated_rewards, bins=50, edgecolor="black")
    plt.title("Distribution of Aggregated Rewards per Episode")
    plt.xlabel("Aggregated Reward")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Total number of episodes: {len(episode_lengths)}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    print(f"Min episode length: {np.min(episode_lengths)}")
    print(f"Max episode length: {np.max(episode_lengths)}")
    print(f"Average aggregated reward: {np.mean(aggregated_rewards):.2f}")
    print(f"Min aggregated reward: {np.min(aggregated_rewards):.2f}")
    print(f"Max aggregated reward: {np.max(aggregated_rewards):.2f}")


def plot_reward_histogram(
    rewards_list, bins=20, title="Reward Distribution per Episode"
):
    if not rewards_list:
        print("Warning: Cannot plot an empty list of rewards.")
        return

    plt.figure(figsize=(10, 6))

    # Calculate mean and standard deviation for the legend
    mean_reward = np.mean(rewards_list)
    std_reward = np.std(rewards_list)
    legend_label = f"mean={mean_reward:.2f}, std={std_reward:.2f}"

    # Plot the histogram
    plt.hist(rewards_list, bins=bins, alpha=0.75, edgecolor="black", label=legend_label)

    plt.title(title, fontsize=16)
    plt.xlabel("Total Reward per Episode", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
