from matplotlib import pyplot as plt
import numpy as np


def visualize_trajectory(observations: np.ndarray, horizon: int, verbose=False, y_limits= (1, 1.5)):
    plot_horizon = min(horizon, len(observations))
    time_steps = np.arange(plot_horizon)
    torso_height = observations[:plot_horizon, 0]
    torso_angle = observations[:plot_horizon, 1]
    thigh_joint_angle = observations[:plot_horizon, 2]
    leg_joint_angle = observations[:plot_horizon, 3]

    num_plots = 4 if verbose else 1
    figsize = (12, 10) if verbose else (12, 4)

    fig, axs = plt.subplots(num_plots, 1, figsize=figsize, sharex=True, squeeze=False)
    axs = axs.flatten()

    fig.suptitle("Trajectory Visualization", fontsize=16)
    
    axs[0].plot(time_steps, torso_height, color="dodgerblue")
    axs[0].set_ylabel("Torso Height (m)")
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].set_ylim(y_limits)

    if verbose:
        axs[1].plot(time_steps, torso_angle, color="coral")
        axs[1].set_ylabel("Torso Angle (rad)")
        axs[1].grid(True, linestyle="--", alpha=0.6)

        axs[2].plot(time_steps, thigh_joint_angle, color="limegreen")
        axs[2].set_ylabel("Thigh Joint (rad)")
        axs[2].grid(True, linestyle="--", alpha=0.6)

        axs[3].plot(time_steps, leg_joint_angle, color="orchid")
        axs[3].set_ylabel("Leg Joint (rad)")
        axs[3].set_xlabel("Time Step")
        axs[3].grid(True, linestyle="--", alpha=0.6)
    else:
        axs[0].set_xlabel("Time Step")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
