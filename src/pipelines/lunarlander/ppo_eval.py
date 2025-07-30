import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from conf.environment import LunarLanderConfig
import pygame

model_name = f"ppo-{LunarLanderConfig.env_name}"
eval_env = Monitor(gym.make("LunarLander-v3", continuous=True, render_mode="human"))

# model = PPO.load(f"models/{model_name}/model", env=eval_env)
model = PPO.load("../logs/ppo/LunarLanderContinuous-v3_1/best_model.zip", env=eval_env)
mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward} +/- {std_reward}")
episode_lengths = eval_env.get_episode_lengths()
print(f"Lengths of evaluated episodes: {episode_lengths}")
print(f"Number of evaluated episodes: {len(episode_lengths)}")
if len(episode_lengths) > 0:
    print(f"Average episode length: {sum(episode_lengths) / len(episode_lengths):.2f}")

pygame.display.quit()
pygame.quit()
