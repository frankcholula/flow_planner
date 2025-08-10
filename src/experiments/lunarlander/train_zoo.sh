#!/bin/bash
# hyperparameters taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/hyperparams/ppo.yml#L125-L134
python -m rl_zoo3.train \
    --algo ppo \
    --env LunarLanderContinuous-v3 \
    --n-timesteps 1000000 \
    --track \
    --wandb-project-name "Flow Planner" \
    --wandb-entity frankcholula \
    --device cpu \
    --hyperparams \
    n_envs:16 \
    n_steps:1024 \
    batch_size:64 \
    n_epochs:4 \
    gae_lambda:0.98 \
    gamma:0.999 \
    ent_coef:0.01
