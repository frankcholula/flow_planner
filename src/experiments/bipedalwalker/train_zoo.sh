#!/bin/bash
# hyperparameters taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/hyperparams/ppo.yml#L86-L98
python -m rl_zoo3.train \
    --algo ppo \
    --env BipedalWalker-v3 \
    --n-timesteps 5000000 \
    --track \
    --wandb-project-name "Flow Planner" \
    --wandb-entity frankcholula \
    --device cpu \
    --hyperparams \
    normalize:True \
    n_envs:32 \
    n_steps:2048 \
    batch_size:64 \
    gae_lambda:0.95 \
    gamma:0.999 \
    n_epochs:10 \
    ent_coef:0.0 \
    learning_rate:3e-4 \
    clip_range:0.18
