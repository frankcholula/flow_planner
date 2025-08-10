#!/bin/bash
python -m rl_zoo3.train \
    --algo ppo \
    --env BipedalWalker-v3 \
    --n-timesteps 5000000 \
    --track \
    --wandb-project-name FRL \
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
