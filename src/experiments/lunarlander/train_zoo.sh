#!/bin/bash
python -m rl_zoo3.train \
    --algo ppo \
    --env LunarLanderContinuous-v3 \
    --n-timesteps 1000000 \
    --track \
    --wandb-project-name FRL \
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
