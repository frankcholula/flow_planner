#!/bin/bash
# hyperparameters taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/hyperparams/ppo.yml#L125-L134

python -m rl_zoo3.train \
    --algo ppo \
    --env LunarLanderContinuous-v3 \
    --track \
    --wandb-project-name "Flow Planner" \
    --wandb-entity frankcholula \
    --device auto

