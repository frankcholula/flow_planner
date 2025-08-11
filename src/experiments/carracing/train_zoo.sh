#!/bin/bash
# hyperparameters taken from https://github.com/DLR-RM/rl-baselines3-zoo/blob/ab4aadb57c6c42abcf1016318c2bebe35c4c1270/hyperparams/ppo.yml#L350-L379

# python -m rl_zoo3.train \
#     --algo ppo \
#     --env CarRacing-v3 \
#     --track \
#     --wandb-project-name "Flow Planner" \
#     --wandb-entity frankcholula \
#     --device auto

python -m rl_zoo3.train \
    --algo ppo_lstm \
    --env CarRacing-v3 \
    --track \
    --wandb-project-name "Flow Planner" \
    --wandb-entity frankcholula \
    --device auto
