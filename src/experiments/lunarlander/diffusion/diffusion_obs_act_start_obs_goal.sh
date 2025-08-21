#!/bin/bash
# model both state and actions and condition on start_obs_goal
python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 100 \
    --print-every 5 \
    --eval-every 10 \
    --lr 1e-3 \
    --model-type "unet" \
    --hidden-dim 64 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --num-inference-steps 1000 \
    --condition-on "start_obs_goal" \
    --model-target "obs_act"
