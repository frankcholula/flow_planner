#!/bin/bash
# model both state and actions and condition on start_obs_goal
# try with ccnn fist
python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 10 \
    --print-every 1 \
    --eval-every 5 \
    --lr 1e-3 \
    --model-type "ccnn" \
    --kernel-size 5 git\
    --hidden-dim 64 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --num-inference-steps 1000 \
    --condition-on "start_obs_goal" \
    --model-target "obs_act"

# try with unet
# python -m src.diffusion \
#     --environment "LunarLander-v3" \
#     --horizon 100 \
#     --batch-size 32 \
#     --num-epochs 10 \
#     --print-every 1 \
#     --eval-every 5 \
#     --lr 1e-3 \
#     --model-type "unet" \
#     --hidden-dim 64 \
#     --scheduler "ddpm" \
#     --num-train-timesteps 1000 \
#     --num-inference-steps 1000 \
#     --condition-on "start_obs_goal" \
#     --model-target "obs_act"