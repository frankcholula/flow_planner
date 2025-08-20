#!/bin/bash
python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 10 \
    --print-every 1 \
    --lr 1e-3 \
    --model-type "unet" \
    --hidden-dim 64 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --condition-on "start_obs" \
    --model-target "act_only"
