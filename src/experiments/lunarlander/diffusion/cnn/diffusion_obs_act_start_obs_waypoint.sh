#!/bin/bash

python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 100 \
    --print-every 1 \
    --eval-every 5 \
    --lr 1e-3 \
    --model-type "ccnn" \
    --kernel-size 5 \
    --hidden-dim 128 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --num-inference-steps 100 \
    --inference-batch-size 1 \
    --condition-on "start_obs_waypoint" \
    --model-target "obs_act" \
    --fusion-strategy "concat"
