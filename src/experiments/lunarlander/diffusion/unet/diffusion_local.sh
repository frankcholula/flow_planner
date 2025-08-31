#!/bin/bash
# set eval-every to bypass eval because there is no obs.
python -m src.run \
    --environment "LunarLander-v3" \
    --horizon 25 \
    --batch-size 32 \
    --num-epochs 50 \
    --print-every 1 \
    --eval-every 500 \
    --lr 1e-3 \
    --model-type "unet" \
    --hidden-dim 64 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --num-inference-steps 100 \
    --inference-batch-size 1 \
    --condition-on "start_obs_goal" \
    --model-target "act_only"
