#!/bin/bash
python -m src.run \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 100 \
    --print-every 1 \
    --eval-every 5 \
    --lr 1e-3 \
    --model-type "unet" \
    --hidden-dim 64 \
    --step-size 0.05 \
    --solver-method "midpoint" \
    --inference-batch-size 1 \
    --condition-on "start_obs_waypoint" \
    --model-target "obs_act"
