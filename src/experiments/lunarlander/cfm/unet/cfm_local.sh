#!/bin/bash
# set eval-every to bypass eval because there is no obs.
python -m src.run \
    --environment "LunarLander-v3" \
    --horizon 25 \
    --batch-size 32 \
    --num-epochs 100 \
    --print-every 1 \
    --eval-every 500 \
    --lr 1e-3 \
    --model-type "unet" \
    --hidden-dim 64 \
    --step-size 0.05 \
    --solver-method "midpoint" \
    --inference-batch-size 1 \
    --condition-on "start_obs_goal" \
    --model-target "act_only"
