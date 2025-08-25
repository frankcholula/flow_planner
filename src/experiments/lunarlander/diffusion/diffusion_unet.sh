#!/bin/bash
python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 2 \
    --print-every 1 \
    --eval-every 1 \
    --lr 1e-4 \
    --model-type "unet" \
    --hidden-dim 64 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --num-inference-steps 100 \
    --inference-batch-size 1
