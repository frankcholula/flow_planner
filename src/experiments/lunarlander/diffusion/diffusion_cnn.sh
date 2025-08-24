#!/bin/bash
python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 2 \
    --print-every 1 \
    --eval-every 1 \
    --lr 1e-3 \
    --model-type "cnn" \
    --kernel-size 5 \
    --hidden-dim 128 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000 \
    --num-inference-steps 1000 \
    --inference-batch-size 1
