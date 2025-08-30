#!/bin/bash
python -m src.diffusion \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 50 \
    --print-every 1 \
    --eval-every 5 \
    --lr 1e-3 \
    --model-type "mlp" \
    --hidden-dim 128 \
    --scheduler "ddpm" \
    --num-train-timesteps 1000  \
    --num-inference-steps 100 \
    --inference-batch-size 1
