#!/bin/bash

python -m src.run \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 50 \
    --print-every 1 \
    --eval-every 5 \
    --lr 1e-3 \
    --model-type "mlp" \
    --hidden-dim 128 \
    --step-size 0.05 \
    --solver-method "midpoint" \
    --inference-batch-size 1
