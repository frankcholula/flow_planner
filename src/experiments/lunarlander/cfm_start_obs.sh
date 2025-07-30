#!/bin/bash
python -m src.run \
    --environment "LunarLander-v3" \
    --horizon 100 \
    --batch-size 32 \
    --num-epochs 100 \
    --print-every 10 \
    --lr 1e-3 \
    --model-type "ccnn" \
    --kernel-size 10 \
    --hidden-dim 128 \
    --step-size 0.05 \
    --solver-method "midpoint" \
    --inference-batch-size 1 \
    --condition-on "start_obs"


# uncomment for using MLP
# python -m src.run \
#     --environment "LunarLander-v3" \
#     --horizon 100 \
#     --batch-size 32 \
#     --num-epochs 1000 \
#     --print-every 10 \
#     --lr 1e-3 \
#     --model-type "mlp" \
#     --kernel-size 5 \
#     --hidden-dim 128 \
#     --step-size 0.05 \
#     --solver-method "midpoint" \
#     --inference-batch-size 1 \
#     --condition-on "start_obs"