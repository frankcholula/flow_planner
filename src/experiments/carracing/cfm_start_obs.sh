#!/bin/bash
# ccnn
# python -m src.run \
#     --environment CarRacing-v3 \
#     --horizon 30 \
#     --batch-size 64 \
#     --num-epochs 100 \
#     --model-type ccnn \
#     --condition-on start_obs \
#     --model-target obs_only \
#     --lr 1e-3

# unet
python -m src.run \
    --environment CarRacing-v3 \
    --horizon 25 \
    --batch-size 32 \
    --num-epochs 100 \
    --print-every 10 \
    --model-type unet \
    --condition-on start_obs \
    --model-target obs_only \
    --lr 1e-3