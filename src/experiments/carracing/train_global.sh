python -m src.run \
    --environment CarRacing-v3 \
    --horizon 30 \
    --batch-size 64 \
    --num-epochs 100 \
    --model-type ccnn \
    --condition-on start_obs \
    --chunk-type obs_only \
    --lr 1e-3
