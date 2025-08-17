#!/bin/bash
python -m src.pipelines.carracing.vae_training \
    --epochs 50 \
    --lr 1e-3 \
    --latent_dim 32 \
    --beta 1.0 \
    --dataset_name Box2D/CarRacing-v3/mixed-v0