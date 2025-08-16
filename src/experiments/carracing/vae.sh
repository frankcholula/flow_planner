#!/bin/bash
python -m src.pipelines.carracing.vae_training \
    --epochs 20 \
    --lr 1e-3 \
    --latent_dim 32 \
    --beta 1.0
