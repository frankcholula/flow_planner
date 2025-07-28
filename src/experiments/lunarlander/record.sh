#!/bin/bash
python -m rl_zoo3.record_video \
    --env LunarLanderContinuous-v3 \
    --n-timesteps 1000 \
    --n-envs 16 \
    --folder logs \
    --algo ppo \
    --load-best
