#!/bin/bash

# DreamerV3 Breakout Training Script
# 1 million steps, checkpoint saved every 5 minutes

uv run python -m dreamerv3.main \
  --configs atari size1m \
  --task atari_breakout \
  --run.steps 1000000 \
  --run.save_every 300 \
  --jax.platform cpu \
  --logdir ./dreamer_breakout_logdir
