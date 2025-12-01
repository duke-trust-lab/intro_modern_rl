#!/bin/bash
# Record a video of the trained agent
# Video will be automatically saved to logs/<algo>/<env>_<exp_id>/videos/ directory
# Example: logs/dqn/CartPole-v1_1/videos/final-model-dqn-CartPole-v1-step-0-to-step-1000.mp4


python -m rl_zoo3.record_video \
    --algo dqn \
    --env CartPole-v1 \
    --folder logs/ \
    -n 500 \
    --deterministic \
    --load-best

echo "Video saved to logs/dqn/CartPole-v1_*/videos/ directory"