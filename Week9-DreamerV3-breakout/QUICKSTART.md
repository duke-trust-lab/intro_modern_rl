# DreamerV3 Quick Start Guide

## Installation

```bash
# Method 1: Using UV (Recommended)
uv add git+https://github.com/danijar/dreamerv3

# Method 2: Local Installation
git clone https://github.com/danijar/dreamerv3
cd dreamerv3
uv sync

# For CUDA support (Linux/Windows)
uv sync --extra cuda12
```

## Training Models

```bash
# Train Atari Pong
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer_pong \
  --configs atari \
  --task atari_pong \
  --run.steps 1000000

# Train DMC Walker Walk
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer_walker \
  --configs dmc \
  --task dmc_walker_walk \
  --run.steps 1000000
```

## Inference and Visualization

### Method 1: Record Video (Recommended)

```bash
# Install video dependencies
uv add imageio imageio-ffmpeg

# Run inference and record video
python inference_video.py \
  --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
  --task atari_pong \
  --episodes 5 \
  --output ./videos/

# View video
open ./videos/episode_000_score_21.0.mp4
```

### Method 2: Using Scope Viewer

```bash
# Install Scope
pip install -U scope

# Start viewer
python -m scope.viewer --basedir ~/logdir --port 8000

# Open browser at http://localhost:8000
```

## Project Structure

```
dreamerv3/
├── dreamerv3/              # Main package
│   ├── main.py            # Training entry point
│   ├── agent.py           # DreamerV3 agent
│   ├── rssm.py            # State space model
│   └── configs.yaml       # Configuration file
├── embodied/              # Utility library
│   ├── core/              # Core functionality
│   ├── envs/              # Environment wrappers
│   ├── jax/               # JAX utilities
│   └── run/               # Training scripts
├── inference_video.py     # Inference video recording
├── pyproject.toml         # UV project configuration
└── README.md              # Main documentation
```

## Common Commands

```bash
# Build package
uv build

# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# View package info
uv tree
```

## Additional Documentation

- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - Detailed inference and visualization guide
- [MIGRATION_TO_UV.md](MIGRATION_TO_UV.md) - UV migration analysis
- [README.md](README.md) - Complete project documentation

## Typical Workflow

1. **Train model**
   ```bash
   python dreamerv3/main.py --configs atari --task atari_pong
   ```

2. **Monitor training**
   ```bash
   python -m scope.viewer --basedir ~/logdir --port 8000
   ```

3. **Inference visualization**
   ```bash
   python inference_video.py \
     --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
     --task atari_pong \
     --output ./videos/
   ```

4. **View results**
   ```bash
   open ./videos/episode_*.mp4
   ```

## Troubleshooting

### CUDA Errors
```bash
# Use CPU
python dreamerv3/main.py --jax.platform cpu ...

# Reduce batch size
python dreamerv3/main.py --batch_size 1 ...
```

### Out of Memory
```bash
# Use smaller model
python dreamerv3/main.py --configs atari size12m ...
```

### Checkpoint Not Found
```bash
# Find latest checkpoint
ls -lt ~/logdir/dreamer_*/checkpoint.pkl | head -1
```

## Start Your First Experiment!

```bash
# Complete workflow example
git clone https://github.com/danijar/dreamerv3
cd dreamerv3
uv sync
uv add imageio imageio-ffmpeg

# Train (can use debug config for quick testing)
python dreamerv3/main.py \
  --logdir ~/logdir/test \
  --configs debug atari \
  --task atari_pong \
  --run.steps 10000

# Inference
python inference_video.py \
  --checkpoint ~/logdir/test/checkpoint.pkl \
  --task atari_pong \
  --episodes 3 \
  --output ./test_videos/

# View
open ./test_videos/
```

Good luck!
