# DreamerV3 Inference Guide

Run a trained model and generate videos of the agent's behavior.

##  Quick Start

1.  **Install Dependencies**
    ```bash
    uv add imageio imageio-ffmpeg
    ```

2.  **Run Inference**
    ```bash
    python inference_video.py \
      --checkpoint ~/logdir/dreamer_pong/checkpoint.pkl \
      --task atari_pong \
      --episodes 5 \
      --output ./videos/
    ```

3.  **View Results**
    Videos are saved as `.mp4` files in your output directory:
    ```bash
    open ./videos/episode_000_*.mp4
    ```

##  Common Issues

- **Black Screen?** Ensure `render=True` is enabled in your env config.
- **No Checkpoint?** Checkpoints are auto-saved to `~/logdir/dreamer_{task}/{timestamp}/`.
