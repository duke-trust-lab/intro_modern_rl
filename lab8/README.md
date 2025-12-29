
# Lab 8
# World Models with DreamerV3

### Instructions:
This week explores world model-based reinforcement learning using DreamerV3 on the Atari Breakout environment.

Using the provided code and the Quickstart guide, train Atari Pong OR train DMC Walker Walk. Run inference and record a video. 

### Objectives:
* Explain world-model RL and how DreamerV3 differs from model-free RL by learning in latent space via imagination
* Describe the DreamerV3 loop (world model, actor, critic) at a conceptual level
* Run DreamerV3 end-to-end on Pong or Walker and evaluate behavior via recorded rollouts


### Attribution

This code is based on [DreamerV3](https://github.com/danijar/dreamerv3) by **Danijar Hafner**.

- **Original Repository**: https://github.com/danijar/dreamerv3
- **Paper**: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- **License**: MIT License (see [LICENSE](LICENSE) file in this directory)

The code has been modified to:
- Run specifically on the Breakout environment from the Atari suite
- Include additional training scripts and inference utilities
- Add quick-start guides for educational purposes

### Getting Started

```bash
cd Week9-DreamerV3-breakout
uv sync
```

For detailed instructions, see:
- [QUICKSTART.md](QUICKSTART.md) - Quick setup and training guide
- [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md) - How to run inference and generate videos

### Project Structure

```
Week9-DreamerV3-breakout/
├── dreamerv3/          # Core DreamerV3 implementation
│   ├── agent.py        # Main agent class
│   ├── rssm.py         # Recurrent State Space Model
│   ├── configs.yaml    # Configuration file
│   └── main.py         # Training entry point
├── embodied/           # Supporting library for RL environments
├── train_breakout.sh   # Training script
├── inference_video.py  # Generate demo videos
└── visualize_dream.py  # Visualize world model predictions
```

### References

```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```
