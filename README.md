# AIPI 531 - Introduction to Modern Reinforcement Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)

Code repository for the **AIPI 531: Introduction to Modern Reinforcement Learning** course. This project contains assignments, weekly exercises, and implementations of various RL algorithms.

## Project Structure

This repository uses a **monorepo workspace** architecture. Each subfolder is an **independent project** with its own dependencies and virtual environment:

```
intro_modern_rl/
├── Week2-QLearning-FrozenLake/    # Independent project
├── Week3-DQN-CartPole/            # Independent project
├── Week4-PPO-minigrid/            # Independent project
├── ...
├── Assignment1-PPO-Dynamic/       # Independent project
├── Assignment2-HuggingFace-RLHF/  # Independent project
└── ...
```

- **Weeks**: Weekly coding exercises covering RL fundamentals
- **Assignments**: Major course projects with more complex implementations

> **Note**: The root directory contains no runnable code. You must navigate into a specific subfolder to run any code.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (Recommended for dependency management)

### Quick Start (Recommended)

Each subfolder is a standalone project. To run any project:

```bash
# 1. Clone the repository
git clone https://github.com/duke-trust-lab/intro_modern_rl.git
cd intro_modern_rl

# 2. Navigate to the project you want to run
cd Assignment1-PPO-Dynamic

# 3. Install dependencies (creates .venv automatically)
uv sync

# 4. Run the code
uv run python main.py
```

### Alternative: Using pip

If you prefer not to use `uv`:

```bash
cd Assignment1-PPO-Dynamic
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python main.py
```

## Usage

Each subfolder is a standalone project. Depending on the project, you may find:

| File | Purpose | How to Run |
|------|---------|------------|
| `main.py` | Command-line script for local execution | `uv run python main.py` |
| `main.ipynb` | Jupyter notebook for Colab or local notebook server | Open in Colab or Jupyter |
| `pyproject.toml` | Project dependencies | Used by `uv sync` or `pip install -e .` |

> **Note**: Not all projects have both `main.py` and `main.ipynb`. Check each subfolder for available files.

### Running Locally (Command Line)

```bash
cd Assignment1-PPO-Dynamic
uv sync
uv run python main.py
```

### Running on Google Colab / Jupyter

Projects with `main.ipynb` can be opened directly in Google Colab or any Jupyter environment. The first cell installs all dependencies automatically:

```python
# First cell in notebook - installs dependencies from pyproject.toml
%pip install -q -e .
```

To run locally with Jupyter:

```bash
cd Week2-QLearning-FrozenLake
uv sync
uv run jupyter lab
# Then open main.ipynb
```

### Troubleshooting

If `uv sync` fails due to platform compatibility issues:
```bash
uv lock --upgrade
uv sync
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Code

This repository includes third-party code with separate licensing:

- **Week9-DreamerV3-breakout**: Contains code from [DreamerV3](https://github.com/danijar/dreamerv3) by Danijar Hafner, licensed under the MIT License. See [Week9-DreamerV3-breakout/LICENSE](Week9-DreamerV3-breakout/LICENSE) for details.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of contributors.