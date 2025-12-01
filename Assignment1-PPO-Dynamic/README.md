# Assignment 1: PPO on MiniGrid Dynamic Obstacles

## Current Status

The code is configured for `MiniGrid-Dynamic-Obstacles-5x5-v0` and trains successfully.

## The Challenge

**When you scale up to 8x8 or 16x16 environments, the agent gets stuck spinning in circles.**

### Why This Happens

1. **Sparse Rewards**: The agent only gets +1 reward when reaching the goal. In larger grids (8x8, 16x16), it takes 20-50 steps to reach the goal, making the reward signal too weak.

2. **Easy Penalties**: Every step has a small time penalty. The agent learns that "doing nothing is safer than exploring" because exploration leads to hitting walls or obstacles.

3. **The Problem**: Spinning in place has higher expected reward than trying to reach the goal.
   - Spinning: Small consistent penalty ≈ -1.0
   - Exploring: High risk of failure ≈ -2.0

## Your Task

Scale the environment to 8x8 or 16x16 and make the agent learn successfully.

**Hint**: The reward structure is the core issue. How can you provide better guidance to the agent?

## How to Run

```bash
python main.py
```

## Configuration

Change the environment in `main.py`:
```python
ENV_NAME = "MiniGrid-Dynamic-Obstacles-5x5-v0"   # Current (works)
ENV_NAME = "MiniGrid-Dynamic-Obstacles-8x8-v0"   # Challenge
ENV_NAME = "MiniGrid-Dynamic-Obstacles-16x16-v0" # Harder challenge
```
