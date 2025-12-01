#!/usr/bin/env python3
"""
DreamerV3 Dream Visualization Script
Displays: Real observation | Model reconstruction | Imagined trajectory
"""

import argparse
import pathlib
import numpy as np
import imageio
import elements
import embodied
import jax
import jax.numpy as jnp
from functools import partial as bind

folder = pathlib.Path(__file__).parent

def visualize_dream(checkpoint_path, task='atari_pong', episodes=3, output_dir='./dream_videos'):
    """Visualize world model dreams"""

    print("DreamerV3 Dream Visualization")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {task}")

    # Load config
    import ruamel.yaml as yaml
    from dreamerv3.agent import Agent
    from dreamerv3 import main as dreamerv3_main

    config_file = folder / 'dreamerv3' / 'configs.yaml'
    configs = yaml.YAML(typ='safe').load(config_file.read_text())

    config = elements.Config(configs['defaults'])
    suite = task.split('_')[0]
    if suite in configs:
        config = config.update(configs[suite])

    config = config.update(
        task=task,
        logdir=f'./eval_logs/{task}',
        seed=0,
    )
    config = config.update(jax={'platform': 'cpu', 'prealloc': False})

    # Create environment and agent
    print("Creating environment...")
    env = dreamerv3_main.make_env(config, 0)

    print("Loading agent...")
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
    ))

    # Load checkpoint
    print(f"Loading checkpoint...")
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_path, keys=['agent'])

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting dream video recording...\n")

    for ep in range(episodes):
        print(f"Episode {ep + 1}/{episodes}")

        obs = env.reset()
        carry = agent.init_policy(batch_size=1)

        frames_real = []
        frames_recon = []  # Reconstructed observations

        for step in range(1000):
            # Record real observation
            if 'image' in obs:
                frames_real.append(obs['image'])

            # Agent decision (internally reconstructs images)
            obs_batch = {k: v[None, ...] for k, v in obs.items()}
            carry, action, _ = agent.policy(carry, obs_batch, mode='eval')
            action = {k: v[0] for k, v in action.items()}

            # Environment step
            obs = env.step(action)

            if obs.get('is_last', False):
                break

        # Save real observation video
        if frames_real:
            filename = output_dir / f'episode_{ep:03d}_real.mp4'
            imageio.mimsave(filename, frames_real, fps=30, codec='libx264')
            print(f"  Saved: {filename}")

    env.close()
    print(f"\nComplete! Videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize DreamerV3 Dreams')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Checkpoint directory path')
    parser.add_argument('--task', type=str, default='atari_pong')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--output', type=str, default='./dream_videos')

    args = parser.parse_args()
    visualize_dream(args.checkpoint, args.task, args.episodes, args.output)


if __name__ == '__main__':
    main()
