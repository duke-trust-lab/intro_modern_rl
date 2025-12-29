#!/usr/bin/env python3
"""
DreamerV3 Inference and Video Recording Script

Usage:
  python inference_video.py \
    --checkpoint ~/logdir/dreamer/xxx/checkpoint.pkl \
    --task atari_pong \
    --episodes 5 \
    --output ./videos/
"""

import argparse
import pathlib
import sys
from functools import partial as bind

import elements
import embodied
import numpy as np
import imageio

# Add project path
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder))

from dreamerv3.agent import Agent
from dreamerv3 import main as dreamerv3_main


def make_video_logger(output_dir, episode_num):
    """Create video logger"""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    class VideoLogger:
        def add_frame(self, frame):
            """Add frame to video"""
            if frame.dtype == np.uint8:
                frames.append(frame)

        def save(self, name=None):
            """Save video"""
            if not frames:
                print("Warning: No frames to save")
                return

            filename = output_dir / (name or f'episode_{episode_num:03d}.mp4')
            print(f"Saving video: {filename} ({len(frames)} frames)")

            # Save video using imageio
            imageio.mimsave(
                filename,
                frames,
                fps=30,
                codec='libx264',
                quality=8
            )
            frames.clear()

    return VideoLogger()


def inference_with_video(
    checkpoint_path,
    task='atari_pong',
    num_episodes=5,
    output_dir='./videos',
    render=True,
    max_steps=10000,
):
    """Run inference and record video"""

    print("=" * 80)
    print("DreamerV3 Inference and Video Recording")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task: {task}")
    print(f"Episodes: {num_episodes}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Load config - read from training directory
    import ruamel.yaml as yaml

    checkpoint_dir = pathlib.Path(checkpoint_path).parent.parent
    saved_config_path = checkpoint_dir / 'config.yaml'

    print(f"Loading training config: {saved_config_path}")

    if saved_config_path.exists():
        # Use training config
        config = elements.Config(yaml.YAML(typ='safe').load(saved_config_path.read_text()))
        print("Using training config")
    else:
        # Fallback: use default config
        print("Warning: Training config not found, using default config")
        config_file = folder / 'dreamerv3' / 'configs.yaml'
        configs = yaml.YAML(typ='safe').load(config_file.read_text())
        config = elements.Config(configs['defaults'])
        suite = task.split('_')[0]
        if suite in configs:
            config = config.update(configs[suite])
        config = config.update(task=task, seed=0)

    # Ensure CPU usage and update paths
    config = config.update(jax={'platform': 'cpu', 'prealloc': False})
    config = config.update(logdir=f'./eval_logs/{task}')

    # Create environment
    print("\nCreating environment...")
    env = dreamerv3_main.make_env(config, 0)

    # Create agent
    print("Creating agent...")
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}

    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_path, keys=['agent'])

    # Inference loop
    print(f"\nStarting inference ({num_episodes} episodes)...\n")

    for ep in range(num_episodes):
        print(f"Episode {ep + 1}/{num_episodes}")
        video_logger = make_video_logger(output_dir, ep)

        # Reset environment and agent
        obs = env.reset()
        carry = agent.init_policy(batch_size=1)

        episode_reward = 0
        episode_length = 0

        for step in range(max_steps):
            # Convert observation to batch format
            obs_batch = {k: v[None, ...] for k, v in obs.items()}

            # Agent decision
            carry, action, _ = agent.policy(carry, obs_batch, mode='eval')

            # Extract action (remove batch dimension)
            action = {k: v[0] for k, v in action.items()}

            # Environment step
            obs = env.step(action)

            # Record video frame
            if 'image' in obs and render:
                video_logger.add_frame(obs['image'])

            episode_reward += obs.get('reward', 0)
            episode_length += 1

            # Check if done
            if obs.get('is_last', False):
                break

        # Save video
        video_logger.save(f'episode_{ep:03d}_score_{episode_reward:.1f}.mp4')

        print(f"  Done - Score: {episode_reward:.2f}, Length: {episode_length}")

    env.close()
    print(f"\nComplete! Videos saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DreamerV3 Inference and Video Recording')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint file path')
    parser.add_argument('--task', type=str, default='atari_pong',
                        help='Task name (e.g., atari_pong, dmc_walker_walk)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to record')
    parser.add_argument('--output', type=str, default='./videos',
                        help='Video output directory')
    parser.add_argument('--max-steps', type=int, default=10000,
                        help='Maximum steps per episode')
    parser.add_argument('--no-render', action='store_true',
                        help='Do not record video')

    args = parser.parse_args()

    inference_with_video(
        checkpoint_path=args.checkpoint,
        task=args.task,
        num_episodes=args.episodes,
        output_dir=args.output,
        render=not args.no_render,
        max_steps=args.max_steps,
    )


if __name__ == '__main__':
    main()
