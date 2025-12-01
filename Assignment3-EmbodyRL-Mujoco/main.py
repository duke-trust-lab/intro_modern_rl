import os
from pathlib import Path

import gymnasium as gym
import imageio
from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env


ENV_NAME = "Reacher-v5"
CURRENT_DIR = Path(__file__).parent
MODEL_PATH = CURRENT_DIR / "ppo-Reacher-v5.zip"


def train():
    """Train PPO on Reacher-v5 environment"""
    env = make_vec_env(ENV_NAME, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=42
    )

    model.learn(2_000_000, progress_bar=True)
    env.close()

    return model


def evaluate(model, n_episodes=100):
    """Evaluate model performance"""
    env = gym.make(ENV_NAME)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)

    print(f"Evaluation results ({n_episodes} episodes):")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward


def record_video(model, filename="demo.mp4"):
    """Record evaluation video"""
    env = gym.make(ENV_NAME, render_mode="rgb_array")

    frames = []
    obs, _ = env.reset()
    frames.append(env.render())

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(env.render())

    output_path = CURRENT_DIR / filename
    imageio.mimsave(output_path, frames, fps=12)
    print(f"Video saved to {output_path}")

    env.close()

def load_or_train_model():
    """Load existing model or train a new one"""
    env = gym.make(ENV_NAME)

    if MODEL_PATH.exists():
        print(f"Loading model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Training new model...")
        model = train()
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    env.close()
    return model


def main():
    model = load_or_train_model()
    evaluate(model)
    record_video(model)

if __name__ == "__main__":
    main()

