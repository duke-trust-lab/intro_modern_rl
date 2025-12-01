import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import imageio
from pathlib import Path
import numpy as np
import random

CURRENT_DIR = Path(__file__).parent
ENV_NAME = "MiniGrid-Dynamic-Obstacles-5x5-v0"
SEED = 42


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


def train():
    """Train PPO model on MiniGrid environment"""
    random.seed(SEED)
    np.random.seed(SEED)
    th.manual_seed(SEED)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = ImgObsWrapper(env)
    env.reset(seed=SEED)

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=SEED,
        verbose=1,
        tensorboard_log=str(CURRENT_DIR / "logs"),
    )

    print("Starting training...")
    model.learn(total_timesteps=200000, progress_bar=True)

    model.save(CURRENT_DIR / "model")
    print(f"Model saved to {CURRENT_DIR / 'model.zip'}")
    print("\nTo view losses in TensorBoard, run:")
    print(f"  tensorboard --logdir {CURRENT_DIR / 'logs'}")

    return model


def evaluate(model, n_episodes=10):
    """Evaluate model performance"""
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = ImgObsWrapper(env)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)

    print(f"\nEvaluation results ({n_episodes} episodes):")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    env.close()
    return mean_reward, std_reward


def record_video(model, filename="demo.mp4"):
    """Record evaluation video"""
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = ImgObsWrapper(env)

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
    imageio.mimsave(output_path, frames, fps=1)
    print(f"Video saved to {output_path}")

    env.close()


def main():
    model = train()

    evaluate(model)

    record_video(model)


if __name__ == "__main__":
    main()
