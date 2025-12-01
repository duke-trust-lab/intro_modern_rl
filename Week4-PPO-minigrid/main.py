
from pathlib import Path
import imageio
import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
import random
import numpy as np
import torch as th


import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

CURRENT_DIRECTORY = Path(__file__).parent
model_output = CURRENT_DIRECTORY / "ppo_minigrid.zip"



# Official recommendation: custom feature extractor for MiniGrid image input
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
	def __init__(self, observation_space, features_dim=128):
		super().__init__(observation_space, features_dim)
		n_input_channels = observation_space.shape[0]
		self.cnn = nn.Sequential(
			nn.Conv2d(n_input_channels, 16, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(16, 32, (2, 2)),
			nn.ReLU(),
			nn.Conv2d(32, 64, (2, 2)),
			nn.ReLU(),
			nn.Flatten(),
		)
		with th.no_grad():
			n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
		self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
	def forward(self, observations):
		return self.linear(self.cnn(observations))

policy_kwargs = dict(
	features_extractor_class=MinigridFeaturesExtractor,
	features_extractor_kwargs=dict(features_dim=128),
)



# Create MiniGrid-Empty-8x8-v0 environment with image observation
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)
env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)


if not model_output.exists():
	print("Training new model...")
	model = PPO(
		"CnnPolicy",
		env,
		policy_kwargs=policy_kwargs,
		verbose=1,
	)
	model.learn(total_timesteps=int(2e5), progress_bar=True)
	model.save(model_output)
else:
	print("Model already exists, loading...")
	model = PPO.load(model_output, env=env)


# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

def record_video(env, agent):
	frames = []
	state, _ = env.reset()
	# Record the first frame
	frames.append(env.render())

	done = False
	while not done:
		action, _ = agent.predict(state, deterministic=True)
		state, _, terminated, truncated, _ = env.step(action)
		frames.append(env.render())
		done = terminated or truncated

	# Save the recorded frames as a video
	imageio.mimsave(CURRENT_DIRECTORY / "demo.mp4", frames, fps=1)

record_video(env, model)