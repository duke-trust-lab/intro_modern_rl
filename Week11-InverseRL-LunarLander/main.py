# filename: main.py
# -*- coding: utf-8 -*-
"""
AIRL Training for LunarLander-v2
Reusable training functions with hyperparameter tuning support
"""
import os, numpy as np, torch, matplotlib.pyplot as plt, imageio.v2 as imageio
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3 import PPO
from imitation.util.util import make_vec_env
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet


# ============================================================================
# Core Reusable Functions
# ============================================================================

def load_or_train_expert(
    env_id="LunarLander-v2",
    out_dir="outputs",
    n_envs=16,
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    seed=42,
    verbose=1
):
    """
    Load or train Expert PPO model

    Returns:
        expert: PPO model
        env: Vectorized environment
        expert_mean: Mean reward from evaluation
        expert_std: Std reward from evaluation
    """
    rng = np.random.default_rng(seed=seed)
    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        rng=rng,
        post_wrappers=[lambda env, idx: RolloutInfoWrapper(env)]
    )

    expert_path = os.path.join(out_dir, "expert_ppo.zip")
    if os.path.exists(expert_path):
        print("Loading existing expert...")
        expert = PPO.load(expert_path, env=env)
        print("Expert loaded successfully!")
    else:
        print("Training expert...")
        expert = PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
        )
        print(f"Training expert for {total_timesteps} steps...")
        expert.learn(total_timesteps)
        expert.save(expert_path)
        print(f"Expert saved to {expert_path}")

    # Evaluate expert
    print("Evaluating expert...")
    expert_mean, expert_std = evaluate_policy(expert, env_id, n_episodes=10)
    print(f"Expert mean reward: {expert_mean:.1f} +/- {expert_std:.1f}")

    if expert_mean < 150:
        print("\nWARNING: Expert did not train well! Mean reward is too low.")
        print("   LunarLander expert should achieve around 200-250.")
    else:
        print("Expert training completed successfully!")

    return expert, env, expert_mean, expert_std


def collect_demonstrations(expert, env, min_timesteps=50000, seed=42):
    """
    Collect expert demonstration trajectories

    Returns:
        transitions: Flattened expert transitions
        rollouts: Raw rollout data
    """
    print(f"\nCollecting expert demonstrations (min_timesteps={min_timesteps})...")
    rng = np.random.default_rng(seed=seed)
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=min_timesteps),
        rng=rng
    )
    transitions = rollout.flatten_trajectories(rollouts)
    print(f"Collected {len(transitions)} expert transitions")
    print(f"Number of trajectories: {len(rollouts)}")
    return transitions, rollouts


def train_airl(
    transitions,
    env_id="LunarLander-v2",
    out_dir="outputs",
    # Environment params
    n_envs=8,
    env_seed=999,
    # Reward network params
    use_next_state=False,
    use_done=True,
    reward_hid_sizes=[512, 512],
    potential_hid_sizes=[512, 512],
    discount_factor=0.99,
    # Generator (PPO) params
    gen_learning_rate=6.35e-4,
    gen_n_steps=2048,
    gen_batch_size=64,
    gen_n_epochs=10,
    gen_gamma=0.99,
    gen_gae_lambda=0.949,
    gen_clip_range=0.2,
    gen_ent_coef=0.0488,
    # Discriminator params
    disc_learning_rate=4.9e-4,
    n_disc_updates_per_round=4,
    demo_batch_size=1024,
    # Training params
    gen_train_timesteps_multiplier=0.813,  # multiplier of n_envs * n_steps
    total_timesteps=2_000_000,
    # Control flags
    load_existing=True,
    verbose=1
):
    """
    Train AIRL model

    Returns:
        airl: AIRL object
        airl_mean: Mean reward from evaluation
        airl_std: Std reward from evaluation
    """
    print("\nSetting up AIRL...")

    # Create AIRL training environment
    env_airl = make_vec_env(
        env_id,
        n_envs=n_envs,
        rng=np.random.default_rng(seed=env_seed),
        post_wrappers=[lambda env, idx: RolloutInfoWrapper(env)]
    )

    # Create reward network
    airl_reward_net = BasicShapedRewardNet(
        env_airl.observation_space,
        env_airl.action_space,
        use_state=True,
        use_action=True,
        use_next_state=use_next_state,
        use_done=use_done,
        reward_hid_sizes=reward_hid_sizes,
        potential_hid_sizes=potential_hid_sizes,
        discount_factor=discount_factor,
    )

    # Calculate gen_train_timesteps
    gen_train_timesteps = int(n_envs * gen_n_steps * gen_train_timesteps_multiplier)

    # Create PPO learner
    airl_learner = PPO(
        "MlpPolicy",
        env_airl,
        verbose=verbose,
        learning_rate=gen_learning_rate,
        n_steps=gen_n_steps,
        batch_size=gen_batch_size,
        n_epochs=gen_n_epochs,
        gamma=gen_gamma,
        gae_lambda=gen_gae_lambda,
        clip_range=gen_clip_range,
        ent_coef=gen_ent_coef,
    )

    # Define save paths
    airl_policy_path = os.path.join(out_dir, "airl_policy.zip")
    airl_reward_net_path = os.path.join(out_dir, "airl_reward_net.pt")

    # Load or train
    if load_existing and os.path.exists(airl_policy_path) and os.path.exists(airl_reward_net_path):
        print("Found existing AIRL model! Loading...")

        # Load policy
        airl_learner = PPO.load(airl_policy_path, env=env_airl)

        # Load reward network
        airl_reward_net.load_state_dict(torch.load(airl_reward_net_path))

        # Rebuild AIRL object
        airl = AIRL(
            demonstrations=transitions,
            demo_batch_size=demo_batch_size,
            venv=env_airl,
            gen_algo=airl_learner,
            reward_net=airl_reward_net,
            allow_variable_horizon=True,
            disc_opt_kwargs={"lr": disc_learning_rate},
            n_disc_updates_per_round=n_disc_updates_per_round,
            gen_train_timesteps=gen_train_timesteps,
        )

        print("AIRL model loaded successfully!")

    else:
        print("Training AIRL from scratch...")

        airl = AIRL(
            demonstrations=transitions,
            demo_batch_size=demo_batch_size,
            venv=env_airl,
            gen_algo=airl_learner,
            reward_net=airl_reward_net,
            allow_variable_horizon=True,
            disc_opt_kwargs={"lr": disc_learning_rate},
            n_disc_updates_per_round=n_disc_updates_per_round,
            gen_train_timesteps=gen_train_timesteps,
        )

        print(f"Training AIRL for {total_timesteps} steps...")
        airl.train(total_timesteps=total_timesteps)
        print("AIRL training completed!")

        # Save model
        print("Saving AIRL model...")
        airl.gen_algo.save(airl_policy_path)
        torch.save(airl._reward_net.state_dict(), airl_reward_net_path)
        print(f"  - Policy saved to: {airl_policy_path}")
        print(f"  - Reward network saved to: {airl_reward_net_path}")

    # Evaluate AIRL (increase evaluation episodes for more stable results)
    print("Evaluating AIRL...")
    airl_mean, airl_std = evaluate_policy(airl.gen_algo, env_id, n_episodes=30)
    print(f"AIRL:   {airl_mean:.1f} +/- {airl_std:.1f}")

    return airl, airl_mean, airl_std


def evaluate_policy(model, env_id, n_episodes=10):
    """
    Evaluate policy performance

    Returns:
        mean_reward: Mean episode reward
        std_reward: Std episode reward
    """
    rewards = []
    for _ in range(n_episodes):
        eval_env = gym.make(env_id)
        obs, _ = eval_env.reset()
        total_reward = 0
        for _ in range(1000):  # Max steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        eval_env.close()
    return np.mean(rewards), np.std(rewards)


def record_gif(model, env_id, out_dir, name):
    """Generate MP4 demo video"""
    env_vis = gym.make(env_id, render_mode="rgb_array")
    frames, obs = [], env_vis.reset()[0]
    total_reward = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_vis.step(action)
        total_reward += reward
        frames.append(env_vis.render())
        if terminated or truncated:
            break
    path = os.path.join(out_dir, f"{name}.mp4")
    imageio.mimsave(path, frames, fps=30, codec='libx264')
    print(f"[Saved] {path} (Episode reward: {total_reward:.1f}, Steps: {len(frames)})")
    return total_reward


def visualize_reward(reward_net, env_id, out_dir, n_samples=50):
    """
    Visualize learned reward function (general version, averaging over sampled states)
    """
    print("\nVisualizing learned rewards...")
    print("Generating position reward heatmap (averaging over sampled states)...")

    x_range = np.linspace(-1.5, 1.5, 60)
    y_range = np.linspace(0, 1.5, 60)
    actions = [0, 1, 2, 3]

    # Sample ranges based on typical LunarLander state ranges
    vx_samples = np.random.uniform(-1.0, 1.0, n_samples)
    vy_samples = np.random.uniform(-1.5, 0.5, n_samples)
    angle_samples = np.random.uniform(-0.5, 0.5, n_samples)
    angular_vel_samples = np.random.uniform(-1.0, 1.0, n_samples)
    left_leg_samples = np.zeros(n_samples)
    right_leg_samples = np.zeros(n_samples)

    # Compute average optimal reward for each position
    max_reward_grid = np.zeros((len(y_range), len(x_range)))
    best_action_grid = np.zeros((len(y_range), len(x_range)), dtype=int)

    print(f"  Computing reward for {len(y_range)}x{len(x_range)} positions with {n_samples} samples each...")

    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            max_rewards_sampled = []
            action_votes = [0, 0, 0, 0]

            for sample_idx in range(n_samples):
                state = np.array([
                    x, y,
                    vx_samples[sample_idx],
                    vy_samples[sample_idx],
                    angle_samples[sample_idx],
                    angular_vel_samples[sample_idx],
                    left_leg_samples[sample_idx],
                    right_leg_samples[sample_idx]
                ], dtype=np.float32)

                rewards_for_actions = []
                for action in actions:
                    state_t = torch.from_numpy(state).unsqueeze(0)
                    action_t = F.one_hot(torch.tensor([action]), num_classes=4).float()
                    with torch.no_grad():
                        reward = reward_net(state_t, action_t, state_t, torch.tensor([False]))
                        rewards_for_actions.append(reward.item())

                max_reward = max(rewards_for_actions)
                best_action = np.argmax(rewards_for_actions)
                max_rewards_sampled.append(max_reward)
                action_votes[best_action] += 1

            max_reward_grid[i, j] = np.mean(max_rewards_sampled)
            best_action_grid[i, j] = np.argmax(action_votes)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(y_range)} rows completed")

    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(max_reward_grid, extent=[x_range[0], x_range[-1], y_range[0], y_range[-1]],
                   origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel('Horizontal Position (x)', fontsize=14)
    ax.set_ylabel('Vertical Position (y)', fontsize=14)
    ax.set_title(f'AIRL Learned Reward Function (General)\nAveraged over {n_samples} sampled states per position',
                 fontsize=16, fontweight='bold')
    ax.axvline(x=0, color='blue', linestyle='--', linewidth=2.5, alpha=0.8, label='Landing Zone (x=0)')
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=2.5, alpha=0.8, label='Ground (y=0)')
    ax.scatter([0], [0], s=200, c='blue', marker='*', edgecolors='white', linewidths=2,
              label='Target Landing', zorder=10)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Learned Reward (higher is better)', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.text(0.02, 0.98, f'Sampled ranges (n={n_samples}):\n'
                        f'  vx: [-1.0, 1.0]\n'
                        f'  vy: [-1.5, 0.5]\n'
                        f'  angle: [-0.5, 0.5]\n'
                        f'  angular_vel: [-1.0, 1.0]',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_simple_position.png"), dpi=150, bbox_inches='tight')
    print(f"[Saved] reward_simple_position.png")
    plt.close()

    print("Reward visualization generated!")


def save_results(out_dir, expert_mean, expert_std, airl_mean, airl_std, transitions, rollouts):
    """Save training results to file"""
    results_path = os.path.join(out_dir, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"AIRL LunarLander-v2 Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Configuration:\n")
        f.write(f"  Expert training steps: 1,000,000\n")
        f.write(f"  Expert demonstrations: 50,000 transitions\n")
        f.write(f"  AIRL training steps: 2,000,000\n")
        f.write(f"  Expert transitions collected: {len(transitions)}\n")
        f.write(f"  Expert trajectories: {len(rollouts)}\n")
        f.write(f"  AIRL network architecture (Optuna-optimized):\n")
        f.write(f"    - Reward net: [512, 512]\n")
        f.write(f"    - Potential net: [512, 512]\n")
        f.write(f"    - Discriminator lr: 4.9e-4, n_updates: 4\n")
        f.write(f"    - Generator lr: 6.35e-4, ent_coef: 0.0488\n")
        f.write(f"    - Batch size: 1024\n")
        f.write(f"\nPerformance (mean +/- std over 10 episodes):\n")
        f.write(f"  Expert:  {expert_mean:.1f} +/- {expert_std:.1f}\n")
        f.write(f"  AIRL:    {airl_mean:.1f} +/- {airl_std:.1f} ({100*airl_mean/expert_mean:.1f}% of expert)\n")
        f.write(f"\nConclusion:\n")
        if airl_mean > 150:
            f.write(f"  AIRL successfully learned from expert demonstrations!\n")
            f.write(f"  AIRL achieved {100*airl_mean/expert_mean:.1f}% of expert performance\n")
        else:
            f.write(f"  AIRL partially learned the task (score: {airl_mean:.1f})\n")
            if airl_mean > 0:
                f.write(f"  AIRL achieved positive rewards\n")
    print(f"[Saved] {results_path}")


# ============================================================================
# Main Program - Using Optuna-optimized parameters
# ============================================================================

def main():
    """Main training workflow - Using Optuna-optimized best parameters"""
    ENV_ID = "LunarLander-v2"
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*70)
    print("AIRL Training for LunarLander-v2")
    print("Using Optuna-optimized hyperparameters (95.9% of expert)")
    print("="*70)

    # Step 1: Train/load Expert
    print("\n[Step 1] Expert Training")
    expert, env, expert_mean, expert_std = load_or_train_expert(
        env_id=ENV_ID,
        out_dir=OUT_DIR,
        n_envs=16,
        total_timesteps=1_000_000
    )

    # Step 2: Collect demonstrations
    print("\n[Step 2] Collecting Demonstrations")
    transitions, rollouts = collect_demonstrations(expert, env, min_timesteps=50000)

    # Step 3: Train AIRL (using Optuna best parameters)
    print("\n[Step 3] AIRL Training")
    print("="*70)
    print("Best hyperparameters from 200 trials search (95.9% of expert):")
    print("  Network: reward_net=[512,512], potential_net=[512,512]")
    print("  Discriminator: lr=4.9e-4, n_updates=4")
    print("  Generator: lr=6.35e-4, ent_coef=0.0488, gae_lambda=0.949")
    print("  Training: batch_size=1024")
    print("="*70)

    airl, airl_mean, airl_std = train_airl(
        transitions=transitions,
        env_id=ENV_ID,
        out_dir=OUT_DIR,
        # Optuna-optimized parameters
        n_envs=8,
        reward_hid_sizes=[512, 512],
        potential_hid_sizes=[512, 512],
        use_next_state=False,
        gen_learning_rate=6.35e-4,
        gen_gae_lambda=0.949,
        gen_ent_coef=0.0488,
        disc_learning_rate=4.9e-4,
        n_disc_updates_per_round=4,
        demo_batch_size=1024,
        gen_train_timesteps_multiplier=0.813,
        total_timesteps=2_000_000,
        load_existing=True
    )

    # Step 4: Generate demo videos
    print("\n[Step 4] Generating Demo Videos")
    record_gif(expert, ENV_ID, OUT_DIR, "expert_demo")
    record_gif(airl.gen_algo, ENV_ID, OUT_DIR, "airl_demo")

    # Step 5: Visualize Reward
    print("\n[Step 5] Visualizing Rewards")
    visualize_reward(airl._reward_net, ENV_ID, OUT_DIR, n_samples=50)

    # Save results
    save_results(OUT_DIR, expert_mean, expert_std, airl_mean, airl_std, transitions, rollouts)

    # Final summary
    print(f"\nDone. All output files are in: {os.path.abspath(OUT_DIR)}")
    print(f"\n{'='*60}")
    print(f"Final Results Summary:")
    print(f"{'='*60}")
    print(f"Expert:  {expert_mean:.1f} +/- {expert_std:.1f} (baseline)")
    print(f"AIRL:    {airl_mean:.1f} +/- {airl_std:.1f} ({100*airl_mean/expert_mean:.1f}% of expert)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
