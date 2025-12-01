#!/usr/bin/env python3
"""
Hyperparameter Tuning Script - Using Optuna to search for optimal AIRL hyperparameters
Imports and reuses training functions from main.py
"""
import os
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np

# Import reusable functions from main.py
from main import (
    load_or_train_expert,
    collect_demonstrations,
    train_airl,
    evaluate_policy
)


# ============================================================================
# Global Configuration
# ============================================================================

ENV_ID = "LunarLander-v2"
OUT_DIR_BASE = "outputs_optuna"  # Dedicated directory for hyperparameter search
OUT_DIR_MAIN = "outputs"  # Main training directory (for loading existing expert)
os.makedirs(OUT_DIR_BASE, exist_ok=True)

# Global variables to store expert and data (avoid repeated training)
EXPERT = None
EXPERT_ENV = None
EXPERT_MEAN = None
TRANSITIONS = None
ROLLOUTS = None


def setup_expert_and_data():
    """Initialize Expert and demonstration data (runs once)"""
    global EXPERT, EXPERT_ENV, EXPERT_MEAN, TRANSITIONS, ROLLOUTS

    if EXPERT is not None:
        print("Using cached expert and demonstrations...")
        return

    print("="*70)
    print("Setting up Expert and Demonstrations (one-time setup)")
    print("="*70)

    # Train/load Expert (load from main directory to avoid repeated training)
    expert, env, expert_mean, expert_std = load_or_train_expert(
        env_id=ENV_ID,
        out_dir=OUT_DIR_MAIN,
        n_envs=16,
        total_timesteps=1_000_000
    )

    # Collect demonstrations
    transitions, rollouts = collect_demonstrations(
        expert, env, min_timesteps=50000
    )

    # Cache to global variables
    EXPERT = expert
    EXPERT_ENV = env
    EXPERT_MEAN = expert_mean
    TRANSITIONS = transitions
    ROLLOUTS = rollouts

    print(f"\nExpert setup complete. Mean reward: {expert_mean:.1f}")
    print(f"Collected {len(transitions)} demonstrations\n")


def objective(trial):
    """
    Optuna optimization objective function

    Args:
        trial: Optuna trial object

    Returns:
        AIRL performance as percentage of Expert
    """
    # Ensure expert and data are ready
    setup_expert_and_data()

    # Create independent output directory for this trial
    trial_out_dir = os.path.join(OUT_DIR_BASE, f"trial_{trial.number}")
    os.makedirs(trial_out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: Testing hyperparameters")
    print(f"{'='*70}")

    # ========================================================================
    # Define hyperparameter search space
    # ========================================================================

    # 1. Network architecture
    reward_net_width = trial.suggest_categorical("reward_net_width", [256, 512, 1024])
    reward_hid_sizes = [reward_net_width, reward_net_width]
    potential_hid_sizes = [reward_net_width, reward_net_width]

    # 2. Reward Network options
    use_next_state = trial.suggest_categorical("use_next_state", [True, False])
    use_done = trial.suggest_categorical("use_done", [True, False])

    # 3. Discriminator hyperparameters
    disc_lr = trial.suggest_float("disc_lr", 1e-5, 1e-3, log=True)
    n_disc_updates = trial.suggest_int("n_disc_updates", 1, 8)
    demo_batch_size = trial.suggest_categorical("demo_batch_size", [256, 512, 1024, 2048])

    # 4. Generator (PPO) hyperparameters
    gen_lr = trial.suggest_float("gen_lr", 1e-5, 1e-3, log=True)
    gen_gae_lambda = trial.suggest_float("gen_gae_lambda", 0.9, 0.99)
    gen_ent_coef = trial.suggest_float("gen_ent_coef", 0.001, 0.1, log=True)

    # 5. Environment and training parameters
    n_envs = trial.suggest_categorical("n_envs", [4, 8, 16])
    gen_train_timesteps_multiplier = trial.suggest_float("gen_timesteps_mult", 0.5, 2.0)

    # 6. Fast evaluation mode: reduce total_timesteps to speed up search
    # Full training uses 2M steps, here we use 200K for quick evaluation
    total_timesteps = trial.suggest_categorical("total_timesteps", [200_000, 500_000])

    print(f"\nHyperparameters for trial {trial.number}:")
    print(f"  Network: [{reward_net_width}, {reward_net_width}]")
    print(f"  use_next_state={use_next_state}, use_done={use_done}")
    print(f"  Disc: lr={disc_lr:.2e}, n_updates={n_disc_updates}, batch={demo_batch_size}")
    print(f"  Gen: lr={gen_lr:.2e}, gae_lambda={gen_gae_lambda:.3f}, ent_coef={gen_ent_coef:.4f}")
    print(f"  Env: n_envs={n_envs}, timesteps_mult={gen_train_timesteps_multiplier:.3f}")
    print(f"  Training: {total_timesteps} steps\n")

    # ========================================================================
    # Train AIRL
    # ========================================================================

    try:
        airl, airl_mean, airl_std = train_airl(
            transitions=TRANSITIONS,
            env_id=ENV_ID,
            out_dir=trial_out_dir,
            # Hyperparameters from trial
            n_envs=n_envs,
            reward_hid_sizes=reward_hid_sizes,
            potential_hid_sizes=potential_hid_sizes,
            use_next_state=use_next_state,
            use_done=use_done,
            disc_learning_rate=disc_lr,
            n_disc_updates_per_round=n_disc_updates,
            demo_batch_size=demo_batch_size,
            gen_learning_rate=gen_lr,
            gen_gae_lambda=gen_gae_lambda,
            gen_ent_coef=gen_ent_coef,
            gen_train_timesteps_multiplier=gen_train_timesteps_multiplier,
            total_timesteps=total_timesteps,
            load_existing=False,  # Train from scratch for each trial
            verbose=0  # Reduce output
        )

        # Compute performance metric: AIRL as percentage of Expert
        performance_ratio = airl_mean / EXPERT_MEAN

        print(f"\n{'='*70}")
        print(f"Trial {trial.number} Results:")
        print(f"  Expert: {EXPERT_MEAN:.1f}")
        print(f"  AIRL:   {airl_mean:.1f} +/- {airl_std:.1f}")
        print(f"  Performance: {100 * performance_ratio:.1f}% of expert")
        print(f"{'='*70}\n")

        # Save results for this trial
        trial.set_user_attr("airl_mean", airl_mean)
        trial.set_user_attr("airl_std", airl_std)
        trial.set_user_attr("performance_ratio", performance_ratio)

        return performance_ratio

    except Exception as e:
        print(f"\nTrial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()


def run_optimization(n_trials=50, n_jobs=1):
    """
    Run hyperparameter optimization

    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (1=serial, >1=parallel)
    """
    # Create Optuna study
    study = optuna.create_study(
        study_name="airl_hyperparameter_search",
        direction="maximize",  # Maximize performance ratio
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=f"sqlite:///{OUT_DIR_BASE}/optuna_study.db",
        load_if_exists=True
    )

    print(f"Starting Optuna optimization with {n_trials} trials...")
    print(f"Results will be saved to: {OUT_DIR_BASE}/\n")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # ========================================================================
    # Report results
    # ========================================================================

    print("\n" + "="*70)
    print("Optimization Complete!")
    print("="*70)

    best_trial = study.best_trial
    print(f"\nBest Trial: {best_trial.number}")
    print(f"  Performance: {100 * best_trial.value:.1f}% of expert")
    print(f"  AIRL Mean: {best_trial.user_attrs['airl_mean']:.1f}")
    print(f"  AIRL Std:  {best_trial.user_attrs['airl_std']:.1f}")

    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save report
    report_path = os.path.join(OUT_DIR_BASE, "optimization_report.txt")
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("AIRL Hyperparameter Optimization Report\n")
        f.write("="*70 + "\n\n")

        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Best trial: {best_trial.number}\n\n")

        f.write(f"Best Performance:\n")
        f.write(f"  Expert mean: {EXPERT_MEAN:.1f}\n")
        f.write(f"  AIRL mean:   {best_trial.user_attrs['airl_mean']:.1f} +/- {best_trial.user_attrs['airl_std']:.1f}\n")
        f.write(f"  Ratio:       {100 * best_trial.value:.1f}% of expert\n\n")

        f.write("Best Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("Top 10 Trials:\n")
        f.write("="*70 + "\n")

        top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)[:10]
        for i, t in enumerate(top_trials, 1):
            if t.value is not None:
                f.write(f"\n{i}. Trial {t.number}: {100 * t.value:.1f}% of expert\n")
                for key, value in t.params.items():
                    f.write(f"   {key}: {value}\n")

    print(f"\nReport saved to: {report_path}")

    # Visualization (optional)
    try:
        import plotly
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_html(os.path.join(OUT_DIR_BASE, "param_importance.html"))

        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_html(os.path.join(OUT_DIR_BASE, "optimization_history.html"))

        print(f"Visualizations saved to: {OUT_DIR_BASE}/")
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")

    return study


# ============================================================================
# Main Program
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AIRL Hyperparameter Optimization with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs (1=serial)")
    args = parser.parse_args()

    # Run optimization
    study = run_optimization(n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("\nHyperparameter search complete!")
    print(f"Best performance: {100 * study.best_value:.1f}% of expert")
    print(f"Results saved to: {OUT_DIR_BASE}/")
