#!/usr/bin/env python3
"""End-to-End RLHF Pipeline: Reward Training -> PPO Alignment -> Evaluation"""

import os
import sys
import subprocess


REWARD_DIR = "reward_model"
ALIGNED_DIR = "aligned_model"


def model_exists(path):
    """Check if a trained model already exists"""
    return os.path.isdir(path)


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: {description} failed")
        sys.exit(1)


def main():
    print("RLHF Pipeline\n")
    
    # Step 1: Train reward model
    print("Step 1: Checking reward model...")
    if model_exists(REWARD_DIR):
        print(f"Found existing reward model: {REWARD_DIR}")
    else:
        run_command("uv run python reward_model.py", "Training reward model")
    
    # Step 2: Train PPO alignment
    print("\nStep 2: Checking aligned model...")
    if model_exists(ALIGNED_DIR):
        print(f"Found existing aligned model: {ALIGNED_DIR}")
    else:
        run_command("uv run python ppo_training.py", "Training PPO alignment")
    
    # Step 3: Evaluate
    print("\nStep 3: Evaluating alignment...")
    run_command("uv run python evaluate_alignment.py", "Evaluating models")
    
    print("\nRLHF Pipeline Complete!")


if __name__ == "__main__":
    main()
