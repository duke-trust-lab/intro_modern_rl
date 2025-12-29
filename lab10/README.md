# Lab 12 Reinforcement Learning Engineering  
**Is This the Same Experiment?**

## Overview
In earlier labs, your goal was to *get reinforcement learning working*.  
In this lab, your goal is to **engineer reinforcement learning systems that work reliably, reproducibly, and defensibly**.

You will return to a familiar environment and algorithm, but this time your success will be judged not by a single learning curve, but by your **engineering practices**.

A run is not a result.  
A curve is not evidence.  
An RL system must be interrogated.

---

## Learning Objectives
- Engineer RL experiments that are reproducible
- Separate training performance from evaluation performance
- Diagnose instability using multi-seed experiments
- Detect silent regressions using golden trajectory tests
- Make controlled algorithmic changes and justify conclusions

---

## Environment
- **Task:** CartPole-v1  (Lab 4!)
- **Baseline Algorithm:** DQN  
- **Frameworks:** Stable-Baselines3, RL Zoo 3   

---

## Instructions
You will design your own experiment harness following the instructions below.

You may reuse code from earlier labs, RL Zoo examples, or Stable-Baselines3 documentation.  
However, you must adapt it to meet the engineering requirements specified.

---

## Part 1: Baseline Training (Reproducibility First)
Train a baseline DQN agent on CartPole using **RL Zoo 3**.

### Requirements
You must explicitly record:
- Random seed(s)
- Environment ID
- Algorithm and hyperparameters
- Library versions (Python, gymnasium, Stable-Baselines3, torch)
- Training steps
- Wall-clock time

### Checkpoint
Why is running an experiment with `pip install latest` an engineering risk in reinforcement learning?

---

## Part 2: Evaluation Is a Separate System
Training reward is **not** evaluation.

Implement a separate evaluation procedure with the following properties:
- Fixed random seed
- Deterministic (greedy) action selection
- At least 10 evaluation episodes
- Mean and standard deviation of returns reported

You must clearly distinguish:
- Training reward (with exploration)
- Evaluation reward (without exploration)

### Checkpoint
Why can training reward increase while evaluation reward does not?

---

## Part 3: Stability via Multi-Seed Experiments
Repeat the **same experiment** using **at least three different random seeds**.

You should report:
- Mean evaluation performance across seeds
- Variability across seeds (e.g., standard deviation)

### Checkpoint Question
What does “stable learning” mean operationally? Give a concrete definition.

---

## Part 4: Failure Injection
Intentionally introduce **one change** that breaks experimental reliability. Examples include:
- Removing or altering a random seed
- Changing a library version
- Evaluating with stochastic actions
- Mixing training and evaluation environments

Demonstrate how this change affects:
- Reproducibility
- Reported performance
- Confidence in your conclusions

### Checkpoint Question
Which failures are visible in learning curves, and which failures are silent?

---

## Part 5: Golden Trajectory Test
Create a **golden trajectory test** to detect unintended regressions.

1. Fix the environment seed and the policy
2. Roll out the policy for a fixed number of steps
3. Record a compact signature of the trajectory (e.g., a hash of actions or observations)
4. Re-run the rollout and verify that the signature matches

This test should fail if a meaningful implementation or configuration change occurs unintentionally.

### Checkpoint
Why can a golden trajectory test detect bugs that reward curves cannot?

---

## Part 6: Controlled Change: Dueling DQN
Re-implement training **directly in Stable-Baselines3** and replace the standard DQN architecture with **Dueling DQN**.

### Constraints
- Use the same evaluation protocol
- Use the same seed strategy
- Use the same logging and reporting structure

You should compare DQN and Dueling DQN using evaluation performance across multiple seeds.

### Checkpoint
How do you isolate the effect of an architectural change from differences in the experimental setup?
