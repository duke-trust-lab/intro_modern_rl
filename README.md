# Introduction to Modern Reinforcement Learning - Lab

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## About the course
#### Introduction to Modern Reinforcement Learning (AIPI 590)
**Taught by Dr. Brinnae Bent, Duke University**

This course provides a comprehensive, hands-on introduction to reinforcement learning (RL), bridging foundational theory with state-of-the-art methods used in modern AI systems. Students will learn how agents learn from interaction through the lens of Markov Decision Processes, value functions, and policy optimization. Beginning with classical tabular methods, the course progresses through deep RL architectures (DQN, PPO), human-in-the-loop learning (RLHF), and advanced topics in RL.

Weekly labs emphasize implementation and experimentation and four open-ended challenges integrate concepts like safety, generalization, and alignment. By the end of the semester, students will be equipped to design, train, and critically evaluate RL agents.

Please view the course website for more information: [Website](https://www.tomorrowtoolbox.com/courses/imrl)

## Labs

| Lab | Title | Objectives |
|-----|-------|------------|
| **1** | Alignment Cleaning Robot Challenge | Warm up to reinforcement learning |
| **2** | Tabular RL on FrozenLake: Planning vs Learning | Describe an RL problem as an MDP ⟨S, A, P, R, γ⟩; Compute an optimal value function using Value Iteration (planning; model-based); Learn an optimal action-value function using Q-learning (learning; model-free); Explain how both use Bellman optimality backups |
| **3** | Reward Design & Failure Modes | Design reward functions and explain behavioral implications; Anticipate unintended behaviors induced by reward optimization; Diagnose and mitigate reward-design failures; Connect reward specification choices to real-world RL failures |
| **4** | CartPole with DQN Variations | Train a baseline DQN using RL Zoo 3; Diagnose stability via learning curves and evaluation; Implement Dueling DQN; Independently implement Double DQN |
| **5** | PPO on MiniGrid Dynamic Obstacles | Identify reward misalignment as a cause of RL failure; Understand reward sparsity and long horizons; Apply reward shaping to enable successful learning |
| **6** | Human-in-the-Loop RL (Part 1) | Design an end-to-end system for collecting human preference data; Construct a small, targeted prompt dataset; Understand how interface, logging, and data-collection choices shape training signals |
| **7** | Human-in-the-Loop RL (Part 2) | Train a model using preference-based objectives; Compare DPO vs KL-regularized RL; Observe alignment drift and over-optimization |
| **8** | World Models with DreamerV3 | Explain world-model RL and latent imagination; Describe the DreamerV3 loop; Run DreamerV3 end-to-end and evaluate behavior |
| **9** | AIRL on Lunar Lander | Explain the difference between RL, IRL, and imitation learning; Describe how AIRL learns rewards adversarially; Evaluate whether learned policies match expert behavior |
| **10** | Is This the Same Experiment? | Engineer reproducible RL experiments; Separate training from evaluation; Diagnose instability with multi-seed experiments; Detect regressions using golden trajectory tests; Make controlled algorithmic changes |

## Additional Code
Xiaoquan Kong has put together additional code examples beyond the labs, which can be found in the subrepo 'examples'.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

#### Third-Party Code

This repository includes third-party code with separate licensing:

- **lab8**: Contains code from [DreamerV3](https://github.com/danijar/dreamerv3) by Danijar Hafner, licensed under the MIT License. 

## Contributors

- **Brinnae Bent, PhD** ([@brinnaebent](https://github.com/brinnaebent)) - Course designer and instructor
- **Xiaoquan Kong** ([@howl-anderson](https://github.com/howl-anderson)) - Initial code implementation (2025)
