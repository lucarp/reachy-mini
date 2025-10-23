# Project 2: DQN (Deep Q-Network)

**Status**: 🚧 Coming Soon
**Difficulty**: ⭐⭐⭐☆☆
**Estimated Time**: 2-3 days

---

## Overview

DQN (Deep Q-Network) is a value-based RL algorithm that learns to estimate action-values Q(s, a) and selects actions greedily.

**Key Innovations**:
- Experience replay buffer for sample efficiency
- Target network for stability
- ε-greedy exploration
- Handles discrete action spaces

---

## What You'll Learn

- ✅ Q-learning and Bellman equation
- ✅ Experience replay for off-policy learning
- ✅ Target networks to stabilize training
- ✅ Epsilon-greedy exploration strategy
- ✅ Discretization of continuous action spaces

---

## Implementation Plan

This project will include:

1. **dqn.py** - DQN algorithm with replay buffer and target network
2. **replay_buffer.py** - Experience replay implementation
3. **train.py** - Training script
4. **test.py** - Evaluation script
5. **README.md** - Complete theory guide (Bellman equation, DQN architecture, etc.)

---

## Papers

- [Playing Atari with Deep Reinforcement Learning (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

---

## Note

Complete **Project 1: REINFORCE** first to understand policy-based methods before learning value-based methods.

**Continue with Project 1** → [../project1_reinforce/](../project1_reinforce/)
