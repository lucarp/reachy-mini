# Project 3: A2C (Advantage Actor-Critic)

**Status**: 🚧 Coming Soon
**Difficulty**: ⭐⭐⭐⭐☆
**Estimated Time**: 3 days

---

## Overview

A2C (Advantage Actor-Critic) combines policy-based (actor) and value-based (critic) methods for improved sample efficiency and stability.

**Key Concepts**:
- Separate actor (policy) and critic (value) networks
- Advantage function A(s, a) = Q(s, a) - V(s)
- N-step returns for bias-variance balance
- Parallel environment training

---

## What You'll Learn

- ✅ Actor-critic architecture
- ✅ Advantage function and why it reduces variance
- ✅ N-step TD methods
- ✅ Bias-variance tradeoff
- ✅ Parallel environment rollouts

---

## Implementation Plan

This project will include:

1. **a2c.py** - A2C algorithm with actor and critic networks
2. **train.py** - Training script with parallel environments
3. **test.py** - Evaluation script
4. **README.md** - Complete theory guide (advantage function, TD learning, etc.)

---

## Papers

- [Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)
- [Actor-Critic Algorithms (Konda & Tsitsiklis, 2003)](https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf)

---

## Prerequisites

Complete **Project 1 (REINFORCE)** and **Project 2 (DQN)** first to understand:
- Policy gradients (from REINFORCE)
- Value functions (from DQN)
- A2C combines both!

**Start with Project 1** → [../project1_reinforce/](../project1_reinforce/)
