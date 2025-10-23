# 🎓 RL Mini-Projects: Progressive Learning Path

**Your Journey from RL Fundamentals to State-of-the-Art**

This learning curriculum takes you through 4 carefully designed projects that teach reinforcement learning from the ground up. Each project builds on the previous, using the same **Ball Tracking** task so you can directly compare algorithms.

---

## 📋 Overview

| Project | Algorithm | Type | Difficulty | Time | Key Concepts |
|---------|-----------|------|------------|------|--------------|
| **1** | REINFORCE | Policy-based | ⭐⭐☆☆☆ | 2 days | Policy gradients, Monte Carlo, Baseline |
| **2** | DQN | Value-based | ⭐⭐⭐☆☆ | 2-3 days | Q-learning, Replay buffer, Target networks |
| **3** | A2C | Actor-Critic | ⭐⭐⭐⭐☆ | 3 days | Actor-critic, Advantage, N-step returns |
| **4** | PPO | Actor-Critic | ⭐⭐⭐⭐⭐ | 4 days | Clipped objective, GAE, Mini-batches |

**Total Time**: 11-12 days (1-2 weeks)

---

## 🎯 Learning Objectives

By the end of this curriculum, you will:

### Fundamentals
- ✅ Understand the RL problem formulation (MDP, Bellman equations)
- ✅ Implement neural network policies and value functions
- ✅ Master on-policy vs off-policy learning
- ✅ Debug RL training (vanishing rewards, high variance, instability)

### Algorithms
- ✅ Implement REINFORCE from scratch (policy gradients)
- ✅ Implement DQN from scratch (value-based methods)
- ✅ Implement A2C from scratch (actor-critic)
- ✅ Implement PPO from scratch (advanced policy optimization)
- ✅ Compare your implementations with stable-baselines3

### Practical Skills
- ✅ Design reward functions for robotics tasks
- ✅ Tune hyperparameters effectively
- ✅ Visualize and analyze training curves
- ✅ Evaluate policies and measure performance
- ✅ Identify and fix common RL bugs

---

## 🧭 Learning Path

### Week 1: Foundations

#### Days 1-2: Project 1 - REINFORCE
**Focus**: Policy Gradient Basics

**You'll Learn**:
- Policy gradient theorem and derivation
- Monte Carlo return estimation
- Variance reduction with baselines
- Gaussian policies for continuous actions

**Deliverables**:
- Working REINFORCE implementation
- Trained agent that tracks the ball
- Understanding of policy-based methods

**Study Materials**:
- `project1_reinforce/README.md` - Complete theory guide
- Sutton & Barto Chapter 13
- [Spinning Up: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

---

#### Days 3-5: Project 2 - DQN
**Focus**: Value-Based Methods

**You'll Learn**:
- Q-learning and Bellman equation
- Experience replay for sample efficiency
- Target networks for stability
- Epsilon-greedy exploration
- Discrete action spaces

**Deliverables**:
- Working DQN implementation with replay buffer
- Compare sample efficiency with REINFORCE
- Understanding of off-policy learning

**Study Materials**:
- `project2_dqn/README.md` - Complete theory guide
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Spinning Up: DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)

**Key Comparison**: REINFORCE vs DQN
- On-policy vs off-policy
- Policy-based vs value-based
- Sample efficiency differences

---

### Week 2: Advanced Methods

#### Days 6-8: Project 3 - A2C
**Focus**: Actor-Critic Architecture

**You'll Learn**:
- Actor-critic framework
- Advantage function A(s, a) = Q(s, a) - V(s)
- N-step returns vs Monte Carlo
- Parallel environment training
- Continuous action spaces with critic

**Deliverables**:
- Working A2C implementation
- Compare with REINFORCE (same policy gradient but with critic)
- Understanding of bias-variance tradeoff

**Study Materials**:
- `project3_a2c/README.md` - Complete theory guide
- Sutton & Barto Chapter 13.5
- [A3C Paper (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)

**Key Comparison**: A2C vs REINFORCE
- Advantage vs return
- N-step vs Monte Carlo
- Bias-variance tradeoff

---

#### Days 9-12: Project 4 - PPO
**Focus**: State-of-the-Art Policy Optimization

**You'll Learn**:
- Importance sampling and policy ratios
- Clipped objective function
- Generalized Advantage Estimation (GAE)
- Mini-batch updates
- Trust region methods intuition

**Deliverables**:
- Working PPO implementation from scratch
- Compare with stable-baselines3 PPO
- Final comparison of all 4 algorithms
- Comprehensive analysis and writeup

**Study Materials**:
- `project4_ppo/README.md` - Complete theory guide
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

**Key Comparison**: PPO vs A2C
- Clipped objective vs vanilla policy gradient
- Mini-batch updates vs full gradient
- Stability and sample efficiency

---

## 🎮 The Task: Ball Tracking

All 4 projects solve the same task for direct comparison:

### Environment
- **Red ball** moves in front of camera (sinusoidal or random path)
- **Robot head** must track it by controlling pitch and yaw
- **Episode**: 30 seconds or until ball exits frame

### State Space (6D)
```
[ball_x, ball_y, head_pitch, head_yaw, ball_vel_x, ball_vel_y]
```

### Action Space
- **Continuous** (REINFORCE, A2C, PPO): `[delta_pitch, delta_yaw]` ∈ [-0.1, 0.1]²
- **Discrete** (DQN): 9 actions (stay, N, NE, E, SE, S, SW, W, NW)

### Reward Function
```python
reward = -distance_from_center         # Track the ball
         - 0.1 * action_magnitude       # Smooth movements
         + 1.0 if distance < 0.1        # Bonus for centering
         - 10.0 if out_of_frame         # Penalty for losing ball
```

### Success Metrics
- **Average distance** < 0.3 (good tracking)
- **Average reward** > -30 (strong performance)
- **Success rate** > 70% (ball stays in frame)

---

## 📊 Expected Performance

After training, here's what you should see:

| Algorithm | Mean Reward | Sample Efficiency | Stability | Best For |
|-----------|-------------|-------------------|-----------|----------|
| **Random** | -150 ± 30 | N/A | N/A | Baseline |
| **REINFORCE** | -40 ± 20 | ⭐☆☆☆☆ | ⭐⭐☆☆☆ | Learning PG |
| **DQN** | -35 ± 15 | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | Discrete actions |
| **A2C** | -25 ± 12 | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | Parallel envs |
| **PPO** | -20 ± 10 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | State-of-the-art |

**Training Episodes Required**:
- REINFORCE: 500-800
- DQN: 300-500
- A2C: 200-400
- PPO: 150-300

---

## 🛠️ Setup

### 1. Install Dependencies

```bash
cd rl_miniprojects
pip install -r requirements.txt
```

### 2. Verify Environment

```bash
python -c "from shared import BallTrackerEnv; env = BallTrackerEnv(); print('✅ Environment works!')"
```

### 3. Start with Project 1

```bash
cd project1_reinforce
python train.py
```

---

## 📚 Study Approach

### For Each Project:

#### Phase 1: Theory (30-60 min)
1. Read the project README completely
2. Work through the mathematics on paper
3. Answer the study questions
4. Watch recommended video lectures

#### Phase 2: Implementation (2-4 hours)
1. Read through the code carefully
2. Add comments explaining each part
3. Understand every line before running
4. Identify the key algorithm components

#### Phase 3: Experimentation (2-3 hours)
1. Train with default hyperparameters
2. Observe and understand the learning curves
3. Try different hyperparameters
4. Compare with random baseline

#### Phase 4: Analysis (1-2 hours)
1. Evaluate the trained agent
2. Visualize tracking performance
3. Write down key observations
4. Prepare questions for next project

### Study Tips

✅ **Do**:
- Implement from scratch before looking at code
- Draw diagrams of algorithm flow
- Keep a learning journal
- Experiment with hyperparameters
- Compare algorithms quantitatively

❌ **Don't**:
- Copy-paste without understanding
- Skip the theory sections
- Move on if confused (ask questions!)
- Only run default parameters
- Forget to save your results

---

## 🐛 Debugging Guide

### Common Issues

#### 1. Agent doesn't learn (flat reward curve)

**Possible causes**:
- Learning rate too low/high
- Reward function poorly scaled
- Policy initialization bad
- Bug in gradient computation

**Debug steps**:
1. Check if losses are changing
2. Verify reward function (print rewards)
3. Try different learning rates [1e-5, 1e-4, 1e-3]
4. Check policy outputs (are they reasonable?)

#### 2. Training is unstable (reward oscillates wildly)

**Possible causes**:
- Learning rate too high
- No gradient clipping
- High variance gradients (REINFORCE)
- Replay buffer too small (DQN)

**Debug steps**:
1. Lower learning rate by 10x
2. Add/increase gradient clipping
3. Increase batch size or use baseline
4. Normalize returns/advantages

#### 3. Agent learns then forgets (catastrophic forgetting)

**Possible causes**:
- On-policy algorithms (REINFORCE, A2C, PPO)
- Learning rate too high
- No target network (DQN)

**Debug steps**:
1. This is normal for on-policy methods!
2. Lower learning rate
3. Increase training steps per update
4. For DQN: verify target network updates

---

## 📈 Tracking Progress

Keep a training log for each project:

```markdown
## Project X: [Algorithm]

**Date**: YYYY-MM-DD
**Time Spent**: X hours

### What I Learned
- [Key concept 1]
- [Key concept 2]

### Results
- Mean Reward: X.XX ± Y.YY
- Training Episodes: NNN
- Best Hyperparameters: {...}

### Challenges
- [Issue and how I solved it]

### Questions
- [Unanswered questions]

### Next Steps
- [What to try next]
```

---

## 🎓 After Completion

Once you've finished all 4 projects:

### 1. Comprehensive Comparison

Create a final report comparing:
- Sample efficiency (episodes to convergence)
- Final performance (mean reward, tracking accuracy)
- Stability (variance in training)
- Computational cost (time per episode)
- Ease of implementation
- Hyperparameter sensitivity

### 2. Presentation

Prepare a 10-minute presentation explaining:
- Each algorithm's key idea
- When to use each algorithm
- Empirical comparison on ball tracking
- Lessons learned

### 3. Extension Ideas

Take your learning further:

**Harder Tasks**:
- Multi-object tracking
- Moving obstacles
- Limited field of view
- Noisy observations

**Advanced Algorithms**:
- DDPG (continuous control)
- SAC (maximum entropy RL)
- TD3 (twin delayed DDPG)

**Real Robot**:
- Deploy PPO to real Reachy Mini
- Handle real-world noise and delays
- Compare sim-to-real transfer

---

## 📚 Recommended Resources

### Textbooks
1. **Sutton & Barto (2018)** - "Reinforcement Learning: An Introduction"
   - Chapter 13: Policy Gradient Methods
   - Chapter 6: Temporal Difference Learning
   - Free online: http://incompleteideas.net/book/

2. **Goodfellow, Bengio & Courville (2016)** - "Deep Learning"
   - Chapter 20: Deep Generative Models
   - Free online: https://www.deeplearningbook.org/

### Online Courses
1. **DeepMind x UCL RL Course** - https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb
2. **Berkeley CS 285** - http://rail.eecs.berkeley.edu/deeprlcourse/
3. **Spinning Up in Deep RL** - https://spinningup.openai.com/

### Papers (Chronological)
1. [REINFORCE (1992)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) - Williams
2. [DQN (2015)](https://www.nature.com/articles/nature14236) - Mnih et al.
3. [A3C (2016)](https://arxiv.org/abs/1602.01783) - Mnih et al.
4. [PPO (2017)](https://arxiv.org/abs/1707.06347) - Schulman et al.
5. [GAE (2016)](https://arxiv.org/abs/1506.02438) - Schulman et al.

### Blog Posts
1. [Lilian Weng - Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
2. [Andrej Karpathy - Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
3. [OpenAI - Spinning Up](https://spinningup.openai.com/en/latest/)

---

## 🏆 Certification

After completing all 4 projects, you can:

1. ✅ Explain the difference between value-based and policy-based RL
2. ✅ Implement any of these algorithms from scratch
3. ✅ Choose the right algorithm for a new task
4. ✅ Debug RL training effectively
5. ✅ Tune hyperparameters systematically
6. ✅ Understand the Phase 3 implementation plan

**You're now ready for Phase 3: Deep RL with Human-in-the-Loop!**

---

## 📞 Getting Help

### Self-Debugging Checklist
1. Read error message carefully
2. Check input/output shapes
3. Verify data preprocessing
4. Print intermediate values
5. Test components in isolation

### Resources
- Stack Overflow: [reinforcement-learning] tag
- Reddit: r/reinforcementlearning
- Discord: RL Discord servers

### Common Questions

**Q: Which algorithm should I use for my task?**

A:
- **Discrete actions**: DQN or PPO
- **Continuous actions**: PPO or SAC
- **Sample efficiency critical**: DQN or SAC
- **Stability critical**: PPO
- **Learning**: Start with PPO (most reliable)

**Q: Why is my algorithm not learning?**

A: See debugging guide above. Most common: learning rate, reward scaling, or bug in implementation.

**Q: How do I choose hyperparameters?**

A: Start with defaults from papers, then grid search learning rate first (most important).

---

## 🎯 Success Criteria

You've successfully completed the curriculum when:

- [ ] All 4 algorithms implemented and working
- [ ] Understand the math behind each algorithm
- [ ] Can explain trade-offs between algorithms
- [ ] Have comparison plots showing all 4 algorithms
- [ ] Can debug RL training issues independently
- [ ] Feel confident applying RL to new problems

**Time to move on to Phase 3!** 🚀

---

## 🙏 Acknowledgments

This curriculum is inspired by:
- OpenAI Spinning Up
- Berkeley CS 285
- DeepMind x UCL RL Course
- Sutton & Barto's Textbook

Built for PhD research on embodied AI with the Reachy Mini robot.
