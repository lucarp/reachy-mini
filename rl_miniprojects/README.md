# 🎓 RL Mini-Projects

**Learn Deep Reinforcement Learning through Progressive Hands-On Projects**

This is a pedagogical learning path designed to teach you RL algorithms from fundamentals to state-of-the-art by implementing 4 algorithms that solve the same ball-tracking task.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
cd rl_miniprojects
pip install -r requirements.txt

# 2. Verify environment works
python -c "from shared import BallTrackerEnv; env = BallTrackerEnv(); print('✅ Setup complete!')"

# 3. Start with Project 1
cd project1_reinforce
python train.py

# 4. Evaluate trained agent
python test.py --baseline
```

---

## 📚 Projects

### [Project 1: REINFORCE](project1_reinforce/) - ⭐⭐☆☆☆
**Algorithm**: Policy Gradient (Williams, 1992)
**Time**: 2 days
**Learn**: Policy gradients, Monte Carlo returns, Baseline

```bash
cd project1_reinforce
python train.py --episodes 500
python test.py --baseline
```

---

### Project 2: DQN - ⭐⭐⭐☆☆ (Coming Soon)
**Algorithm**: Deep Q-Network (Mnih et al., 2015)
**Time**: 2-3 days
**Learn**: Q-learning, Replay buffer, Target networks

---

### Project 3: A2C - ⭐⭐⭐⭐☆ (Coming Soon)
**Algorithm**: Advantage Actor-Critic (Mnih et al., 2016)
**Time**: 3 days
**Learn**: Actor-critic, Advantage estimation, N-step returns

---

### Project 4: PPO - ⭐⭐⭐⭐⭐ (Coming Soon)
**Algorithm**: Proximal Policy Optimization (Schulman et al., 2017)
**Time**: 4 days
**Learn**: Clipped objective, GAE, Mini-batches

---

## 🎮 The Task: Ball Tracking

All projects solve the same task:

- **Goal**: Keep a moving ball centered in the robot's camera view
- **Robot**: Control head pitch and yaw
- **Episode**: 30 seconds or until ball exits frame
- **Reward**: Negative distance from center + bonuses/penalties

**State**: `[ball_x, ball_y, head_pitch, head_yaw, ball_vel_x, ball_vel_y]` (6D)
**Action**: `[delta_pitch, delta_yaw]` (2D continuous)

---

## 📖 Documentation

- **[LEARNING_GUIDE.md](LEARNING_GUIDE.md)** - Complete curriculum roadmap (read this first!)
- **[shared/](shared/)** - Shared utilities (environment, visualization, utils)
- **[project1_reinforce/README.md](project1_reinforce/README.md)** - REINFORCE theory and usage

---

## 📊 Expected Results

After training all algorithms:

| Algorithm | Mean Reward | Episodes to Converge | Sample Efficiency |
|-----------|-------------|----------------------|-------------------|
| Random Baseline | -150 | N/A | N/A |
| REINFORCE | -40 | 500-800 | ⭐☆☆☆☆ |
| DQN | -35 | 300-500 | ⭐⭐⭐☆☆ |
| A2C | -25 | 200-400 | ⭐⭐⭐⭐☆ |
| PPO | -20 | 150-300 | ⭐⭐⭐⭐⭐ |

---

## 🎯 Learning Outcomes

After completing all 4 projects, you will:

- ✅ Understand policy-based vs value-based RL
- ✅ Implement neural network policies and value functions
- ✅ Master on-policy vs off-policy learning
- ✅ Use experience replay and target networks
- ✅ Understand actor-critic architectures
- ✅ Implement PPO (used in Phase 3!)
- ✅ Debug and tune RL training
- ✅ Compare algorithms empirically

---

## 🛠️ Project Structure

```
rl_miniprojects/
├── LEARNING_GUIDE.md          # Complete curriculum (START HERE!)
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── shared/                     # Shared code for all projects
│   ├── ball_tracker_env.py    # Gym environment
│   ├── visualizer.py          # Plotting utilities
│   └── utils.py               # Helper functions
├── project1_reinforce/         # Project 1: REINFORCE
│   ├── README.md              # Theory and usage
│   ├── reinforce.py           # Algorithm implementation
│   ├── train.py               # Training script
│   └── test.py                # Evaluation script
├── project2_dqn/               # Project 2: DQN (coming soon)
├── project3_a2c/               # Project 3: A2C (coming soon)
└── project4_ppo/               # Project 4: PPO (coming soon)
```

---

## 📚 Study Approach

For each project:

1. **Read the theory** (project README.md)
2. **Understand the code** (read before running)
3. **Train the agent** (observe learning curves)
4. **Experiment** (try different hyperparameters)
5. **Compare results** (with previous projects)

**Timeline**: 1-2 weeks total (2-4 hours per day)

---

## 🐛 Debugging Tips

### Agent doesn't learn
- Check learning rate (try 1e-5 to 1e-3)
- Verify reward function (print rewards)
- Check policy initialization
- Visualize network outputs

### Training is unstable
- Lower learning rate by 10x
- Add/increase gradient clipping
- Normalize returns/advantages
- Increase batch size

### Reward plateaus
- Try different hyperparameters
- Check if task is achievable
- Visualize agent behavior
- Compare with baseline

---

## 🎯 Next Steps

After completing all 4 projects:

1. **Create final comparison report** (all algorithms)
2. **Present your findings** (10-min presentation)
3. **Read Phase 3 plan** (Deep RL with Human-in-the-Loop)
4. **Apply PPO to Phase 3** (you'll be ready!)

---

## 🙏 Resources

### Must-Read
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Sutton & Barto - RL Textbook](http://incompleteideas.net/book/)

### Video Lectures
- [DeepMind x UCL RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb)
- [Berkeley CS 285](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Papers
- [REINFORCE (1992)](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
- [DQN (2015)](https://www.nature.com/articles/nature14236)
- [A3C (2016)](https://arxiv.org/abs/1602.01783)
- [PPO (2017)](https://arxiv.org/abs/1707.06347)

---

## ✅ Completion Checklist

- [ ] Read LEARNING_GUIDE.md completely
- [ ] Complete Project 1: REINFORCE
- [ ] Complete Project 2: DQN
- [ ] Complete Project 3: A2C
- [ ] Complete Project 4: PPO
- [ ] Create final comparison analysis
- [ ] Can explain each algorithm to others
- [ ] Ready for Phase 3!

---

**Start your RL journey now! Open [LEARNING_GUIDE.md](LEARNING_GUIDE.md) and begin with Project 1.** 🚀
