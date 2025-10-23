# Project 1: REINFORCE (Policy Gradient)

**Learning Time**: 2 days
**Difficulty**: ⭐⭐☆☆☆ (Beginner)
**Algorithm Type**: On-policy, Model-free, Policy-based

---

## 📚 Overview

REINFORCE is the simplest and most fundamental **policy gradient** algorithm in reinforcement learning. It directly optimizes the policy by following the gradient of expected return.

**Key Idea**: Instead of learning a value function (like Q-learning), REINFORCE learns a policy π(a|s) that directly maps states to actions.

---

## 🎯 Learning Objectives

After completing this project, you will understand:

1. ✅ **Policy Gradient Theorem** - The mathematical foundation of policy gradients
2. ✅ **Monte Carlo Returns** - How to compute cumulative rewards
3. ✅ **Baseline** - Why and how to reduce variance
4. ✅ **On-policy Learning** - Difference between on-policy and off-policy
5. ✅ **Episode-based Updates** - Training after full episode completion
6. ✅ **Stochastic Policies** - Why exploration naturally emerges from stochastic policies

---

## 📖 Theory

### The Goal of RL

Find policy π that maximizes expected return:

```
J(π) = E_{τ~π} [ Σ_t γ^t r_t ]
```

where τ is a trajectory (s₀, a₀, r₁, s₁, a₁, ...).

### Policy Gradient Theorem

The **policy gradient theorem** tells us how to improve the policy:

```
∇_θ J(θ) = E_{τ~π_θ} [ Σ_t ∇_θ log π_θ(a_t | s_t) * G_t ]
```

**Intuition**:
- Increase probability of action `a_t` if it led to high return `G_t`
- Decrease probability if it led to low return
- The gradient ∇log π tells us which direction to move parameters θ

### REINFORCE Algorithm (Williams, 1992)

```
Initialize policy parameters θ randomly

for each episode:
    Generate episode τ = (s₀, a₀, r₁, ..., s_T) using π_θ

    for each timestep t = 0, 1, ..., T-1:
        Compute return: G_t = Σ_{k=t}^T γ^{k-t} * r_k

        Compute policy gradient:
            ∇_θ J ≈ ∇_θ log π_θ(a_t | s_t) * G_t

        Update parameters:
            θ ← θ + α * ∇_θ log π_θ(a_t | s_t) * G_t
```

### Return Calculation

The **return** G_t is the discounted sum of future rewards starting from time t:

```
G_t = r_{t+1} + γ * r_{t+2} + γ² * r_{t+3} + ...
    = r_{t+1} + γ * G_{t+1}
```

where γ ∈ [0, 1] is the discount factor.

**Example**:
- Rewards: [1, 2, 3, 4]
- Discount γ = 0.9
- Returns: [1 + 0.9*2 + 0.81*3 + 0.729*4 = 8.116, ...]

### Baseline (Variance Reduction)

Raw REINFORCE has **high variance**. We reduce it with a baseline b(s):

```
∇_θ J ≈ ∇_θ log π_θ(a_t | s_t) * (G_t - b(s_t))
```

**Best baseline**: State value function V(s_t)

**Why it works**:
- If G_t > V(s_t): This action was better than expected → increase probability
- If G_t < V(s_t): This action was worse than expected → decrease probability

**Important**: Baseline does NOT introduce bias (proof in Sutton & Barto, ch. 13)

---

## 🏗️ Architecture

### Policy Network

```
Input: State (6D)
  ↓
FC Layer (64 units) + ReLU
  ↓
FC Layer (64 units) + ReLU
  ↓
Split into two heads:
  ├─→ Mean Head → μ(s)        [action mean]
  └─→ Log Std Head → log σ(s) [action std]
  ↓
Sample action: a ~ N(μ(s), σ(s))
```

### Value Network (Baseline)

```
Input: State (6D)
  ↓
FC Layer (64 units) + ReLU
  ↓
FC Layer (64 units) + ReLU
  ↓
Value Head → V(s)  [scalar]
```

---

## 🔍 Key Concepts Explained

### 1. Why Policy Gradients?

**Value-based methods** (like Q-learning):
- Learn Q(s, a) or V(s)
- Derive policy implicitly: π(s) = argmax_a Q(s, a)
- ❌ Can only handle discrete actions
- ❌ Policy is deterministic (no exploration after convergence)

**Policy-based methods** (like REINFORCE):
- Learn π(a|s) directly
- ✅ Can handle continuous actions
- ✅ Natural exploration (stochastic policy)
- ✅ Can learn stochastic optimal policies

### 2. Stochastic vs Deterministic Policies

**Deterministic**: π(s) = a (always same action)

**Stochastic**: π(a|s) = P(action = a | state = s) (probability distribution)

For continuous actions, we use **Gaussian policy**:
```
π(a|s) = N(μ_θ(s), σ_θ(s))
```

### 3. On-Policy vs Off-Policy

**On-policy** (REINFORCE):
- Use data collected by current policy π
- Throw away data after each update
- ❌ Sample inefficient
- ✅ Stable, simple

**Off-policy** (DQN, DDPG):
- Use data collected by old policies
- Store in replay buffer
- ✅ Sample efficient
- ❌ More complex, can be unstable

### 4. Why Wait for Full Episode?

REINFORCE is a **Monte Carlo** method:
- Needs full episode to compute returns G_t
- Cannot update after each step
- Works well for episodic tasks
- Not suitable for continuous/infinite-horizon tasks

---

## 💻 Implementation Details

### Gaussian Policy

```python
# Forward pass
mean, std = policy_network(state)

# Sample action
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum()

# Loss (policy gradient)
policy_loss = -(log_prob * (return - baseline)).mean()
```

### Return Normalization

Normalize returns to reduce variance:

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

This helps because:
- Neural networks work better with normalized inputs
- Prevents exploding/vanishing gradients

### Gradient Clipping

Clip gradients to prevent updates that are too large:

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
```

---

## 📊 Hyperparameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `policy_lr` | 3e-4 | Policy learning rate (Adam) |
| `value_lr` | 1e-3 | Value learning rate (Adam) |
| `gamma` | 0.99 | Discount factor (future rewards) |
| `hidden_dim` | 64 | Neural network hidden layer size |

**Tuning Tips**:
- If training is unstable → Lower learning rates
- If convergence is slow → Increase learning rates (carefully)
- If agent is too short-sighted → Increase gamma
- If agent never terminates → Decrease gamma

---

## 🚀 Usage

### Training

```bash
# Basic training (500 episodes)
python train.py

# Custom parameters
python train.py --episodes 1000 --policy-lr 1e-4 --gamma 0.95

# With different random seed
python train.py --seed 123
```

### Evaluation

```bash
# Evaluate trained model
python test.py

# Compare with random baseline
python test.py --baseline

# Evaluate specific checkpoint
python test.py --checkpoint reinforce_epoch500.pt
```

---

## 📈 Expected Results

After 500 episodes of training:

- **Mean Reward**: -50 to -20 (higher is better)
- **Mean Tracking Distance**: 0.2 to 0.4 (lower is better)
- **Success Rate**: 60-80% (ball stays in frame)

**Comparison to Random**:
- Random policy: ~-150 reward, 0.8 distance
- REINFORCE: 3-5x improvement

**Learning Curve**:
- Episodes 0-100: High variance, slow improvement
- Episodes 100-300: Steady improvement
- Episodes 300-500: Convergence, low variance

---

## 🐛 Common Issues

### 1. Training is unstable (reward oscillates wildly)

**Causes**:
- Learning rate too high
- No baseline (high variance)
- Returns not normalized

**Solutions**:
```bash
python train.py --policy-lr 1e-4 --value-lr 5e-4
```

### 2. Agent doesn't learn anything

**Causes**:
- Learning rate too low
- Discount factor too low (agent too short-sighted)
- Bad initialization

**Solutions**:
- Increase learning rates
- Increase gamma to 0.99
- Try different random seeds

### 3. Training is very slow

**Causes**:
- Episodes are too long (300 steps)
- Policy gradient has high variance

**Solutions**:
- This is expected! REINFORCE is sample-inefficient
- Project 2 (DQN) and Project 4 (PPO) will be faster

---

## 🧠 Study Questions

Test your understanding:

1. **Why does REINFORCE multiply gradient by return G_t?**
   <details>
   <summary>Answer</summary>
   To weight the gradient by how good the outcome was. If G_t is high, we want to increase probability of that action. If G_t is low, decrease it.
   </details>

2. **Why use a baseline? Does it introduce bias?**
   <details>
   <summary>Answer</summary>
   Baseline reduces variance in gradient estimates, making training more stable. It does NOT introduce bias because E[∇log π(a|s) * b(s)] = 0.
   </details>

3. **Why can't REINFORCE use a replay buffer like DQN?**
   <details>
   <summary>Answer</summary>
   REINFORCE is on-policy - it must use data from current policy π_θ. Old data from π_θ_old is no longer valid after θ changes. Off-policy methods like DQN use importance sampling to correct for this.
   </details>

4. **What's the difference between return and reward?**
   <details>
   <summary>Answer</summary>
   - **Reward** r_t: Immediate signal at one timestep
   - **Return** G_t: Cumulative discounted sum of all future rewards from timestep t onward
   </details>

5. **Why use a stochastic policy instead of deterministic?**
   <details>
   <summary>Answer</summary>
   - Natural exploration (built into policy)
   - Can learn stochastic optimal policies (e.g., rock-paper-scissors)
   - Gradient estimation works better with continuous probability distributions
   </details>

---

## 📚 Further Reading

### Papers
- **REINFORCE**: Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning."
- **Policy Gradient Methods**: Sutton et al. (1999). "Policy gradient methods for reinforcement learning with function approximation."

### Textbooks
- Sutton & Barto (2018), Chapter 13: "Policy Gradient Methods"
- Goodfellow, Bengio & Courville (2016), Chapter 20.3: "Policy Gradient"

### Online Resources
- [Spinning Up in Deep RL - Policy Gradients](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
- [Lilian Weng's Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

---

## ✅ Checklist

Mark off as you complete:

- [ ] Read through all theory sections
- [ ] Understand policy gradient theorem
- [ ] Run training script and observe learning
- [ ] Evaluate trained agent vs random baseline
- [ ] Experiment with hyperparameters (learning rate, gamma)
- [ ] Answer all study questions
- [ ] Can explain REINFORCE to someone else

---

## 🎯 Next Steps

Once you've mastered REINFORCE, move on to:

**→ Project 2: DQN (Deep Q-Network)**
- Learn value-based methods
- Understand off-policy learning
- Use replay buffers for sample efficiency

This will give you a complete picture: policy-based (REINFORCE) vs value-based (DQN).
