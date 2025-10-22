# Phase 3 Implementation Plan
## Deep Reinforcement Learning with Human-in-the-Loop

**Version**: 1.0
**Created**: October 2025
**Status**: Planning Stage
**Prerequisites**: Phase 1 ✅ Complete | Phase 2 ✅ Complete

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [System Architecture](#system-architecture)
4. [Implementation Timeline](#implementation-timeline)
5. [Technical Components](#technical-components)
6. [Success Metrics](#success-metrics)
7. [Safety Considerations](#safety-considerations)
8. [Benchmark Tasks](#benchmark-tasks)
9. [Files to Create](#files-to-create)
10. [Research Questions](#research-questions)
11. [Next Steps](#next-steps)

---

## Overview

### Goal
Enable the Reachy Mini robot to continuously improve through human feedback and demonstrations using Deep Reinforcement Learning (DRL).

### Duration
**6 months** (Months 7-12 of project)

### Key Innovation
- Convert natural language feedback into reward signals using a fine-tuned reward model
- Combine human demonstrations with reinforcement learning for sample-efficient policy improvement
- Safe exploration with human oversight and emergency stop capabilities

### Current Foundation
- ✅ **Phase 1**: Multi-agent system (Coordinator, Robot, Vision agents)
- ✅ **Phase 2**: Self-coding agent with tool generation and validation
- 🎯 **Phase 3**: Add continuous learning and improvement

---

## Objectives

### 1. Natural Language → Reward Conversion
**Goal**: Train a reward model to convert human feedback to scalar rewards

**Target Metrics**:
- Reward prediction accuracy: >85% (compared to human ground-truth ratings)
- Response time: <100ms per prediction
- Feedback types supported: Text, voice, quick buttons (good/bad/stop)

**Approach**:
- Fine-tune Gemma 4B model as reward predictor
- Collect 500+ diverse feedback examples
- Encode: (state, action, outcome, feedback) → scalar reward [-1, 1]

### 2. Human Demonstrations
**Goal**: Collect and utilize expert demonstrations for policy initialization

**Target Metrics**:
- Demonstration collection: 100+ examples across 15 tasks
- Behavioral cloning baseline: >60% success rate
- Imitation learning improvement: +15% over random policy

**Approach**:
- Build demonstration recording interface
- Store state-action trajectories
- Use for policy pre-training (behavioral cloning)
- Implement DAgger (Dataset Aggregation) for iterative improvement

### 3. Policy Training
**Goal**: Train policy using PPO to maximize cumulative reward

**Target Metrics**:
- Success rate improvement: +20% over Phase 2 baseline
- Sample efficiency: <1000 episodes to convergence
- Training stability: <10% performance variance

**Approach**:
- Implement Proximal Policy Optimization (PPO)
- Combine task rewards + human feedback rewards + safety penalties
- Use Generalized Advantage Estimation (GAE)
- Train in simulation first, then sim-to-real transfer

### 4. Safety & Evaluation
**Goal**: Safe exploration and comprehensive benchmarking

**Target Metrics**:
- Safety violations: <1% of actions
- Generalization: >70% success on novel tasks
- Human satisfaction: >4.0/5.0 rating

**Approach**:
- Hard constraints on dangerous actions
- Out-of-distribution detection
- Human emergency stop always available
- Evaluate on 15 benchmark tasks

---

## System Architecture

### New Components for Phase 3

```
┌─────────────────────────────────────────────────────────────────┐
│                    Phase 3: DRL Training Layer                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Reward Model (Gemma 4B)                     │  │
│  │                                                            │  │
│  │  Input: (robot_state, action, outcome, feedback_text)    │  │
│  │  Output: Scalar reward ∈ [-1, 1]                         │  │
│  │                                                            │  │
│  │  Training: Fine-tuned on (feedback, reward) pairs        │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │              Policy Network (PPO Actor)                   │  │
│  │                                                            │  │
│  │  Input: State (robot + scene + task)                     │  │
│  │  Output: Action distribution π(a|s)                       │  │
│  │                                                            │  │
│  │  Architecture: 3-layer MLP (256 hidden units)            │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │              Value Network (PPO Critic)                   │  │
│  │                                                            │  │
│  │  Input: State                                             │  │
│  │  Output: State value V(s)                                 │  │
│  │                                                            │  │
│  │  Architecture: 3-layer MLP (256 hidden units)            │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │              Experience Replay Buffer                     │  │
│  │                                                            │  │
│  │  Storage: (s_t, a_t, r_t, s_{t+1}, done)                │  │
│  │  Capacity: 10,000 transitions                             │  │
│  │  Sampling: Episode-based batches                          │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                            │                                    │
│  ┌─────────────────────────▼────────────────────────────────┐  │
│  │           Demonstration Database (PostgreSQL)             │  │
│  │                                                            │  │
│  │  • Expert demonstrations (state-action trajectories)      │  │
│  │  • Task descriptions and success criteria                 │  │
│  │  • Metadata: timestamp, annotator, quality rating         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │        Human Feedback Interface (Web UI)                  │  │
│  │                                                            │  │
│  │  • Real-time feedback during episodes                     │  │
│  │  • Quick buttons: Good / Bad / Stop                       │  │
│  │  • Natural language feedback box                          │  │
│  │  • Episode replay and annotation                          │  │
│  │  • Emergency stop (instant policy halt)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Loop Flow

```
┌──────────────────────────────────────────────────────────────┐
│                   DRL Training Loop                           │
└──────────────────────────────────────────────────────────────┘

Episode Start
    │
    ▼
┌─────────────────────────┐
│ 1. Agent Interaction    │
│                         │
│   • Execute policy:     │
│     π_θ(a|s)           │
│   • Collect trajectory: │
│     τ = (s₀,a₀,...,sₙ) │
│   • Store in buffer     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 2. Human Feedback       │
│    (async, optional)    │
│                         │
│   • Human observes      │
│   • Provides feedback:  │
│     - Text              │
│     - Rating            │
│     - Emergency stop    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 3. Reward Assignment    │
│                         │
│   r_total = α·r_task +  │
│             β·r_human + │
│             γ·r_safety  │
│                         │
│   where:                │
│   • r_task: Task metric │
│   • r_human: Reward     │
│     model prediction    │
│   • r_safety: Penalties │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 4. Policy Update (PPO)  │
│    (every 10 episodes)  │
│                         │
│   • Compute advantages: │
│     Â = Q(s,a) - V(s)  │
│   • Update policy:      │
│     θ ← θ + ∇L_CLIP    │
│   • Update value net:   │
│     φ ← φ + ∇L_value   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 5. Evaluation           │
│    (every 50 episodes)  │
│                         │
│   • Test on held-out    │
│     tasks               │
│   • Measure success     │
│     rate, efficiency    │
│   • Checkpoint if       │
│     improved            │
└───────────┬─────────────┘
            │
            ▼
    Next Episode
```

---

## Implementation Timeline

### Month 7-8: Reward Model & Feedback System

#### **Week 1-2: Feedback Collection Infrastructure**

**Tasks**:
- [ ] Design feedback database schema (PostgreSQL)
  - Tables: episodes, feedback, ratings, demonstrations
  - Indexes: episode_id, timestamp, user_id
- [ ] Implement FastAPI feedback endpoints
  - POST `/api/v1/feedback` - Submit feedback
  - POST `/api/v1/feedback/quick` - Quick rating buttons
  - GET `/api/v1/episodes/{id}` - Get episode details
- [ ] Create feedback web UI (React/Next.js)
  - Real-time episode view
  - Quick feedback buttons (👍 👎 🛑)
  - Text feedback box
  - Episode replay with video
- [ ] Add episode tracking to agent system
  - Capture: state, action, outcome at each step
  - Store: robot state, camera frames, audio
- [ ] Test feedback collection pipeline end-to-end

**Deliverables**:
- Working feedback database
- Functional web UI for feedback
- Episode recording system integrated

#### **Week 3-4: Reward Model Training**

**Tasks**:
- [ ] Collect initial feedback dataset
  - Run 100 episodes with human observation
  - Collect diverse feedback examples
  - Include: positive, negative, nuanced feedback
  - Label with ground-truth rewards [-1, 1]
- [ ] Prepare dataset for training
  - Create (state, action, outcome, feedback) tuples
  - Split: 80% train / 10% val / 10% test
  - Data augmentation: paraphrase feedback
- [ ] Fine-tune Gemma 4B as reward model
  - Base model: google/gemma-4b
  - Task: Regression (predict scalar reward)
  - Loss: MSE between predicted and ground-truth
  - Training: 10 epochs, lr=1e-5, batch_size=8
- [ ] Implement reward prediction API
  - POST `/api/v1/reward/predict` - Get reward
  - Inference time: <100ms per prediction
- [ ] Validate reward model accuracy
  - Correlation with human ratings: target >0.85
  - Test on unseen feedback examples

**Deliverables**:
- Fine-tuned reward model (>85% accuracy)
- Reward prediction API
- Evaluation report

#### **Week 5-6: Feedback Integration**

**Tasks**:
- [ ] Integrate reward model with training loop
  - Load model at training start
  - Call during trajectory processing
  - Cache predictions for efficiency
- [ ] Add real-time feedback processing
  - WebSocket connection for live feedback
  - Asynchronous feedback queue
  - Timeout handling (default: 30s)
- [ ] Implement feedback → reward conversion
  - Encode current episode context
  - Call reward model
  - Combine with task reward
  - Apply to trajectory
- [ ] Test end-to-end feedback loop
  - Run episode
  - Submit feedback
  - Verify reward assignment
  - Check policy update
- [ ] Create feedback dashboard
  - Visualize feedback over time
  - Show reward predictions
  - Track model performance

**Deliverables**:
- Integrated feedback system
- Real-time feedback processing
- Monitoring dashboard

#### **Week 7-8: Testing & Refinement**

**Tasks**:
- [ ] Collect more diverse feedback (500+ examples)
  - Multiple annotators
  - Diverse tasks and scenarios
  - Edge cases and ambiguous feedback
- [ ] Retrain reward model with expanded dataset
  - Improved accuracy target: >90%
  - Better generalization
- [ ] Evaluate on held-out test set
  - Measure: accuracy, F1, correlation
  - Compare to baseline (random, rule-based)
- [ ] Tune hyperparameters
  - Model size: 4B vs 7B
  - Training epochs
  - Feedback encoding strategy
- [ ] Performance analysis
  - Inference latency profiling
  - Memory usage optimization
  - Batch prediction for efficiency

**Deliverables**:
- Refined reward model (>90% accuracy)
- Performance benchmarks
- Documentation and usage guide

---

### Month 9-10: Policy Training System

#### **Week 1-2: Environment Setup**

**Tasks**:
- [ ] Create RL environment wrapper for ReachyMini
  - Inherit from gym.Env interface
  - Implement reset(), step(), render()
- [ ] Define state space (observation)
  - Robot state: joint positions, velocities
  - Visual: camera image (encoded)
  - Task: goal description (encoded)
  - Dimension: ~512 (concatenated)
- [ ] Define action space
  - Discrete: tool selection (20 tools)
  - Continuous: tool parameters
  - Action encoding strategy
- [ ] Implement environment reset/step functions
  - Reset: Random initial state + task
  - Step: Execute action, get reward, next state
  - Done: Task success/failure/timeout
- [ ] Test environment in MuJoCo simulation
  - Verify state transitions
  - Check reward signals
  - Test edge cases

**Deliverables**:
- Working RL environment wrapper
- State/action space definitions
- Environment test suite

#### **Week 3-4: PPO Implementation**

**Tasks**:
- [ ] Implement policy network (actor)
  - Architecture: 3-layer MLP [512 → 256 → 256 → action_dim]
  - Output: Action probabilities (softmax)
  - Initialization: Xavier uniform
- [ ] Implement value network (critic)
  - Architecture: 3-layer MLP [512 → 256 → 256 → 1]
  - Output: State value estimate
  - Shared feature extractor option
- [ ] Create PPO trainer with GAE
  - Generalized Advantage Estimation (λ=0.95)
  - Clipped objective (ε=0.2)
  - Entropy bonus (β=0.01)
- [ ] Add experience buffer
  - Storage: Episode-based (not single transitions)
  - Capacity: 10,000 transitions
  - Sampling: Batch by episode
- [ ] Test PPO update steps
  - Verify policy improvement
  - Check gradient flow
  - Monitor KL divergence

**Deliverables**:
- PPO policy network
- PPO value network
- PPO trainer with GAE
- Experience replay buffer

#### **Week 5-6: Training Pipeline**

**Tasks**:
- [ ] Implement main training loop
  - Collect trajectories
  - Wait for feedback (async)
  - Assign rewards
  - Update policy (every 10 episodes)
  - Evaluate (every 50 episodes)
- [ ] Add checkpoint management
  - Save: policy, value net, optimizer state
  - Load: Resume training from checkpoint
  - Best model tracking
- [ ] Create evaluation metrics
  - Success rate
  - Average reward per episode
  - Episode length
  - Sample efficiency (episodes to convergence)
- [ ] Integrate with reward model
  - Call reward model for feedback
  - Combine rewards: task + human + safety
  - Weight tuning: α, β, γ
- [ ] Test on simple tasks
  - Task 1: Move head to look at object
  - Task 2: Nod yes when asked
  - Task 3: Shake head no when asked
  - Baseline: Phase 2 agent performance

**Deliverables**:
- Complete training pipeline
- Checkpoint system
- Evaluation framework
- Simple task results

#### **Week 7-8: Demonstration System**

**Tasks**:
- [ ] Build demonstration recording interface
  - Web UI for human control
  - Joystick / keyboard input
  - Record: states, actions, outcomes
- [ ] Implement behavioral cloning baseline
  - Dataset: collected demonstrations
  - Model: Policy network
  - Loss: Cross-entropy (discrete) + MSE (continuous)
  - Training: Supervised learning
- [ ] Create demonstration database
  - Schema: demo_id, task, states, actions, metadata
  - Storage: PostgreSQL + file storage for states
- [ ] Test imitation learning
  - Measure: success rate on demonstrated tasks
  - Target: >60% of human performance
- [ ] Combine demonstrations + RL
  - Pre-train with BC
  - Fine-tune with PPO
  - Compare: BC only vs BC+PPO vs PPO only

**Deliverables**:
- Demonstration recording system
- Behavioral cloning baseline
- Demonstration database
- BC + RL integration

---

### Month 11: Integration & Advanced Training

#### **Week 1-2: Full System Integration**

**Tasks**:
- [ ] Connect all Phase 3 components
  - Reward model → Training loop
  - Feedback interface → Reward assignment
  - Demonstration DB → Policy pre-training
  - Evaluation → Checkpointing
- [ ] Integrate with Phase 1 + Phase 2 agents
  - Use multi-agent system for complex tasks
  - Generate new tools when needed (Phase 2)
  - Train policy over tool usage
- [ ] Test end-to-end DRL loop
  - Full episode: reset → interact → feedback → reward → update
  - Verify all components working together
- [ ] Add comprehensive monitoring and logging
  - Training metrics: loss, reward, success rate
  - System metrics: latency, memory, GPU usage
  - Dashboard: Real-time visualization
- [ ] Performance profiling and optimization
  - Identify bottlenecks
  - Optimize: data loading, model inference, updates
  - Target: <5 min per training iteration

**Deliverables**:
- Fully integrated DRL system
- Monitoring dashboard
- Performance benchmarks
- Integration test suite

#### **Week 3-4: Advanced Training Features**

**Tasks**:
- [ ] Implement curriculum learning
  - Start with simple tasks
  - Gradually increase difficulty
  - Automatic task progression based on success rate
- [ ] Add safety constraints
  - Hard limits on dangerous actions
  - Safety layer over policy (constrained optimization)
  - Safety violation detection and penalties
- [ ] Create diverse task distribution
  - 15 benchmark tasks
  - Task variants for generalization testing
  - Task difficulty ratings
- [ ] Implement continual learning
  - Train on new tasks without forgetting old ones
  - Elastic Weight Consolidation (EWC)
  - Replay buffer management
- [ ] Meta-learning experiments (optional)
  - Model-Agnostic Meta-Learning (MAML)
  - Fast adaptation to new tasks
  - Few-shot learning evaluation

**Deliverables**:
- Curriculum learning system
- Safety-constrained training
- Task distribution with 15 benchmarks
- Continual learning implementation

---

### Month 12: Evaluation & Publication

#### **Week 1-2: Benchmark Evaluation**

**Tasks**:
- [ ] Run full benchmark suite (15 tasks)
  - Phase 1 tasks (5): Visual perception and interaction
  - Phase 2 tasks (5): Tool generation and management
  - Phase 3 tasks (5): Learning and adaptation
- [ ] Measure success rate improvement
  - Baseline: Phase 2 agent (before DRL)
  - After training: Phase 3 agent (with DRL)
  - Target: +20% improvement
- [ ] Calculate sample efficiency
  - Episodes to 90% of max performance
  - Target: <1000 episodes
  - Compare: PPO vs BC+PPO vs SAC
- [ ] Test generalization to novel tasks
  - Task variants not in training set
  - Target: >70% success rate
  - Measure: Transfer learning capability
- [ ] Collect human satisfaction ratings
  - User study: 10+ participants
  - Tasks: Interact with Phase 2 vs Phase 3 agent
  - Rating: 1-5 scale for task quality
  - Target: >4.0/5.0 for Phase 3

**Deliverables**:
- Benchmark results report
- Success rate improvement analysis
- Sample efficiency comparison
- Generalization study
- Human satisfaction survey results

#### **Week 3: Analysis & Documentation**

**Tasks**:
- [ ] Statistical analysis of results
  - T-tests for significance
  - Confidence intervals
  - Ablation studies: What components matter most?
- [ ] Create visualization dashboards
  - Learning curves
  - Success rate over time
  - Reward distribution
  - State-action heatmaps
- [ ] Write comprehensive documentation
  - Architecture overview
  - API documentation
  - Training guide
  - Deployment instructions
- [ ] Prepare demo videos
  - Before/after training comparison
  - Real robot demonstrations
  - Human-in-the-loop interaction
- [ ] Create tutorial notebooks
  - Reward model training
  - Policy training walkthrough
  - Evaluation and analysis

**Deliverables**:
- Statistical analysis report
- Visualization dashboard
- Complete documentation (50+ pages)
- Demo videos (3-5 minutes)
- Tutorial Jupyter notebooks

#### **Week 4: Research Publication**

**Tasks**:
- [ ] Write research paper
  - Title: "Human-in-the-Loop Deep Reinforcement Learning for Embodied Robotics"
  - Abstract: 250 words
  - Sections: Intro, Related Work, Method, Experiments, Results, Discussion
  - Length: 8-10 pages (conference format)
- [ ] Prepare conference submission
  - Target: ICRA, IROS, CoRL, or NeurIPS
  - Format: IEEE or ACM template
  - Supplementary materials: videos, code
- [ ] Create project website
  - URL: reachy-drl.ai (example)
  - Sections: Demo, Paper, Code, Results
  - Interactive: Try the reward model online
- [ ] Public demo preparation
  - Live demonstration setup
  - Q&A materials
  - Presentation slides (20 slides)
- [ ] Open-source release
  - Clean up code
  - Add documentation
  - Create Docker image
  - GitHub release with DOI

**Deliverables**:
- Research paper (ready for submission)
- Project website (live)
- Public demo materials
- Open-source release (GitHub)

---

## Technical Components

### 1. Reward Model

**File**: `src/drl/reward_model.py`

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn

class RewardModel:
    """
    Convert natural language feedback to scalar rewards.

    Architecture: Fine-tuned Gemma 4B with regression head
    Input: (state, action, outcome, feedback) encoded as text
    Output: Scalar reward in [-1, 1]
    """

    def __init__(self, model_name: str = "google/gemma-4b"):
        """Initialize reward model."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Regression task
            torch_dtype=torch.float16  # Memory optimization
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode_interaction(
        self,
        state: dict,
        action: dict,
        outcome: dict,
        feedback: str
    ) -> str:
        """
        Encode interaction for reward model.

        Args:
            state: Current state (robot + scene + task)
            action: Action taken (tool + params)
            outcome: Result of action
            feedback: Natural language feedback

        Returns:
            Formatted prompt string
        """
        prompt = f"""
Task: {state.get('task_description', 'Unknown')}
Robot State: Joints={state.get('joint_positions', [])}, Head Pose={state.get('head_pose', [])}
Scene: {state.get('scene_description', 'No description')}

Action: {action.get('tool', 'unknown')}({action.get('params', {})})
Outcome: Status={outcome.get('status', 'unknown')}, Result={outcome.get('result', '')}
Execution Time: {outcome.get('time', 0):.2f}s

Human Feedback: "{feedback}"

Rate the agent's performance from -1 (very bad) to +1 (excellent):
"""
        return prompt.strip()

    def predict_reward(
        self,
        state: dict,
        action: dict,
        outcome: dict,
        feedback: str
    ) -> float:
        """
        Predict reward from feedback.

        Args:
            state: Current state
            action: Action taken
            outcome: Result
            feedback: Natural language feedback

        Returns:
            Reward in [-1, 1]
        """
        # Encode interaction
        prompt = self.encode_interaction(state, action, outcome, feedback)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            reward = torch.tanh(outputs.logits[0, 0])  # Squash to [-1, 1]

        return reward.item()

    def train_from_examples(
        self,
        examples: list[dict],
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-5
    ) -> dict:
        """
        Fine-tune reward model from human feedback examples.

        Args:
            examples: List of (state, action, outcome, feedback, reward) dicts
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            Training metrics
        """
        from torch.utils.data import DataLoader, Dataset

        # Custom dataset
        class RewardDataset(Dataset):
            def __init__(self, examples, tokenizer, reward_model):
                self.examples = examples
                self.tokenizer = tokenizer
                self.reward_model = reward_model

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                ex = self.examples[idx]
                prompt = self.reward_model.encode_interaction(
                    ex['state'], ex['action'], ex['outcome'], ex['feedback']
                )
                encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=512,
                    padding='max_length',
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': torch.tensor([ex['reward']], dtype=torch.float32)
                }

        # Create dataset and dataloader
        dataset = RewardDataset(examples, self.tokenizer, self)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # Training loop
        self.model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = nn.functional.mse_loss(outputs.logits, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

        return {'losses': losses, 'final_loss': losses[-1]}
```

### 2. Policy Network (Actor)

**File**: `src/drl/policy_network.py`

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Policy network (Actor) for action selection.

    Architecture: 3-layer MLP
    Input: State vector (concatenated robot + scene + task)
    Output: Action probability distribution
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize policy network.

        Args:
            state_dim: Dimension of state vector
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Action probabilities [batch_size, action_dim]
        """
        return self.network(state)

    def select_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Select action from current policy.

        Args:
            state: State tensor [state_dim] or [batch_size, state_dim]
            deterministic: If True, select argmax; else sample

        Returns:
            action: Selected action index
            log_prob: Log probability of action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        action_probs = self.forward(state)
        dist = Categorical(action_probs)

        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()
```

### 3. Value Network (Critic)

**File**: `src/drl/value_network.py`

```python
import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """
    Value network (Critic) for state value estimation.

    Architecture: 3-layer MLP
    Input: State vector
    Output: Scalar state value V(s)
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        Initialize value network.

        Args:
            state_dim: Dimension of state vector
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            State values [batch_size, 1]
        """
        return self.network(state)
```

### 4. PPO Trainer

**File**: `src/drl/ppo_trainer.py`

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from collections import deque

class PPOTrainer:
    """
    Proximal Policy Optimization trainer.

    Implements:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function learning
    """

    def __init__(
        self,
        policy_net,
        value_net,
        reward_model,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_net: Policy network (actor)
            value_net: Value network (critic)
            reward_model: Reward model for feedback
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clip parameter
            gae_lambda: GAE lambda
            entropy_coef: Entropy bonus coefficient
        """
        self.policy_net = policy_net
        self.value_net = value_net
        self.reward_model = reward_model

        self.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.experience_buffer = deque(maxlen=10000)

    def compute_gae(
        self,
        rewards: list,
        values: list,
        dones: list
    ) -> tuple:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        # Returns = advantages + values
        returns = [adv + val for adv, val in zip(advantages, values)]

        return advantages, returns

    def update_policy(
        self,
        trajectories: list,
        epochs: int = 10,
        batch_size: int = 64
    ) -> dict:
        """
        Update policy using PPO.

        Args:
            trajectories: List of collected trajectories
            epochs: Number of update epochs
            batch_size: Minibatch size

        Returns:
            Training metrics
        """
        # Prepare data
        states = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []

        for traj in trajectories:
            # Compute values
            traj_states = [torch.FloatTensor(step['state']) for step in traj]
            traj_values = [self.value_net(s).item() for s in traj_states]

            # Compute advantages
            traj_rewards = [step['reward'] for step in traj]
            traj_dones = [step['done'] for step in traj]
            traj_advantages, traj_returns = self.compute_gae(
                traj_rewards, traj_values, traj_dones
            )

            # Collect
            for i, step in enumerate(traj):
                states.append(step['state'])
                actions.append(step['action'])
                old_log_probs.append(step['log_prob'])
                advantages.append(traj_advantages[i])
                returns.append(traj_returns[i])

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        policy_losses = []
        value_losses = []

        for epoch in range(epochs):
            # Minibatch updates
            indices = np.random.permutation(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Current policy
                action_probs = self.policy_net(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

                # Value function
                values = self.value_net(batch_states).squeeze()
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
        }
```

---

## Success Metrics

### Reward Model Performance

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| Prediction Accuracy | 50% (random) | >85% | Correlation with ground-truth ratings |
| MSE Loss | N/A | <0.1 | Mean squared error on test set |
| Inference Latency | N/A | <100ms | Time per prediction |
| Generalization | N/A | >80% | Accuracy on unseen feedback styles |

### Policy Improvement

| Metric | Phase 2 Baseline | Phase 3 Target | Improvement |
|--------|------------------|----------------|-------------|
| Success Rate | 70% | 84%+ | +20% |
| Average Reward | 0.5 | 0.7+ | +40% |
| Episode Length | 50 steps | 35 steps | -30% (efficiency) |
| Human Satisfaction | 3.5/5.0 | 4.0+/5.0 | +14% |

### Sample Efficiency

| Metric | Standard PPO | Phase 3 Target | Method |
|--------|-------------|----------------|--------|
| Episodes to 90% | ~5000 | <1000 | BC + PPO + Human Feedback |
| Wall-clock Time | ~50 hours | <10 hours | Efficient reward model |
| Human Feedback Hours | N/A | <20 hours | Asynchronous collection |

### Generalization

| Task Type | Training Performance | Test Performance | Generalization Gap |
|-----------|---------------------|------------------|-------------------|
| Seen Tasks | 90% | 85% | 5% (excellent) |
| Task Variants | 90% | 70%+ | 20% (good) |
| Novel Tasks | 90% | 70%+ | 20% (target) |

### Safety

| Metric | Target | Consequence if Violated |
|--------|--------|------------------------|
| Safety Violations | <1% | Immediate policy rollback |
| OOD Detection | >95% | Flag for human review |
| Emergency Stops | 100% responsive | System shutdown |

---

## Safety Considerations

### Design Principles

1. **Human Always in Control**
   - Emergency stop button always active
   - Human can override any action
   - Policy suggestions, not commands

2. **Conservative Exploration**
   - Hard constraints on dangerous actions
   - Safety layer over policy output
   - Limited exploration early in training

3. **Monitoring & Alerting**
   - Real-time safety violation detection
   - Automatic policy rollback on failures
   - Human notification system

4. **Simulation First**
   - Train in MuJoCo simulation
   - Validate safety before real robot
   - Sim-to-real transfer with caution

5. **Incremental Deployment**
   - Start with simple, safe tasks
   - Gradually increase complexity
   - Continuous evaluation

### Safety Mechanisms

#### 1. Action Constraints

```python
class SafetyLayer:
    """Enforce hard constraints on actions."""

    SAFE_RANGES = {
        'head_pitch': (-45, 45),  # degrees
        'head_yaw': (-45, 45),
        'head_roll': (-30, 30),
        'movement_speed': (0, 1.0),  # normalized
    }

    def validate_action(self, action: dict) -> bool:
        """Check if action is safe."""
        for param, value in action.items():
            if param in self.SAFE_RANGES:
                min_val, max_val = self.SAFE_RANGES[param]
                if not (min_val <= value <= max_val):
                    return False
        return True

    def clip_action(self, action: dict) -> dict:
        """Clip action to safe range."""
        safe_action = {}
        for param, value in action.items():
            if param in self.SAFE_RANGES:
                min_val, max_val = self.SAFE_RANGES[param]
                safe_action[param] = np.clip(value, min_val, max_val)
            else:
                safe_action[param] = value
        return safe_action
```

#### 2. Out-of-Distribution Detection

```python
class OODDetector:
    """Detect out-of-distribution states."""

    def __init__(self, training_states: list):
        """Initialize with training data statistics."""
        self.mean = np.mean(training_states, axis=0)
        self.std = np.std(training_states, axis=0)
        self.threshold = 3.0  # Standard deviations

    def is_ood(self, state: np.ndarray) -> bool:
        """Check if state is out-of-distribution."""
        z_scores = np.abs((state - self.mean) / (self.std + 1e-8))
        return np.any(z_scores > self.threshold)
```

#### 3. Emergency Stop

```python
@app.post("/emergency_stop")
async def emergency_stop():
    """Immediately halt all robot actions."""
    global policy_execution_enabled
    policy_execution_enabled = False

    # Stop robot
    robot.stop()

    # Log incident
    logger.critical("Emergency stop triggered!")

    return {"status": "stopped", "timestamp": time.time()}
```

---

## Benchmark Tasks

### Phase 1 Tasks (Visual Perception - 5 tasks)

1. **Look and Describe**
   - Task: Look at object and verbally describe it
   - Success: Correct object identification + description
   - Difficulty: Easy

2. **Find Object**
   - Task: Scan environment to find specific object
   - Success: Object located and centered in view
   - Difficulty: Medium

3. **Visual Q&A**
   - Task: Answer questions about scene ("How many objects?")
   - Success: Correct answer
   - Difficulty: Medium

4. **Multi-step Command**
   - Task: "Look left, take photo, describe what you see"
   - Success: All steps completed correctly
   - Difficulty: Medium-Hard

5. **Handle Interruptions**
   - Task: Execute command, handle mid-execution correction
   - Success: Graceful handling of interruption
   - Difficulty: Hard

### Phase 2 Tasks (Tool Generation - 5 tasks)

6. **Generate New Tool**
   - Task: Create tool for novel capability (e.g., "tilt head diagonally")
   - Success: Tool generated, validated, and working
   - Difficulty: Medium

7. **Fix Failing Tool**
   - Task: Debug and fix tool based on error message
   - Success: Tool repaired and functional
   - Difficulty: Hard

8. **Tool Composition**
   - Task: Combine multiple tools for complex behavior
   - Success: Successful multi-tool execution
   - Difficulty: Hard

9. **Version Management**
   - Task: Update tool, rollback if worse
   - Success: Correct version control usage
   - Difficulty: Medium

10. **Safety Rejection**
    - Task: Refuse dangerous tool generation request
    - Success: Correct rejection with explanation
    - Difficulty: Medium

### Phase 3 Tasks (Learning & Adaptation - 5 tasks)

11. **Improve from Feedback**
    - Task: Adjust movement based on "too fast" / "too slow" feedback
    - Success: Observable improvement over 10 attempts
    - Difficulty: Medium
    - Metric: Reduce error by 50%

12. **Learn from Demonstration**
    - Task: Watch human demonstrate smooth head tracking, then replicate
    - Success: Match human trajectory within 10%
    - Difficulty: Hard
    - Metric: Trajectory similarity >90%

13. **Transfer Knowledge**
    - Task: Apply learned pick-and-place to new object types
    - Success: Generalization to 3+ new objects
    - Difficulty: Hard
    - Metric: Success rate >70% on novel objects

14. **Handle Distribution Shift**
    - Task: Adapt to new lighting conditions (trained on bright, test on dim)
    - Success: Maintain >80% success rate
    - Difficulty: Hard
    - Metric: Performance degradation <20%

15. **Safe Exploration**
    - Task: Learn new behavior while respecting safety constraints
    - Success: Zero safety violations during training
    - Difficulty: Very Hard
    - Metric: 100% safety compliance + task success

---

## Files to Create

### Core DRL Components (9 files)

```
src/drl/
├── __init__.py                    # Module exports
├── reward_model.py                # NL feedback → scalar reward (300 lines)
├── policy_network.py              # Actor network (150 lines)
├── value_network.py               # Critic network (100 lines)
├── ppo_trainer.py                 # PPO implementation (400 lines)
├── environment.py                 # RL env wrapper (300 lines)
├── experience_buffer.py           # Replay buffer (150 lines)
├── demonstration_db.py            # Demo storage (200 lines)
└── evaluation.py                  # Metrics & benchmarking (250 lines)
```

**Total**: ~1850 lines

### API Endpoints (5 files)

```
src/api/routes/
├── feedback.py                    # Feedback endpoints (150 lines)
├── demonstration.py               # Demo recording (150 lines)
├── training.py                    # Training control (200 lines)
├── evaluation.py                  # Eval results (100 lines)
└── rl_websocket.py               # Real-time feedback WS (150 lines)
```

**Total**: ~750 lines

### Database Schema (4 files)

```
src/database/
├── feedback.py                    # Feedback storage (150 lines)
├── demonstrations.py              # Demo storage (150 lines)
├── episodes.py                    # Episode tracking (150 lines)
└── metrics.py                     # Training metrics (100 lines)
```

**Total**: ~550 lines

### Training Scripts (5 files)

```
scripts/
├── train_reward_model.py          # Reward model training (300 lines)
├── train_policy.py                # Policy training loop (400 lines)
├── collect_demonstrations.py      # Demo collection (200 lines)
├── evaluate_policy.py             # Benchmark eval (300 lines)
└── analyze_results.py             # Results analysis (250 lines)
```

**Total**: ~1450 lines

### Configuration Files (3 files)

```
configs/
├── reward_model.yaml              # Reward model config
├── ppo_config.yaml                # PPO hyperparameters
└── evaluation.yaml                # Evaluation settings
```

### Web UI (Frontend - separate repo or directory)

```
web_ui/
├── components/
│   ├── FeedbackPanel.tsx          # Feedback interface
│   ├── EpisodeViewer.tsx          # Episode replay
│   ├── TrainingDashboard.tsx      # Metrics visualization
│   └── DemoRecorder.tsx           # Demo recording UI
├── pages/
│   ├── index.tsx                  # Home page
│   ├── feedback.tsx               # Feedback page
│   ├── training.tsx               # Training page
│   └── evaluation.tsx             # Evaluation page
```

### Documentation (5 files)

```
docs/
├── PHASE3_ARCHITECTURE.md         # Technical architecture
├── PHASE3_API.md                  # API documentation
├── PHASE3_TRAINING_GUIDE.md       # How to train
├── PHASE3_EVALUATION.md           # Evaluation guide
└── PHASE3_TROUBLESHOOTING.md      # Common issues
```

### Test Suite (4 files)

```
tests/test_drl/
├── test_reward_model.py           # Reward model tests
├── test_ppo_trainer.py            # PPO tests
├── test_environment.py            # Environment tests
└── test_integration.py            # End-to-end tests
```

---

## Research Questions

### 1. Can NL feedback effectively train reward models?

**Hypothesis**: Natural language feedback contains sufficient signal to learn accurate reward functions for robotic tasks.

**Experiment**:
- Collect 500+ (feedback, ground-truth reward) pairs
- Train reward model on 80% of data
- Evaluate correlation on held-out 20%
- Compare: NL-only vs NL+explicit rating vs explicit rating only

**Success Criterion**: NL-only achieves >85% correlation with ground-truth

### 2. What's the optimal combination of feedback + demos?

**Hypothesis**: Combining behavioral cloning (from demos) with RL (from feedback) achieves best sample efficiency.

**Experiment**:
- Baseline 1: Pure PPO (no BC, no demos)
- Baseline 2: BC-only (demos, no RL)
- Approach 1: BC pre-training + PPO fine-tuning
- Approach 2: Mixed training (BC loss + PPO loss simultaneously)

**Success Criterion**: BC+PPO reaches 90% max performance in <1000 episodes (vs ~5000 for pure PPO)

### 3. How does the agent generalize?

**Hypothesis**: Agent trained with diverse feedback + demos can generalize to task variants and novel tasks.

**Experiment**:
- Train on 10 tasks
- Test on:
  - Same tasks (memorization)
  - Task variants (e.g., different object sizes)
  - Novel tasks (not in training set)
- Measure: Success rate, transfer learning effectiveness

**Success Criterion**: >70% success on novel tasks

### 4. What are the safety implications?

**Hypothesis**: DRL with human feedback maintains safety during exploration.

**Experiment**:
- Define safety constraints (joint limits, speed limits)
- Track violations during training
- Compare: PPO vs constrained PPO vs human oversight

**Success Criterion**: <1% safety violations with human oversight

### 5. Is real-time human feedback practical?

**Hypothesis**: Asynchronous human feedback is practical and effective for robotic DRL.

**Experiment**:
- Measure: Human time per episode
- Survey: User fatigue, satisfaction
- Compare: Real-time vs post-hoc feedback

**Success Criterion**: <20 hours of human time for full training, >4.0/5.0 satisfaction

---

## Next Steps

### Immediate (Week 1)

**Priority 1: Decide Starting Point**

Choose one of these three options:

1. **Option A: Reward Model First (Recommended)**
   - Pros: Foundation for everything, can collect data immediately
   - Cons: Requires human feedback collection infrastructure
   - Time: 2 weeks for basic version

2. **Option B: Feedback UI First**
   - Pros: Enables data collection, user-friendly
   - Cons: Doesn't directly advance DRL capabilities
   - Time: 1 week for MVP

3. **Option C: Environment Wrapper First**
   - Pros: Enables early RL experiments
   - Cons: Can't test with real rewards yet
   - Time: 1 week for basic version

**Recommendation**: Start with **Option A (Reward Model)** because:
- It's the most critical component
- Can begin with simple text feedback (no UI needed)
- Provides immediate value for evaluating Phase 2 agent

**Priority 2: Set Up Infrastructure**

- [ ] Install PyTorch + Transformers library
- [ ] Download Gemma 4B model
- [ ] Create `src/drl/` directory structure
- [ ] Set up feedback database schema

**Priority 3: Collect Initial Data**

- [ ] Run 20 episodes with Phase 2 agent
- [ ] Manually provide feedback for each episode
- [ ] Label with ground-truth rewards [-1, 1]
- [ ] Create initial training dataset

### Short Term (Month 1)

- [ ] Implement and train basic reward model
- [ ] Build simple feedback collection UI
- [ ] Create environment wrapper
- [ ] Run first PPO experiments in simulation

### Medium Term (Months 2-3)

- [ ] Full PPO trainer with GAE
- [ ] Demonstration system
- [ ] Integration with Phase 1+2 agents
- [ ] Initial benchmarking

### Long Term (Months 4-6)

- [ ] Advanced training features (curriculum, safety, continual learning)
- [ ] Full benchmark evaluation
- [ ] Research paper writing
- [ ] Public demo and open-source release

---

## Summary

**Phase 3** will transform the Reachy Mini system from a capable multi-agent AI (Phases 1+2) into a **continuously learning robot** that improves through human interaction.

**Key Innovations**:
1. ✅ Natural language feedback → scalar rewards (>85% accuracy)
2. ✅ Sample-efficient learning with BC + PPO (<1000 episodes)
3. ✅ Safe exploration with human oversight (<1% violations)
4. ✅ Strong generalization (>70% on novel tasks)

**Timeline**: 6 months (Months 7-12)

**Outcome**: Production-ready DRL system + research publication + open-source release

---

**Ready to Begin Phase 3!** 🚀

Let's discuss which component to start with, or review/modify the plan as needed.
