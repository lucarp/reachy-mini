"""
REINFORCE Algorithm Implementation (Policy Gradient)

REINFORCE is the simplest policy gradient algorithm. It:
1. Samples a full episode trajectory using current policy
2. Computes returns (discounted sum of rewards) for each timestep
3. Uses policy gradient theorem to update policy parameters

Algorithm:
    Initialize policy network π_θ with random weights θ
    for each episode:
        Generate episode (s_0, a_0, r_1, s_1, ..., s_T) using π_θ
        for each timestep t:
            G_t = Σ_{k=t}^T γ^{k-t} r_k  (return from timestep t)
            θ ← θ + α ∇_θ log π_θ(a_t | s_t) * (G_t - b)  (policy gradient)

where:
    - G_t is the discounted return (actual cumulative reward)
    - b is a baseline (typically state value) to reduce variance
    - α is the learning rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import List, Tuple, Dict


class PolicyNetwork(nn.Module):
    """
    Policy network for continuous actions.

    Architecture:
        - Input: state (6D)
        - Hidden layers: 2 layers with 64 units each
        - Output: mean and log_std for Gaussian policy
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize policy network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
        """
        super().__init__()

        # Shared feature extractor
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Policy head (outputs mean and log_std for Gaussian)
        self.mean_head = nn.Linear(hidden_dim, action_dim)

        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            mean: Action mean [batch_size, action_dim]
            std: Action std [batch_size, action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = torch.tanh(self.mean_head(x)) * 0.1  # Scale to [-0.1, 0.1]
        std = torch.exp(self.log_std).expand_as(mean)  # Ensure std > 0

        return mean, std

    def get_action(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Sample action from policy.

        Args:
            state: State array

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            mean, std = self.forward(state_tensor)

        # Create Gaussian distribution
        dist = Normal(mean, std)

        # Sample action
        action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor).sum(dim=-1)

        action = action_tensor.squeeze(0).numpy()
        log_prob_value = log_prob.item()

        return action, log_prob_value


class ValueNetwork(nn.Module):
    """
    Value network for baseline (state value function).

    This reduces variance in policy gradient estimates.

    Architecture:
        - Input: state (6D)
        - Hidden layers: 2 layers with 64 units each
        - Output: scalar state value
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        """
        Initialize value network.

        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer size
        """
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            value: State value [batch_size, 1]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)

        return value


class REINFORCE:
    """
    REINFORCE algorithm with baseline.

    This implementation includes:
    - Policy network (π_θ)
    - Value network (V_φ) as baseline
    - Separate optimizers for policy and value
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        """
        Initialize REINFORCE agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
            policy_lr: Learning rate for policy network
            value_lr: Learning rate for value network
            gamma: Discount factor
        """
        self.gamma = gamma

        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=policy_lr
        )

        # Value network (baseline)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=value_lr
        )

        # Episode buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            action: Selected action
        """
        action, log_prob = self.policy.get_action(state)

        # Store for training
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)

        return action

    def store_reward(self, reward: float):
        """Store reward for current timestep."""
        self.rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """
        Compute discounted returns for episode.

        Returns:
            returns: Tensor of returns [T]
        """
        returns = []
        G = 0

        # Compute returns backwards
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # Normalize returns (reduces variance)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self) -> Dict[str, float]:
        """
        Update policy and value networks after episode.

        Returns:
            Dictionary with loss values
        """
        if len(self.rewards) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0}

        # Convert episode data to tensors
        states_tensor = torch.FloatTensor(np.array(self.states))
        log_probs_tensor = torch.FloatTensor(self.log_probs)

        # Compute returns
        returns = self.compute_returns()

        # Compute baselines (state values)
        values = self.value(states_tensor).squeeze()

        # Compute advantages (return - baseline)
        advantages = returns - values.detach()  # Detach to not backprop through value

        # --- Policy Update ---
        # Policy gradient: ∇J = E[∇log π(a|s) * (G - b)]
        policy_loss = -(log_probs_tensor * advantages).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()

        # --- Value Update ---
        # MSE between predicted value and actual return
        value_loss = F.mse_loss(values, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
        self.value_optimizer.step()

        # Clear episode buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
