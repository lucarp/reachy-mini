"""
Utility functions for RL training.

Provides:
- Seeding for reproducibility
- Logging utilities
- Model saving/loading
- Metrics tracking
- Configuration management
"""

import numpy as np
import torch
import random
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"🌱 Random seed set to: {seed}")


class Logger:
    """Simple logger for training progress."""

    def __init__(self, log_dir: str = "logs", name: str = "training"):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save logs
            name: Logger name
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info(f"Logger initialized. Logging to {log_file}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)


class MetricsTracker:
    """Track training metrics over time."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'timestamps': [],
        }
        self.episode_buffer = {
            'rewards': [],
            'distances': [],
        }

    def start_episode(self):
        """Start tracking new episode."""
        self.episode_buffer = {
            'rewards': [],
            'distances': [],
        }

    def add_step(self, reward: float, distance: float):
        """Add step to current episode."""
        self.episode_buffer['rewards'].append(reward)
        self.episode_buffer['distances'].append(distance)

    def end_episode(self):
        """End current episode and compute statistics."""
        episode_reward = sum(self.episode_buffer['rewards'])
        episode_length = len(self.episode_buffer['rewards'])
        mean_distance = np.mean(self.episode_buffer['distances']) if self.episode_buffer['distances'] else 0

        self.metrics['episode_rewards'].append(episode_reward)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['timestamps'].append(datetime.now().isoformat())

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'mean_distance': mean_distance,
        }

    def add_loss(self, loss: float):
        """Add loss value."""
        self.metrics['losses'].append(loss)

    def get_recent_mean(self, key: str, n: int = 100) -> float:
        """
        Get mean of last n values for a metric.

        Args:
            key: Metric key
            n: Number of recent values

        Returns:
            Mean value
        """
        values = self.metrics.get(key, [])
        if not values:
            return 0.0
        recent = values[-n:]
        return np.mean(recent)

    def get_all(self) -> Dict[str, List]:
        """Get all metrics."""
        return self.metrics

    def summary(self, last_n: int = 100) -> str:
        """
        Get summary string of recent performance.

        Args:
            last_n: Number of recent episodes

        Returns:
            Summary string
        """
        if not self.metrics['episode_rewards']:
            return "No episodes completed yet"

        recent_rewards = self.metrics['episode_rewards'][-last_n:]
        recent_lengths = self.metrics['episode_lengths'][-last_n:]

        summary = f"""
        📊 Training Summary (Last {len(recent_rewards)} episodes):
        ───────────────────────────────────────────────
        Mean Reward:    {np.mean(recent_rewards):8.2f}
        Std Reward:     {np.std(recent_rewards):8.2f}
        Min Reward:     {np.min(recent_rewards):8.2f}
        Max Reward:     {np.max(recent_rewards):8.2f}
        Mean Length:    {np.mean(recent_lengths):8.2f}
        Total Episodes: {len(self.metrics['episode_rewards']):8d}
        """

        if self.metrics['losses']:
            recent_losses = self.metrics['losses'][-last_n:]
            summary += f"        Mean Loss:      {np.mean(recent_losses):8.4f}\n"

        return summary


class ModelCheckpoint:
    """Save and load model checkpoints."""

    def __init__(self, save_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict] = None,
        name: str = "model",
        epoch: Optional[int] = None,
    ):
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            metrics: Training metrics (optional)
            name: Checkpoint name
            epoch: Epoch number (optional)
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if epoch is not None:
            checkpoint['epoch'] = epoch
            filename = f"{name}_epoch{epoch}.pt"
        else:
            filename = f"{name}.pt"

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)

        print(f"💾 Saved checkpoint: {save_path}")

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        name: str = "model",
        epoch: Optional[int] = None,
    ) -> Dict:
        """
        Load model checkpoint.

        Args:
            model: PyTorch model to load into
            optimizer: Optimizer to load into (optional)
            name: Checkpoint name
            epoch: Epoch number (optional)

        Returns:
            Checkpoint dictionary
        """
        if epoch is not None:
            filename = f"{name}_epoch{epoch}.pt"
        else:
            filename = f"{name}.pt"

        load_path = self.save_dir / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"📂 Loaded checkpoint: {load_path}")

        return checkpoint

    def list_checkpoints(self, pattern: str = "*.pt") -> List[Path]:
        """
        List all checkpoints matching pattern.

        Args:
            pattern: File pattern

        Returns:
            List of checkpoint paths
        """
        return sorted(self.save_dir.glob(pattern))


def save_config(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON.

    Args:
        config: Configuration dictionary
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"💾 Saved config: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON.

    Args:
        filepath: Path to config file

    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config not found: {filepath}")

    with open(filepath, 'r') as f:
        config = json.load(f)

    print(f"📂 Loaded config: {filepath}")

    return config


def print_header(text: str, width: int = 70):
    """
    Print a formatted header.

    Args:
        text: Header text
        width: Total width
    """
    print()
    print("=" * width)
    print(text.center(width))
    print("=" * width)
    print()


def print_section(text: str, width: int = 70):
    """
    Print a formatted section header.

    Args:
        text: Section text
        width: Total width
    """
    print()
    print("-" * width)
    print(text)
    print("-" * width)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def compute_returns(rewards: List[float], gamma: float = 0.99) -> np.ndarray:
    """
    Compute discounted returns from rewards.

    Args:
        rewards: List of rewards
        gamma: Discount factor

    Returns:
        Array of discounted returns
    """
    returns = []
    G = 0

    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    return np.array(returns, dtype=np.float32)


def compute_advantages(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Array of advantage estimates
    """
    advantages = []
    gae = 0

    values = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    return np.array(advantages, dtype=np.float32)
