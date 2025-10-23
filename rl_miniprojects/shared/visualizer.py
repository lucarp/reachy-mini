"""
Visualization utilities for RL training.

Provides tools to:
- Plot training curves (rewards, losses)
- Visualize tracking performance
- Create comparison plots across algorithms
- Save and load training statistics
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class TrainingVisualizer:
    """Visualize training progress and results."""

    def __init__(self, save_dir: str = "plots"):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    def plot_training_curve(
        self,
        rewards: List[float],
        title: str = "Training Progress",
        ylabel: str = "Episode Reward",
        window_size: int = 10,
        save_name: Optional[str] = None,
    ):
        """
        Plot training curve with moving average.

        Args:
            rewards: List of episode rewards
            title: Plot title
            ylabel: Y-axis label
            window_size: Window size for moving average
            save_name: Filename to save plot (if None, only display)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        episodes = np.arange(len(rewards))

        # Plot raw rewards
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')

        # Plot moving average
        if len(rewards) >= window_size:
            moving_avg = self._moving_average(rewards, window_size)
            ax.plot(
                episodes[window_size - 1:],
                moving_avg,
                color='red',
                linewidth=2,
                label=f'Moving Avg ({window_size})'
            )

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.show()
        plt.close()

    def plot_multi_metric(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        window_size: int = 10,
        save_name: Optional[str] = None,
    ):
        """
        Plot multiple metrics on the same figure.

        Args:
            metrics: Dictionary of metric_name -> values
            title: Plot title
            window_size: Window size for moving average
            save_name: Filename to save plot
        """
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

        if len(metrics) == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, metrics.items()):
            episodes = np.arange(len(values))

            # Plot raw values
            ax.plot(episodes, values, alpha=0.3, label='Raw')

            # Plot moving average
            if len(values) >= window_size:
                moving_avg = self._moving_average(values, window_size)
                ax.plot(
                    episodes[window_size - 1:],
                    moving_avg,
                    linewidth=2,
                    label=f'Moving Avg ({window_size})'
                )

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.show()
        plt.close()

    def plot_tracking_performance(
        self,
        distances: List[float],
        title: str = "Ball Tracking Performance",
        save_name: Optional[str] = None,
    ):
        """
        Visualize tracking performance (distance from center over time).

        Args:
            distances: List of distances per step
            title: Plot title
            save_name: Filename to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot distance over time
        steps = np.arange(len(distances))
        ax1.plot(steps, distances, alpha=0.7, color='orange')
        ax1.axhline(y=0.1, color='green', linestyle='--', label='Good tracking (<0.1)')
        ax1.fill_between(steps, 0, 0.1, alpha=0.2, color='green')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Distance from Center')
        ax1.set_title('Tracking Error Over Episode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot histogram of distances
        ax2.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0.1, color='green', linestyle='--', label='Good tracking')
        ax2.axvline(x=np.mean(distances), color='red', linestyle='-', label=f'Mean: {np.mean(distances):.3f}')
        ax2.set_xlabel('Distance from Center')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Tracking Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.show()
        plt.close()

    def compare_algorithms(
        self,
        algorithm_results: Dict[str, List[float]],
        title: str = "Algorithm Comparison",
        ylabel: str = "Episode Reward",
        window_size: int = 10,
        save_name: Optional[str] = None,
    ):
        """
        Compare multiple algorithms on the same plot.

        Args:
            algorithm_results: Dictionary of algorithm_name -> rewards
            title: Plot title
            ylabel: Y-axis label
            window_size: Window size for moving average
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

        for i, (algo_name, rewards) in enumerate(algorithm_results.items()):
            color = colors[i % len(colors)]
            episodes = np.arange(len(rewards))

            # Plot moving average (primary line)
            if len(rewards) >= window_size:
                moving_avg = self._moving_average(rewards, window_size)
                ax.plot(
                    episodes[window_size - 1:],
                    moving_avg,
                    color=color,
                    linewidth=2.5,
                    label=algo_name
                )

                # Add shaded region for std
                moving_std = self._moving_std(rewards, window_size)
                ax.fill_between(
                    episodes[window_size - 1:],
                    moving_avg - moving_std,
                    moving_avg + moving_std,
                    color=color,
                    alpha=0.2
                )

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.show()
        plt.close()

    def plot_learning_curves_grid(
        self,
        metrics_dict: Dict[str, Dict[str, List[float]]],
        title: str = "Learning Curves",
        save_name: Optional[str] = None,
    ):
        """
        Create a grid of learning curves for different metrics.

        Args:
            metrics_dict: Nested dict {algorithm: {metric_name: values}}
            title: Overall title
            save_name: Filename to save plot
        """
        # Extract metric names
        first_algo = list(metrics_dict.keys())[0]
        metric_names = list(metrics_dict[first_algo].keys())

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        colors = ['blue', 'red', 'green', 'purple']

        for i, metric_name in enumerate(metric_names):
            ax = axes[i]

            for j, (algo_name, metrics) in enumerate(metrics_dict.items()):
                if metric_name in metrics:
                    values = metrics[metric_name]
                    episodes = np.arange(len(values))
                    color = colors[j % len(colors)]

                    # Plot moving average
                    if len(values) >= 10:
                        moving_avg = self._moving_average(values, 10)
                        ax.plot(
                            episodes[9:],
                            moving_avg,
                            color=color,
                            linewidth=2,
                            label=algo_name
                        )

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.show()
        plt.close()

    def plot_final_comparison_bar(
        self,
        algorithm_scores: Dict[str, float],
        title: str = "Final Performance Comparison",
        ylabel: str = "Average Reward (Last 100 Episodes)",
        save_name: Optional[str] = None,
    ):
        """
        Create a bar chart comparing final performance.

        Args:
            algorithm_scores: Dictionary of algorithm_name -> final_score
            title: Plot title
            ylabel: Y-axis label
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = list(algorithm_scores.keys())
        scores = list(algorithm_scores.values())

        colors = ['blue', 'red', 'green', 'purple', 'orange']
        bars = ax.bar(algorithms, scores, color=colors[:len(algorithms)], alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved plot: {save_path}")

        plt.show()
        plt.close()

    @staticmethod
    def _moving_average(data: List[float], window_size: int) -> np.ndarray:
        """Compute moving average."""
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    @staticmethod
    def _moving_std(data: List[float], window_size: int) -> np.ndarray:
        """Compute moving standard deviation."""
        data_arr = np.array(data)
        result = []

        for i in range(window_size - 1, len(data_arr)):
            window = data_arr[i - window_size + 1:i + 1]
            result.append(np.std(window))

        return np.array(result)

    def save_metrics(
        self,
        metrics: Dict[str, any],
        filename: str = "training_metrics.json"
    ):
        """
        Save training metrics to JSON.

        Args:
            metrics: Dictionary of metrics
            filename: Filename to save
        """
        save_path = self.save_dir / filename

        # Convert numpy arrays to lists
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        metrics_serializable = convert_to_serializable(metrics)

        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        print(f"💾 Saved metrics: {save_path}")

    def load_metrics(self, filename: str = "training_metrics.json") -> Dict:
        """
        Load training metrics from JSON.

        Args:
            filename: Filename to load

        Returns:
            Dictionary of metrics
        """
        load_path = self.save_dir / filename

        with open(load_path, 'r') as f:
            metrics = json.load(f)

        print(f"📂 Loaded metrics: {load_path}")
        return metrics
