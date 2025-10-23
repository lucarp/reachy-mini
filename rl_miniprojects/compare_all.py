#!/usr/bin/env python3
"""
Compare all 4 RL algorithms on the ball tracking task.

This script:
1. Loads metrics from all 4 projects
2. Creates comparison visualizations
3. Generates a final performance report

Usage:
    python compare_all.py
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from shared import TrainingVisualizer, print_header, print_section


def load_project_metrics(project_name: str) -> dict:
    """Load training metrics for a project."""
    metrics_file = Path(project_name) / "plots" / "training_metrics.json"

    if not metrics_file.exists():
        print(f"⚠️  Metrics not found for {project_name}")
        return None

    try:
        visualizer = TrainingVisualizer(save_dir=f"{project_name}/plots")
        metrics = visualizer.load_metrics("training_metrics.json")
        return metrics
    except Exception as e:
        print(f"⚠️  Error loading {project_name}: {e}")
        return None


def compute_final_scores(metrics: dict, last_n: int = 100) -> dict:
    """Compute final performance scores."""
    rewards = metrics['episode_rewards']

    return {
        'mean_reward': np.mean(rewards[-last_n:]),
        'std_reward': np.std(rewards[-last_n:]),
        'best_reward': np.max(rewards),
        'final_reward': np.mean(rewards[-10:]),
        'episodes_to_converge': len(rewards),
        'total_episodes': len(rewards),
    }


def main():
    print_header("📊 RL Algorithms Comparison - Ball Tracking Task")

    # Load metrics from all projects
    print_section("📂 Loading Project Metrics")

    projects = {
        'REINFORCE': 'project1_reinforce',
        'DQN': 'project2_dqn',
        'A2C': 'project3_a2c',
        'PPO': 'project4_ppo',
    }

    all_metrics = {}
    all_scores = {}

    for algo_name, project_dir in projects.items():
        print(f"Loading {algo_name}...", end=" ")
        metrics = load_project_metrics(project_dir)

        if metrics is not None:
            all_metrics[algo_name] = metrics
            all_scores[algo_name] = compute_final_scores(metrics)
            print("✅")
        else:
            print("❌ (not found)")

    if not all_metrics:
        print("\n❌ No project metrics found!")
        print("   Please train at least one algorithm first.")
        return

    # Create comparison visualizations
    print_section("📊 Creating Comparison Visualizations")

    visualizer = TrainingVisualizer(save_dir="comparison_plots")

    # 1. Training curves comparison
    print("Creating training curves comparison...", end=" ")
    rewards_dict = {
        algo: metrics['episode_rewards']
        for algo, metrics in all_metrics.items()
    }

    visualizer.compare_algorithms(
        rewards_dict,
        title="Algorithm Comparison: Ball Tracking Task",
        ylabel="Episode Reward",
        window_size=20,
        save_name="all_algorithms_comparison.png",
    )
    print("✅")

    # 2. Final performance bar chart
    print("Creating final performance comparison...", end=" ")
    final_rewards = {
        algo: scores['mean_reward']
        for algo, scores in all_scores.items()
    }

    visualizer.plot_final_comparison_bar(
        final_rewards,
        title="Final Performance Comparison (Last 100 Episodes)",
        ylabel="Mean Episode Reward",
        save_name="final_performance_comparison.png",
    )
    print("✅")

    # 3. Learning curves grid
    print("Creating learning curves grid...", end=" ")
    if len(all_metrics) >= 2:
        visualizer.plot_learning_curves_grid(
            {algo: {'Reward': metrics['episode_rewards'],
                    'Episode Length': metrics['episode_lengths']}
             for algo, metrics in all_metrics.items()},
            title="Learning Curves Comparison",
            save_name="learning_curves_grid.png",
        )
        print("✅")
    else:
        print("⏭️  (need at least 2 algorithms)")

    # Print comparison table
    print_section("📈 Performance Comparison")

    print()
    print("=" * 100)
    print(f"{'Algorithm':<15} {'Mean Reward':<15} {'Std Reward':<15} {'Best Reward':<15} {'Episodes':<15}")
    print("=" * 100)

    for algo, scores in sorted(all_scores.items(), key=lambda x: x[1]['mean_reward'], reverse=True):
        print(f"{algo:<15} "
              f"{scores['mean_reward']:>12.2f}   "
              f"{scores['std_reward']:>12.2f}   "
              f"{scores['best_reward']:>12.2f}   "
              f"{scores['total_episodes']:>12d}")

    print("=" * 100)
    print()

    # Sample efficiency comparison
    print_section("⚡ Sample Efficiency")

    print()
    print("Episodes to reach mean reward > -30:")
    print()

    for algo, metrics in all_metrics.items():
        rewards = metrics['episode_rewards']

        # Find first episode where moving average > -30
        threshold = -30
        found = False

        for i in range(50, len(rewards)):
            avg_reward = np.mean(rewards[i-50:i])
            if avg_reward > threshold:
                print(f"  {algo:<15} {i:>5} episodes")
                found = True
                break

        if not found:
            print(f"  {algo:<15} {'Not reached':>5}")

    print()

    # Save comparison report
    print_section("💾 Saving Comparison Report")

    report = {
        'algorithms': list(all_scores.keys()),
        'scores': all_scores,
        'task': 'Ball Tracking',
        'state_dim': 6,
        'action_dim': 2,
        'max_steps': 300,
    }

    with open('comparison_plots/comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("✅ Saved: comparison_plots/comparison_report.json")
    print()

    # Final summary
    print_header("🎉 Comparison Complete!")

    print()
    print("Visualizations saved to: comparison_plots/")
    print()
    print("Files created:")
    print("  - all_algorithms_comparison.png")
    print("  - final_performance_comparison.png")
    print("  - learning_curves_grid.png")
    print("  - comparison_report.json")
    print()

    # Determine winner
    if all_scores:
        best_algo = max(all_scores.items(), key=lambda x: x[1]['mean_reward'])
        print(f"🏆 Best Algorithm: {best_algo[0]} (Reward: {best_algo[1]['mean_reward']:.2f})")
        print()

        # Recommendations
        print("📝 Recommendations:")
        print()
        print("  - For continuous actions: PPO or A2C")
        print("  - For discrete actions: DQN")
        print("  - For sample efficiency: PPO or DQN")
        print("  - For stability: PPO")
        print("  - For simplicity: REINFORCE")
        print()

    print("Next step: Apply your knowledge to Phase 3! 🚀")
    print()


if __name__ == '__main__':
    main()
