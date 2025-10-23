#!/usr/bin/env python3
"""
Evaluation script for trained REINFORCE agent.

This script:
1. Loads a trained REINFORCE agent
2. Evaluates it on the ball tracking task
3. Visualizes tracking performance
4. Compares with random baseline (optional)

Usage:
    python test.py --checkpoint reinforce_final.pt --episodes 100
    python test.py --baseline  # Compare with random policy
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

from shared import (
    BallTrackerEnv,
    set_seed,
    TrainingVisualizer,
    print_header,
    print_section,
)
from reinforce import REINFORCE


def evaluate_agent(agent, env, n_episodes: int = 100, render: bool = False):
    """
    Evaluate agent performance.

    Args:
        agent: REINFORCE agent
        env: Environment
        n_episodes: Number of episodes to evaluate
        render: Whether to render episodes

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_distances = []

    print(f"Evaluating agent for {n_episodes} episodes...")

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        distances = []

        done = False
        while not done:
            # Select action (no exploration, greedy)
            with torch.no_grad():
                action, _ = agent.policy.get_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_steps += 1
            distances.append(info['distance'])

            state = next_state

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        episode_distances.append(np.mean(distances))

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{n_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Steps = {episode_steps}, "
                  f"Mean Distance = {np.mean(distances):.3f}")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_distances': episode_distances,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_distance': np.mean(episode_distances),
    }


def evaluate_random_baseline(env, n_episodes: int = 100):
    """Evaluate random policy as baseline."""
    episode_rewards = []
    episode_distances = []

    print(f"Evaluating random baseline for {n_episodes} episodes...")

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        distances = []

        done = False
        while not done:
            # Random action
            action = env.action_space.sample()

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            distances.append(info['distance'])

            state = next_state

        episode_rewards.append(episode_reward)
        episode_distances.append(np.mean(distances))

    return {
        'episode_rewards': episode_rewards,
        'episode_distances': episode_distances,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_distance': np.mean(episode_distances),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate REINFORCE agent")

    parser.add_argument('--checkpoint', type=str, default='reinforce_final.pt',
                        help='Checkpoint filename (default: reinforce_final.pt)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--baseline', action='store_true',
                        help='Compare with random baseline')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')

    args = parser.parse_args()

    print_header("🧪 REINFORCE Evaluation")

    # Set seed
    set_seed(args.seed)

    # Create environment
    print_section("📦 Environment Setup")
    env = BallTrackerEnv(
        ball_speed=0.3,
        max_steps=300,
        use_real_robot=False,
        sim_mode=True,
    )
    print(f"✅ Environment created")

    # Create and load agent
    print_section("🤖 Loading Agent")
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
    )

    checkpoint_path = f"project1_reinforce/checkpoints/{args.checkpoint}"
    try:
        agent.load(checkpoint_path)
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the agent first using: python train.py")
        return

    # Evaluate agent
    print_section("🧪 Evaluation")
    agent_results = evaluate_agent(agent, env, n_episodes=args.episodes, render=args.render)

    print()
    print("=" * 70)
    print("REINFORCE Agent Performance".center(70))
    print("=" * 70)
    print(f"Mean Reward:    {agent_results['mean_reward']:8.2f} ± {agent_results['std_reward']:.2f}")
    print(f"Mean Length:    {agent_results['mean_length']:8.2f}")
    print(f"Mean Distance:  {agent_results['mean_distance']:8.4f}")
    print("=" * 70)

    # Compare with baseline
    if args.baseline:
        print_section("🎲 Random Baseline Comparison")
        baseline_results = evaluate_random_baseline(env, n_episodes=args.episodes)

        print()
        print("=" * 70)
        print("Random Policy Performance".center(70))
        print("=" * 70)
        print(f"Mean Reward:    {baseline_results['mean_reward']:8.2f} ± {baseline_results['std_reward']:.2f}")
        print(f"Mean Distance:  {baseline_results['mean_distance']:8.4f}")
        print("=" * 70)

        print()
        print("=" * 70)
        print("Comparison".center(70))
        print("=" * 70)
        improvement_reward = ((agent_results['mean_reward'] - baseline_results['mean_reward']) /
                              abs(baseline_results['mean_reward']) * 100)
        improvement_distance = ((baseline_results['mean_distance'] - agent_results['mean_distance']) /
                                baseline_results['mean_distance'] * 100)

        print(f"Reward Improvement:     {improvement_reward:6.1f}%")
        print(f"Distance Improvement:   {improvement_distance:6.1f}%")
        print("=" * 70)

        # Visualize comparison
        visualizer = TrainingVisualizer(save_dir="project1_reinforce/plots")

        visualizer.compare_algorithms(
            {
                'REINFORCE': agent_results['episode_rewards'],
                'Random Baseline': baseline_results['episode_rewards'],
            },
            title="REINFORCE vs Random Baseline",
            ylabel="Episode Reward",
            window_size=10,
            save_name="reinforce_vs_baseline.png",
        )

        visualizer.plot_final_comparison_bar(
            {
                'REINFORCE': agent_results['mean_reward'],
                'Random': baseline_results['mean_reward'],
            },
            title="Final Performance Comparison",
            ylabel="Mean Episode Reward",
            save_name="reinforce_final_comparison.png",
        )

    # Visualize tracking performance
    print_section("📊 Visualizing Performance")
    visualizer = TrainingVisualizer(save_dir="project1_reinforce/plots")

    # Sample one episode and visualize tracking
    state, _ = env.reset()
    distances = []
    done = False

    while not done:
        with torch.no_grad():
            action, _ = agent.policy.get_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        distances.append(info['distance'])
        state = next_state

    visualizer.plot_tracking_performance(
        distances,
        title="REINFORCE: Ball Tracking Performance (Single Episode)",
        save_name="reinforce_tracking_performance.png",
    )

    print()
    print_header("✅ Evaluation Complete!")

    env.close()


if __name__ == '__main__':
    main()
