#!/usr/bin/env python3
"""
Training script for REINFORCE algorithm on Ball Tracking task.

This script demonstrates:
1. Environment setup
2. Agent initialization
3. Training loop with episode collection
4. Logging and visualization
5. Model checkpointing

Usage:
    python train.py --episodes 500 --seed 42
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import numpy as np

from shared import (
    BallTrackerEnv,
    set_seed,
    MetricsTracker,
    TrainingVisualizer,
    ModelCheckpoint,
    print_header,
    print_section,
    format_time,
)
from reinforce import REINFORCE


def train(args):
    """Main training function."""

    print_header("🎓 Project 1: REINFORCE Training")

    # Set random seed for reproducibility
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
    print(f"   State space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Max episode steps: {env.max_steps}")

    # Create agent
    print_section("🤖 Agent Setup")
    agent = REINFORCE(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=args.hidden_dim,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        gamma=args.gamma,
    )
    print(f"✅ REINFORCE agent created")
    print(f"   Policy network: {sum(p.numel() for p in agent.policy.parameters())} parameters")
    print(f"   Value network: {sum(p.numel() for p in agent.value.parameters())} parameters")
    print(f"   Policy LR: {args.policy_lr}")
    print(f"   Value LR: {args.value_lr}")
    print(f"   Gamma: {args.gamma}")

    # Create metrics tracker and visualizer
    metrics = MetricsTracker()
    visualizer = TrainingVisualizer(save_dir="project1_reinforce/plots")
    checkpoint = ModelCheckpoint(save_dir="project1_reinforce/checkpoints")

    # Training loop
    print_section("🏋️ Training")
    print(f"Training for {args.episodes} episodes...")
    print()

    start_time = time.time()

    for episode in range(args.episodes):
        # Reset environment
        state, _ = env.reset()
        metrics.start_episode()

        episode_reward = 0
        episode_steps = 0

        # Collect episode
        done = False
        while not done:
            # Select action
            action = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store reward
            agent.store_reward(reward)

            # Track metrics
            metrics.add_step(reward, info['distance'])

            episode_reward += reward
            episode_steps += 1

            state = next_state

        # Update policy after episode
        losses = agent.update()

        # End episode
        episode_info = metrics.end_episode()
        metrics.add_loss(losses['policy_loss'])

        # Logging
        if (episode + 1) % args.log_interval == 0:
            mean_reward = metrics.get_recent_mean('episode_rewards', n=args.log_interval)
            mean_length = metrics.get_recent_mean('episode_lengths', n=args.log_interval)
            mean_distance = episode_info['mean_distance']
            elapsed_time = time.time() - start_time

            print(f"Episode {episode + 1:4d}/{args.episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Mean Reward ({args.log_interval}): {mean_reward:7.2f} | "
                  f"Steps: {episode_steps:3d} | "
                  f"Mean Dist: {mean_distance:.3f} | "
                  f"Policy Loss: {losses['policy_loss']:.4f} | "
                  f"Value Loss: {losses['value_loss']:.4f} | "
                  f"Time: {format_time(elapsed_time)}")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            checkpoint.save(
                model=agent.policy,
                optimizer=agent.policy_optimizer,
                metrics=metrics.get_all(),
                name="reinforce_policy",
                epoch=episode + 1,
            )
            agent.save(f"project1_reinforce/checkpoints/reinforce_episode{episode + 1}.pt")

    # Training complete
    total_time = time.time() - start_time

    print()
    print_section("✅ Training Complete")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per episode: {total_time / args.episodes:.2f}s")
    print()
    print(metrics.summary(last_n=100))

    # Save final model
    agent.save("project1_reinforce/checkpoints/reinforce_final.pt")
    print("💾 Saved final model to: project1_reinforce/checkpoints/reinforce_final.pt")

    # Visualizations
    print_section("📊 Generating Visualizations")

    # Plot training curve
    rewards = metrics.get_all()['episode_rewards']
    visualizer.plot_training_curve(
        rewards,
        title="REINFORCE: Ball Tracking Training Progress",
        ylabel="Episode Reward",
        window_size=20,
        save_name="reinforce_training_curve.png",
    )

    # Plot multiple metrics
    visualizer.plot_multi_metric(
        {
            'Episode Reward': metrics.get_all()['episode_rewards'],
            'Episode Length': metrics.get_all()['episode_lengths'],
            'Policy Loss': metrics.get_all()['losses'],
        },
        title="REINFORCE: Training Metrics",
        window_size=20,
        save_name="reinforce_metrics.png",
    )

    # Save metrics
    visualizer.save_metrics(
        metrics.get_all(),
        filename="reinforce_metrics.json",
    )

    print()
    print_header("🎉 REINFORCE Training Complete!")
    print()
    print("Next steps:")
    print("  1. Run evaluation: python test.py")
    print("  2. Compare with baseline: python test.py --baseline")
    print("  3. Visualize policy: python visualize_policy.py")
    print()

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train REINFORCE on Ball Tracking")

    # Training parameters
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Agent parameters
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    parser.add_argument('--policy-lr', type=float, default=3e-4,
                        help='Policy learning rate (default: 3e-4)')
    parser.add_argument('--value-lr', type=float, default=1e-3,
                        help='Value learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')

    # Logging parameters
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N episodes (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save checkpoint every N episodes (default: 100)')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
