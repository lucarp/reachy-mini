#!/usr/bin/env python3
"""
Simple test script to control Reachy Mini in simulation.
Make sure the daemon is running with: mjpython -m reachy_mini.daemon.app.main --sim
"""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time

def main():
    print("Connecting to Reachy Mini...")

    with ReachyMini() as reachy_mini:
        print("Connected!")
        print("Current head pose:\n", reachy_mini.get_current_head_pose())
        print("Current joint positions:", reachy_mini.get_current_joint_positions())

        # Move the head up (10mm on z-axis) and roll it 15 degrees
        print("\n1. Moving head up and rolling 15 degrees...")
        pose = create_head_pose(z=10, roll=15, degrees=True, mm=True)
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        # Tilt the head forward
        print("2. Tilting head forward...")
        pose = create_head_pose(pitch=-20, degrees=True)
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        # Rotate the head left and right
        print("3. Rotating head left...")
        pose = create_head_pose(yaw=30, degrees=True)
        reachy_mini.goto_target(head=pose, duration=1.5)
        time.sleep(2.0)

        print("4. Rotating head right...")
        pose = create_head_pose(yaw=-30, degrees=True)
        reachy_mini.goto_target(head=pose, duration=1.5)
        time.sleep(2.0)

        # Reset to default pose
        print("5. Resetting to default pose...")
        pose = create_head_pose()
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        print("\nTest complete!")

if __name__ == "__main__":
    main()
