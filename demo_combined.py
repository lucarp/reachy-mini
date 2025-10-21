#!/usr/bin/env python3
"""
Demo combining head and antenna movements for expressive behaviors.
Make sure the daemon is running with: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
"""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time

def main():
    print("Reachy Mini Expressive Behavior Demo")
    print("=" * 50)

    with ReachyMini() as reachy_mini:

        # 1. Looking around curiously
        print("\n1. Looking around curiously...")
        reachy_mini.set_target_antenna_joint_positions([-1.5, 0.0])  # One antenna up
        time.sleep(0.5)

        pose = create_head_pose(yaw=30, pitch=-10, degrees=True)
        reachy_mini.goto_target(head=pose, duration=1.5)
        time.sleep(2.0)

        pose = create_head_pose(yaw=-30, pitch=-10, degrees=True)
        reachy_mini.set_target_antenna_joint_positions([0.0, -1.5])  # Switch antenna
        reachy_mini.goto_target(head=pose, duration=1.5)
        time.sleep(2.0)

        # 2. Happy greeting
        print("2. Happy greeting!")
        reachy_mini.set_target_antenna_joint_positions([-2.0, 2.0])  # Wide open
        pose = create_head_pose(z=5, pitch=10, degrees=True, mm=True)
        reachy_mini.goto_target(head=pose, duration=1.0)
        time.sleep(2.0)

        # 3. Nodding yes
        print("3. Nodding 'yes'...")
        reachy_mini.set_target_antenna_joint_positions([-1.0, 1.0])
        for _ in range(3):
            pose = create_head_pose(pitch=15, degrees=True)
            reachy_mini.goto_target(head=pose, duration=0.5)
            time.sleep(0.6)
            pose = create_head_pose(pitch=-15, degrees=True)
            reachy_mini.goto_target(head=pose, duration=0.5)
            time.sleep(0.6)

        # 4. Shaking 'no'
        print("4. Shaking 'no'...")
        reachy_mini.set_target_antenna_joint_positions([0.5, -0.5])  # Sad antennas
        pose = create_head_pose()  # Center first
        reachy_mini.goto_target(head=pose, duration=0.5)
        time.sleep(0.6)

        for _ in range(3):
            pose = create_head_pose(yaw=25, degrees=True)
            reachy_mini.goto_target(head=pose, duration=0.4)
            time.sleep(0.5)
            pose = create_head_pose(yaw=-25, degrees=True)
            reachy_mini.goto_target(head=pose, duration=0.4)
            time.sleep(0.5)

        # 5. Thinking pose
        print("5. Thinking...")
        pose = create_head_pose(pitch=-20, yaw=15, degrees=True)
        reachy_mini.set_target_antenna_joint_positions([-0.5, 0.5])
        reachy_mini.goto_target(head=pose, duration=1.5)
        time.sleep(2.5)

        # 6. Reset to neutral
        print("6. Back to neutral position")
        pose = create_head_pose()
        reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        print("\nDemo complete!")

if __name__ == "__main__":
    main()
