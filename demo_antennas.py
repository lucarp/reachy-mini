#!/usr/bin/env python3
"""
Demo script showing antenna movements - antennas can express emotions!
Make sure the daemon is running with: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
"""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time
import math

def main():
    print("Reachy Mini Antenna Expression Demo")
    print("=" * 50)

    with ReachyMini() as reachy_mini:
        # Neutral position
        print("\n1. Neutral position")
        reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])
        time.sleep(1.5)

        # Happy - antennas pointing outward
        print("2. Happy! (antennas outward)")
        reachy_mini.set_target_antenna_joint_positions([-1.0, 1.0])
        time.sleep(1.5)

        # Curious - one antenna up
        print("3. Curious? (one antenna raised)")
        reachy_mini.set_target_antenna_joint_positions([-1.5, 0.0])
        time.sleep(1.5)

        # Very happy - wide open
        print("4. Very happy! (wide open)")
        reachy_mini.set_target_antenna_joint_positions([-2.0, 2.0])
        time.sleep(1.5)

        # Sad - antennas drooping
        print("5. Sad... (antennas drooping)")
        reachy_mini.set_target_antenna_joint_positions([0.5, -0.5])
        time.sleep(1.5)

        # Wiggle animation
        print("6. Wiggling antennas!")
        for i in range(6):
            angle = math.sin(i * 0.5) * 1.5
            reachy_mini.set_target_antenna_joint_positions([angle, -angle])
            time.sleep(0.3)

        # Back to neutral
        print("7. Back to neutral")
        reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])
        time.sleep(1.0)

        print("\nDemo complete!")

if __name__ == "__main__":
    main()
