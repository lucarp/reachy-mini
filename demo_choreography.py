#!/usr/bin/env python3
"""
Advanced demo showing choreographed movements - a little dance!
Make sure the daemon is running with: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
"""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time
import math

def wave_motion(reachy_mini, duration=3.0, amplitude=20):
    """Create a smooth wave-like motion with the head."""
    print("   Performing wave motion...")
    steps = 20
    for i in range(steps):
        t = i / steps
        yaw = amplitude * math.sin(t * 2 * math.pi)
        pitch = amplitude/2 * math.sin(t * 2 * math.pi + math.pi/2)
        pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
        reachy_mini.goto_target(head=pose, duration=duration/steps)
        time.sleep(duration/steps)

def spiral_motion(reachy_mini, rotations=2, duration=4.0):
    """Create a spiral motion with the head."""
    print("   Performing spiral motion...")
    steps = 30
    for i in range(steps):
        t = i / steps
        angle = t * rotations * 2 * math.pi
        radius = 15 * (1 - t)  # Decrease radius as we spiral in
        yaw = radius * math.cos(angle)
        pitch = radius * math.sin(angle)
        pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
        reachy_mini.goto_target(head=pose, duration=duration/steps)
        time.sleep(duration/steps)

def figure_eight(reachy_mini, size=20, duration=5.0):
    """Draw a figure-8 pattern with the head."""
    print("   Drawing figure-8...")
    steps = 40
    for i in range(steps):
        t = i / steps * 2 * math.pi
        yaw = size * math.sin(t)
        pitch = size/2 * math.sin(2 * t)
        pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
        reachy_mini.goto_target(head=pose, duration=duration/steps)
        time.sleep(duration/steps)

def main():
    print("Reachy Mini Choreography Demo")
    print("=" * 50)

    with ReachyMini() as reachy_mini:

        # 1. Greeting sequence
        print("\n1. Greeting sequence")
        # Start with antennas happy
        reachy_mini.set_target_antenna_joint_positions([-2.0, 2.0])
        # Small bow
        pose = create_head_pose(pitch=20, degrees=True)
        reachy_mini.goto_target(head=pose, duration=1.0)
        time.sleep(1.5)
        # Back up
        pose = create_head_pose()
        reachy_mini.goto_target(head=pose, duration=1.0)
        time.sleep(1.5)

        # 2. Wave motion
        print("\n2. Wave motion choreography")
        reachy_mini.set_target_antenna_joint_positions([-1.0, 1.0])
        wave_motion(reachy_mini, duration=4.0, amplitude=25)

        # 3. Spiral in
        print("\n3. Spiral choreography")
        reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])
        spiral_motion(reachy_mini, rotations=2, duration=5.0)

        # 4. Figure-8 pattern
        print("\n4. Figure-8 pattern")
        reachy_mini.set_target_antenna_joint_positions([-1.5, 1.5])
        figure_eight(reachy_mini, size=20, duration=5.0)

        # 5. Excited wiggle
        print("\n5. Excited wiggle!")
        for i in range(8):
            # Alternate head tilt and antenna positions
            yaw = 20 if i % 2 == 0 else -20
            antenna_left = -1.5 if i % 2 == 0 else -0.5
            antenna_right = 1.5 if i % 2 == 0 else 0.5

            pose = create_head_pose(yaw=yaw, pitch=-10, degrees=True)
            reachy_mini.set_target_antenna_joint_positions([antenna_left, antenna_right])
            reachy_mini.goto_target(head=pose, duration=0.3)
            time.sleep(0.35)

        # 6. Slow scan of environment
        print("\n6. Environmental scan")
        scan_positions = [
            (-40, -15),  # Left-down
            (-40, 0),    # Left-center
            (-40, 15),   # Left-up
            (0, 15),     # Center-up
            (40, 15),    # Right-up
            (40, 0),     # Right-center
            (40, -15),   # Right-down
            (0, -15),    # Center-down
        ]

        reachy_mini.set_target_antenna_joint_positions([-1.5, 0.0])  # Curious
        for i, (yaw, pitch) in enumerate(scan_positions):
            print(f"   Scanning position {i+1}/{len(scan_positions)}")
            pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
            reachy_mini.goto_target(head=pose, duration=1.0)
            time.sleep(1.2)

        # 7. Final reset
        print("\n7. Returning to neutral position")
        pose = create_head_pose()
        reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        print("\n" + "=" * 50)
        print("Choreography complete! 🎭")
        print("=" * 50)

if __name__ == "__main__":
    main()
