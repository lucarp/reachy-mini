#!/usr/bin/env python3
"""
Demo showing how to access the camera feed.
Make sure the daemon is running with: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
"""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import cv2
import time

def main():
    print("Reachy Mini Camera Demo")
    print("=" * 50)
    print("Press 'q' to quit the camera view\n")

    with ReachyMini() as reachy_mini:
        # Move head to look at the table
        print("1. Looking at the table...")
        pose = create_head_pose(pitch=-20, degrees=True)
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        print("2. Opening camera feed...")
        print("   (Press 'q' in the camera window to stop)")

        # Capture frames for 10 seconds or until 'q' is pressed
        frame_count = 0
        start_time = time.time()

        while time.time() - start_time < 10:
            # Get the current camera frame
            frame = reachy_mini.media.camera.read()

            if frame is not None:
                frame_count += 1

                # Add some info to the frame
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display the frame
                cv2.imshow("Reachy Mini Camera", frame)

                # Check for 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n   User pressed 'q', stopping...")
                    break
            else:
                print("   Warning: No frame received")
                time.sleep(0.1)

        cv2.destroyAllWindows()
        print(f"3. Captured {frame_count} frames")

        # Look around while showing camera
        print("\n4. Looking around...")
        poses = [
            create_head_pose(yaw=30, pitch=-15, degrees=True),
            create_head_pose(yaw=-30, pitch=-15, degrees=True),
            create_head_pose(yaw=0, pitch=-25, degrees=True),
        ]

        for i, pose in enumerate(poses):
            print(f"   Position {i+1}/3")
            reachy_mini.goto_target(head=pose, duration=1.5)
            time.sleep(2.0)

        # Reset position
        print("5. Resetting to default position")
        pose = create_head_pose()
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        print("\nDemo complete!")

if __name__ == "__main__":
    main()
