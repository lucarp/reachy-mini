#!/usr/bin/env python3
"""
Interactive Gemma Vision: Ask Reachy anything about what it sees!

The robot will:
1. Look at the scene
2. Take a photo
3. Let YOU ask questions about what it sees
4. Use Gemma to answer your questions

Type 'quit' to exit, 'new photo' to take a new picture.

Requirements:
1. Daemon running: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
2. Ollama running: ollama serve
"""

import base64
import requests
import cv2
import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class InteractiveVisionRobot:
    """Robot that can answer questions about what it sees."""

    def __init__(self, reachy, model="gemma3:27b"):
        self.reachy = reachy
        self.model = model
        self.current_image_path = None
        self.base_url = "http://localhost:11434"

    def take_photo(self, pitch=-20, yaw=0):
        """Take a photo from the robot's camera."""
        print("\n📷 Taking photo...")

        # Position head
        self.reachy.set_target_antenna_joint_positions([-1.0, 1.0])
        pose = create_head_pose(pitch=pitch, yaw=yaw, degrees=True)
        self.reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        # Capture
        frame = self.reachy.media.camera.read()
        if frame is None:
            print("❌ Failed to capture image")
            return False

        # Save
        self.current_image_path = "/tmp/reachy_interactive.jpg"
        cv2.imwrite(self.current_image_path, frame)

        # Show to user
        cv2.imshow("Reachy's View", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        print(f"✓ Photo saved: {self.current_image_path}")
        return True

    def ask_about_image(self, question):
        """Ask Gemma about the current image."""
        if not self.current_image_path:
            return "❌ No photo taken yet! Type 'new photo' first."

        print(f"\n🧠 Asking Gemma...")
        print("   (Please wait 30-60 seconds...)")

        # Read and encode image
        try:
            with open(self.current_image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            return f"❌ Error reading image: {e}"

        # Query Gemma
        payload = {
            "model": self.model,
            "prompt": question,
            "images": [image_b64],
            "stream": False,
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.exceptions.Timeout:
            return "⏱️ Gemma took too long (timeout after 120s)"
        except Exception as e:
            return f"❌ Error: {e}"

    def express_emotion(self, emotion):
        """Express an emotion."""
        if emotion == "happy":
            self.reachy.set_target_antenna_joint_positions([-2.0, 2.0])
        elif emotion == "curious":
            self.reachy.set_target_antenna_joint_positions([-1.5, 0.0])
        elif emotion == "thinking":
            self.reachy.set_target_antenna_joint_positions([-0.5, 0.5])
        else:
            self.reachy.set_target_antenna_joint_positions([0.0, 0.0])

        time.sleep(1.0)


def main():
    print("=" * 70)
    print("🤖 Interactive Vision with Reachy Mini + Gemma 3")
    print("=" * 70)
    print("\nCommands:")
    print("  'new photo' or 'photo' - Take a new photo")
    print("  'quit' or 'exit' - Exit the program")
    print("  Or just ask any question about what Reachy sees!")
    print("=" * 70)

    with ReachyMini() as reachy:
        robot = InteractiveVisionRobot(reachy, model="gemma3:27b")

        print("\n✓ Connected to Reachy Mini")

        # Take initial photo
        print("\n📸 Taking initial photo...")
        if not robot.take_photo():
            print("❌ Failed to capture initial image")
            return 1

        robot.express_emotion("happy")
        print("\n✨ Ready! Ask me anything about what I see!")

        # Suggested questions
        print("\n💡 Suggested questions:")
        print("  - What objects can you see?")
        print("  - What colors are in the scene?")
        print("  - Describe the table")
        print("  - Count the items on the table")
        print("  - What is the most interesting object?")

        while True:
            print("\n" + "-" * 70)
            try:
                user_input = input("👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

            if not user_input:
                continue

            # Check for commands
            user_lower = user_input.lower()

            if user_lower in ['quit', 'exit', 'bye', 'goodbye']:
                robot.express_emotion("happy")
                print("\n🤖 Reachy: Goodbye! Thanks for exploring with me!")
                break

            elif user_lower in ['new photo', 'photo', 'take photo', 'new picture']:
                robot.take_photo()
                robot.express_emotion("happy")
                print("🤖 Reachy: New photo taken! What would you like to know?")
                continue

            # Ask Gemma
            robot.express_emotion("thinking")
            answer = robot.ask_about_image(user_input)

            print(f"\n🤖 Reachy: {answer}")
            robot.express_emotion("happy")

        # Reset to neutral
        robot.express_emotion("neutral")
        pose = create_head_pose()
        robot.reachy.goto_target(head=pose, duration=1.5)

        print("\n" + "=" * 70)
        print("Session complete!")
        print("=" * 70)

        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
