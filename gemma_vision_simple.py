#!/usr/bin/env python3
"""
Simple Gemma 3 Vision Test: Quick test of Reachy seeing with Gemma.

This is a simplified version that:
1. Takes ONE photo
2. Asks Gemma to describe it
3. Shows the result

Much faster than the full demo!

Requirements:
1. Daemon running: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
2. Ollama running: ollama serve
"""

import base64
import requests
import cv2
import time
import sys
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def ask_gemma_about_image(image_path, question, model="gemma3:27b"):
    """Ask Gemma to describe an image."""
    print(f"\n🧠 Asking Gemma: '{question}'")
    print("   (This may take 30-60 seconds...)")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {
        "model": model,
        "prompt": question,
        "images": [image_b64],
        "stream": False,
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error: {e}"


def main():
    print("=" * 60)
    print("🤖 Reachy Mini + Gemma 3 Vision - Simple Test")
    print("=" * 60)

    with ReachyMini() as reachy:
        print("\n✓ Connected to Reachy Mini")

        # Look at the table
        print("\n📷 Positioning robot to look at the scene...")
        reachy.set_target_antenna_joint_positions([-1.5, 0.0])  # Curious
        pose = create_head_pose(pitch=-20, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        # Capture image
        print("📸 Capturing image...")
        frame = reachy.media.camera.read()

        if frame is None:
            print("❌ Failed to capture image!")
            return 1

        # Save image
        image_path = "/tmp/reachy_view_gemma.jpg"
        cv2.imwrite(image_path, frame)
        print(f"   Saved to {image_path}")

        # Show image to user
        cv2.imshow("What Reachy Sees", frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()

        # Ask Gemma to describe it
        question = "Describe what you see in this image in 2-3 sentences. Be specific."
        answer = ask_gemma_about_image(image_path, question)

        # Show result
        print("\n" + "=" * 60)
        print("🤖 GEMMA'S RESPONSE:")
        print("=" * 60)
        print(f"\n{answer}\n")
        print("=" * 60)

        # Happy expression!
        print("\n✨ Reachy understood the scene!")
        reachy.set_target_antenna_joint_positions([-2.0, 2.0])
        time.sleep(2.0)

        # Reset
        reachy.set_target_antenna_joint_positions([0.0, 0.0])
        pose = create_head_pose()
        reachy.goto_target(head=pose, duration=1.5)

        print("\n✓ Test complete!")
        print(f"📁 Image saved at: {image_path}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
