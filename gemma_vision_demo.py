#!/usr/bin/env python3
"""
Gemma 3 Vision Demo: Reachy Mini uses its camera + Gemma multimodal LLM to see and describe.

Gemma 3 is a multimodal model that can process both text and images with a 128K context window
and support for 140+ languages. This demo shows Reachy using Gemma to understand its world.

Requirements:
1. Daemon running: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
2. Ollama running with Gemma: ollama serve (with gemma3:27b already pulled)
"""

import base64
import json
import requests
import cv2
import time
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


class GemmaVision:
    """Client for Gemma 3 multimodal vision via Ollama."""

    def __init__(self, model="gemma3:27b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def describe_image(self, image_path, prompt=None):
        """
        Use Gemma to describe an image.

        Args:
            image_path: Path to the image file
            prompt: Optional specific question/instruction
        """
        if prompt is None:
            prompt = "Describe what you see in this image in 2-3 sentences. Be specific about objects, colors, and their positions."

        url = f"{self.base_url}/api/generate"

        # Read and encode image as base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
        }

        try:
            print(f"   🧠 Asking Gemma: {prompt[:60]}...")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.exceptions.Timeout:
            return "⏱️ Gemma took too long to respond (timeout)"
        except Exception as e:
            return f"❌ Error: {e}"


def save_and_show_image(frame, filename, title="Reachy's View"):
    """Save image and display it briefly."""
    cv2.imwrite(filename, frame)
    print(f"   📸 Saved to {filename}")

    # Show image for 2 seconds
    cv2.imshow(title, frame)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    return filename


def main():
    print("=" * 70)
    print("🤖 Reachy Mini + Gemma 3 Vision Demo")
    print("=" * 70)
    print("\nGemma 3 is a multimodal LLM that can see and understand images!")
    print("Let's see what Reachy can observe...\n")

    # Initialize Gemma client
    gemma = GemmaVision(model="gemma3:27b")

    with ReachyMini() as reachy:
        print("✓ Connected to Reachy Mini")

        # Curious expression
        reachy.set_target_antenna_joint_positions([-1.5, 0.0])

        # ===== SCENE 1: Look at the table =====
        print("\n" + "=" * 70)
        print("📷 SCENE 1: Looking at the table")
        print("=" * 70)

        pose = create_head_pose(pitch=-20, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        # Capture and save
        frame = reachy.media.camera.read()
        if frame is None:
            print("❌ Failed to capture image")
            return

        image_path = save_and_show_image(frame, "/tmp/reachy_scene1.jpg", "Scene 1: Table View")

        # Ask Gemma to describe
        print("\n🔍 What does Reachy see?")
        description = gemma.describe_image(image_path)
        print(f"\n   🤖 Reachy: \"{description}\"")

        # Happy expression - robot understood the scene!
        reachy.set_target_antenna_joint_positions([-2.0, 2.0])
        time.sleep(2.0)

        # ===== SCENE 2: Look from the left =====
        print("\n" + "=" * 70)
        print("📷 SCENE 2: Looking from the left side")
        print("=" * 70)

        reachy.set_target_antenna_joint_positions([-1.5, 0.0])  # Curious again
        pose = create_head_pose(yaw=30, pitch=-20, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        frame = reachy.media.camera.read()
        if frame is not None:
            image_path = save_and_show_image(frame, "/tmp/reachy_scene2.jpg", "Scene 2: Left View")

            # Ask a specific question
            question = "What objects can you identify? List them."
            print(f"\n🔍 Question: {question}")
            answer = gemma.describe_image(image_path, question)
            print(f"\n   🤖 Reachy: \"{answer}\"")

            time.sleep(2.0)

        # ===== SCENE 3: Look from the right =====
        print("\n" + "=" * 70)
        print("📷 SCENE 3: Looking from the right side")
        print("=" * 70)

        pose = create_head_pose(yaw=-30, pitch=-20, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        frame = reachy.media.camera.read()
        if frame is not None:
            image_path = save_and_show_image(frame, "/tmp/reachy_scene3.jpg", "Scene 3: Right View")

            question = "Compare this view to what you might see on a typical table. What stands out?"
            print(f"\n🔍 Question: {question}")
            answer = gemma.describe_image(image_path, question)
            print(f"\n   🤖 Reachy: \"{answer}\"")

            time.sleep(2.0)

        # ===== SCENE 4: Closer look =====
        print("\n" + "=" * 70)
        print("📷 SCENE 4: Getting a closer look")
        print("=" * 70)

        # Excited to explore!
        reachy.set_target_antenna_joint_positions([-2.0, 2.0])
        pose = create_head_pose(pitch=-30, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        frame = reachy.media.camera.read()
        if frame is not None:
            image_path = save_and_show_image(frame, "/tmp/reachy_scene4.jpg", "Scene 4: Close-up")

            question = "Describe the textures and materials you can see. Be detailed."
            print(f"\n🔍 Question: {question}")
            answer = gemma.describe_image(image_path, question)
            print(f"\n   🤖 Reachy: \"{answer}\"")

            time.sleep(2.0)

        # ===== SCENE 5: Count objects =====
        print("\n" + "=" * 70)
        print("📷 SCENE 5: Counting objects")
        print("=" * 70)

        # Thinking pose
        reachy.set_target_antenna_joint_positions([-0.5, 0.5])
        pose = create_head_pose(yaw=10, pitch=-20, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        frame = reachy.media.camera.read()
        if frame is not None:
            image_path = save_and_show_image(frame, "/tmp/reachy_scene5.jpg", "Scene 5: Counting")

            question = "Count and list every distinct object you can see. Be thorough."
            print(f"\n🔍 Question: {question}")
            answer = gemma.describe_image(image_path, question)
            print(f"\n   🤖 Reachy: \"{answer}\"")

            time.sleep(2.0)

        # ===== SCENE 6: Creative question =====
        print("\n" + "=" * 70)
        print("📷 SCENE 6: Creative interpretation")
        print("=" * 70)

        pose = create_head_pose(pitch=-15, degrees=True)
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        frame = reachy.media.camera.read()
        if frame is not None:
            image_path = save_and_show_image(frame, "/tmp/reachy_scene6.jpg", "Scene 6: Creative")

            question = "If you were to write a short story about this scene, what would it be about?"
            print(f"\n🔍 Question: {question}")
            answer = gemma.describe_image(image_path, question)
            print(f"\n   🤖 Reachy: \"{answer}\"")

            # Happy with creative answer!
            reachy.set_target_antenna_joint_positions([-2.0, 2.0])
            time.sleep(2.0)

        # ===== Final: Summary =====
        print("\n" + "=" * 70)
        print("📊 FINAL: Comprehensive analysis")
        print("=" * 70)

        # Use the first image for summary
        question = """Based on this image, answer these questions:
1. What is the most prominent object?
2. What colors dominate the scene?
3. What is the spatial arrangement?
4. What might be the purpose of this setup?"""

        print(f"\n🔍 Comprehensive Questions:\n{question}")
        answer = gemma.describe_image("/tmp/reachy_scene1.jpg", question)
        print(f"\n   🤖 Reachy's Analysis:\n   {answer}")

        # Return to neutral
        print("\n" + "=" * 70)
        print("🎬 Demo Complete!")
        print("=" * 70)

        reachy.set_target_antenna_joint_positions([0.0, 0.0])
        pose = create_head_pose()
        reachy.goto_target(head=pose, duration=2.0)
        time.sleep(2.0)

        print("\n📁 Images saved to:")
        print("   /tmp/reachy_scene1.jpg - Table view")
        print("   /tmp/reachy_scene2.jpg - Left view")
        print("   /tmp/reachy_scene3.jpg - Right view")
        print("   /tmp/reachy_scene4.jpg - Close-up")
        print("   /tmp/reachy_scene5.jpg - Counting view")
        print("   /tmp/reachy_scene6.jpg - Creative view")

        print("\n💡 Gemma 3 demonstrated:")
        print("   ✓ Scene description")
        print("   ✓ Object identification")
        print("   ✓ Comparative analysis")
        print("   ✓ Texture/material recognition")
        print("   ✓ Counting and listing")
        print("   ✓ Creative interpretation")
        print("   ✓ Comprehensive analysis")


if __name__ == "__main__":
    main()
