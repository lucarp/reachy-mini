#!/usr/bin/env python3
"""
LLM Vision Integration Demo: Robot uses its camera + LLM to describe what it sees.

This demo:
1. Captures image from robot's camera
2. Saves it temporarily
3. Uses Ollama with a vision model to analyze the image
4. Robot responds expressively based on what it sees

Make sure:
1. Daemon is running: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
2. Ollama is running with a vision model: ollama pull llava
"""

import base64
import json
import os
import requests
import cv2
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time


class OllamaVisionClient:
    """Ollama client for vision models."""

    def __init__(self, model="llava", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def analyze_image(self, image_path, prompt):
        """Analyze an image using a vision model."""
        url = f"{self.base_url}/api/generate"

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_data],
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None


def describe_scene(vision_client, image_path):
    """Get a description of what the robot sees."""
    prompt = """Describe what you see in this image in 2-3 sentences.
Be specific about objects, colors, and spatial relationships.
Keep it concise and conversational."""

    return vision_client.analyze_image(image_path, prompt)


def count_objects(vision_client, image_path):
    """Count objects in the scene."""
    prompt = """List all distinct objects you can see in this image.
Format your response as a simple numbered list.
Be specific (e.g., "red apple" not just "fruit")."""

    return vision_client.analyze_image(image_path, prompt)


def answer_question(vision_client, image_path, question):
    """Answer a question about the image."""
    return vision_client.analyze_image(image_path, question)


def main():
    print("=" * 60)
    print("Reachy Mini + LLM Vision Demo")
    print("=" * 60)

    # Check if vision model is available
    print("\nChecking for vision models...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get("models", [])
        vision_models = [m["name"] for m in models if "llava" in m["name"].lower()
                        or "vision" in m["name"].lower()]

        if not vision_models:
            print("❌ No vision model found!")
            print("\nPlease install a vision model:")
            print("  ollama pull llava")
            print("or:")
            print("  ollama pull llava:13b")
            return

        print(f"✓ Found vision model(s): {', '.join(vision_models)}")
        model_name = vision_models[0]

    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    vision_client = OllamaVisionClient(model=model_name)

    with ReachyMini() as reachy_mini:
        print("\n🤖 Robot is ready!")

        # Look at the table
        print("\n1. Looking at the scene...")
        pose = create_head_pose(pitch=-20, degrees=True)
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.5)

        # Capture image
        print("2. Capturing image...")
        frame = reachy_mini.media.camera.read()

        if frame is None:
            print("❌ Failed to capture image")
            return

        # Save temporarily
        temp_image = "/tmp/reachy_view.jpg"
        cv2.imwrite(temp_image, frame)
        print(f"   Saved to {temp_image}")

        # Show image
        cv2.imshow("What Reachy Sees", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        # Curious expression
        reachy_mini.set_target_antenna_joint_positions([-1.5, 0.0])

        # Describe the scene
        print("\n3. Analyzing scene...")
        print("   (This may take 30-60 seconds with vision models...)")
        description = describe_scene(vision_client, temp_image)

        if description:
            print(f"\n   🤖 Robot: I see {description}")

            # Happy expression
            reachy_mini.set_target_antenna_joint_positions([-2.0, 2.0])
            time.sleep(2.0)

        # Look around at different angles
        print("\n4. Looking from different angles...")
        angles = [
            ("left", 30, -15),
            ("right", -30, -15),
            ("closer", 0, -25),
        ]

        for direction, yaw, pitch in angles:
            print(f"\n   Looking {direction}...")
            pose = create_head_pose(yaw=yaw, pitch=pitch, degrees=True)
            reachy_mini.goto_target(head=pose, duration=1.5)
            time.sleep(2.0)

            # Capture and analyze
            frame = reachy_mini.media.camera.read()
            if frame is not None:
                temp_image = f"/tmp/reachy_view_{direction}.jpg"
                cv2.imwrite(temp_image, frame)

                # Ask a specific question
                question = "What is the most prominent object in this view?"
                print(f"   Asking: {question}")
                answer = answer_question(vision_client, temp_image, question)

                if answer:
                    print(f"   🤖 Robot: {answer}")

        # Count objects
        print("\n5. Counting objects...")
        pose = create_head_pose(pitch=-20, degrees=True)
        reachy_mini.goto_target(head=pose, duration=1.5)
        time.sleep(2.0)

        frame = reachy_mini.media.camera.read()
        if frame is not None:
            cv2.imwrite("/tmp/reachy_view_final.jpg", frame)
            objects = count_objects(vision_client, "/tmp/reachy_view_final.jpg")

            if objects:
                print(f"\n   🤖 Robot: I can see:\n{objects}")

        # Happy finish
        print("\n6. Returning to neutral...")
        reachy_mini.set_target_antenna_joint_positions([0.0, 0.0])
        pose = create_head_pose()
        reachy_mini.goto_target(head=pose, duration=2.0)
        time.sleep(2.0)

        print("\n" + "=" * 60)
        print("Vision demo complete!")
        print("=" * 60)
        print(f"\nImages saved in /tmp/reachy_view*.jpg")


if __name__ == "__main__":
    main()
