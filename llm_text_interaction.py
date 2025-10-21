#!/usr/bin/env python3
"""
LLM Integration Demo: Text-based conversation with expressive robot responses.
The robot uses LLM sentiment analysis to respond with appropriate movements and expressions.

Make sure:
1. Daemon is running: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
2. Ollama is running: ollama serve
"""

import json
import requests
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time


class OllamaClient:
    """Simple Ollama API client."""

    def __init__(self, model="gemma3:27b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt, system_prompt=None):
        """Generate a response from Ollama."""
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None


class ExpressiveRobot:
    """Robot with LLM-driven expressive behaviors."""

    def __init__(self, reachy_mini, llm_client):
        self.robot = reachy_mini
        self.llm = llm_client

    def analyze_emotion(self, text):
        """Use LLM to analyze the emotional content of text."""
        prompt = f"""Analyze the emotion in this text and respond with ONLY a JSON object (no other text):
{{"emotion": "one of: happy, sad, curious, excited, neutral, thinking", "intensity": 0.0-1.0}}

Text: "{text}"

JSON response:"""

        response = self.llm.generate(prompt)

        # Try to extract JSON from response
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return {"emotion": "neutral", "intensity": 0.5}
        except:
            return {"emotion": "neutral", "intensity": 0.5}

    def express_emotion(self, emotion, intensity=0.5):
        """Move the robot to express an emotion."""
        intensity = max(0.0, min(1.0, intensity))  # Clamp to [0, 1]

        print(f"  🤖 Expressing: {emotion} (intensity: {intensity:.2f})")

        if emotion == "happy":
            # Happy: antennas wide, slight head tilt up
            self.robot.set_target_antenna_joint_positions(
                [-2.0 * intensity, 2.0 * intensity]
            )
            pose = create_head_pose(pitch=10 * intensity, degrees=True)
            self.robot.goto_target(head=pose, duration=1.0)

        elif emotion == "sad":
            # Sad: antennas down, head down
            self.robot.set_target_antenna_joint_positions(
                [0.5 * intensity, -0.5 * intensity]
            )
            pose = create_head_pose(pitch=-15 * intensity, degrees=True)
            self.robot.goto_target(head=pose, duration=1.5)

        elif emotion == "curious":
            # Curious: one antenna up, head tilt
            self.robot.set_target_antenna_joint_positions(
                [-1.5 * intensity, 0.0]
            )
            pose = create_head_pose(
                yaw=15 * intensity,
                pitch=-10 * intensity,
                degrees=True
            )
            self.robot.goto_target(head=pose, duration=1.0)

        elif emotion == "excited":
            # Excited: wiggle with wide antennas
            for i in range(3):
                yaw = 15 * intensity if i % 2 == 0 else -15 * intensity
                self.robot.set_target_antenna_joint_positions(
                    [-2.0 * intensity, 2.0 * intensity]
                )
                pose = create_head_pose(yaw=yaw, degrees=True)
                self.robot.goto_target(head=pose, duration=0.3)
                time.sleep(0.35)

        elif emotion == "thinking":
            # Thinking: look away, one antenna curious
            self.robot.set_target_antenna_joint_positions(
                [-0.5 * intensity, 0.5 * intensity]
            )
            pose = create_head_pose(
                yaw=20 * intensity,
                pitch=-15 * intensity,
                degrees=True
            )
            self.robot.goto_target(head=pose, duration=1.5)

        else:  # neutral
            self.robot.set_target_antenna_joint_positions([0.0, 0.0])
            pose = create_head_pose()
            self.robot.goto_target(head=pose, duration=1.0)

        time.sleep(1.5)

    def respond_to(self, user_message):
        """Generate a response and express it with the robot."""
        print(f"\n👤 User: {user_message}")

        # Generate LLM response
        print("  🧠 Thinking...")
        system_prompt = "You are a friendly, helpful robot assistant. Keep responses concise (1-2 sentences)."
        response = self.llm.generate(user_message, system_prompt=system_prompt)

        if not response:
            print("  ❌ Failed to get LLM response")
            return

        print(f"  💬 Robot: {response}")

        # Analyze emotion of the response
        emotion_data = self.analyze_emotion(response)

        # Express the emotion
        self.express_emotion(
            emotion_data.get("emotion", "neutral"),
            emotion_data.get("intensity", 0.5)
        )

        # Return to neutral
        time.sleep(1.0)
        self.robot.set_target_antenna_joint_positions([0.0, 0.0])
        pose = create_head_pose()
        self.robot.goto_target(head=pose, duration=1.0)


def main():
    print("=" * 60)
    print("Reachy Mini + LLM Interactive Demo")
    print("=" * 60)
    print("\nStarting up...")

    # Initialize
    llm = OllamaClient(model="gemma3:27b")

    with ReachyMini() as reachy_mini:
        robot = ExpressiveRobot(reachy_mini, llm)

        # Predefined conversation examples
        conversations = [
            "Hello! How are you today?",
            "Can you help me understand quantum computing?",
            "I just won a prize! I'm so excited!",
            "I'm feeling a bit down today...",
            "What's your favorite color?",
            "Tell me something interesting about robots!",
        ]

        print("\n🤖 Running conversation examples...\n")

        for message in conversations:
            robot.respond_to(message)
            time.sleep(2.0)

        print("\n" + "=" * 60)
        print("Demo complete!")
        print("=" * 60)
        print("\nTo make this interactive, modify the code to use input():")
        print("  while True:")
        print("    user_input = input('You: ')")
        print("    if user_input.lower() in ['quit', 'exit']: break")
        print("    robot.respond_to(user_input)")


if __name__ == "__main__":
    main()
