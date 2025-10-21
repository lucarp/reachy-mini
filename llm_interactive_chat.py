#!/usr/bin/env python3
"""
Interactive Chat Demo: Real-time conversation with Reachy Mini powered by LLM.

The robot:
- Listens to your text input
- Generates responses using Ollama
- Expresses emotions through movement
- Maintains conversation context

Make sure:
1. Daemon is running: mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
2. Ollama is running: ollama serve

Type 'quit' or 'exit' to end the conversation.
"""

import json
import requests
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time
from datetime import datetime


class ConversationalRobot:
    """Interactive conversational robot with memory and emotions."""

    def __init__(self, reachy_mini, model="gemma3:27b"):
        self.robot = reachy_mini
        self.model = model
        self.base_url = "http://localhost:11434"
        self.conversation_history = []

        # System prompt defines robot personality
        self.system_prompt = """You are Reachy Mini, a friendly and curious small robot.
You have a camera, antennas that express emotions, and an expressive head.
You love learning about the world and talking to humans.
Keep responses concise (1-3 sentences) and friendly.
Express emotions naturally in your responses."""

    def generate_response(self, user_message):
        """Generate LLM response with conversation context."""
        url = f"{self.base_url}/api/generate"

        # Build context from history
        context = f"{self.system_prompt}\n\n"
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            context += f"User: {msg['user']}\nRobot: {msg['robot']}\n"
        context += f"User: {user_message}\nRobot:"

        payload = {
            "model": self.model,
            "prompt": context,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            return f"Sorry, I had trouble thinking just now: {e}"

    def detect_emotion(self, text):
        """Simple keyword-based emotion detection."""
        text_lower = text.lower()

        # Excitement keywords
        if any(word in text_lower for word in ['!', 'amazing', 'awesome', 'great', 'wow', 'excited', 'fantastic']):
            return 'excited', 0.8

        # Happy keywords
        if any(word in text_lower for word in ['happy', 'glad', 'good', 'nice', 'wonderful', 'love', '😊']):
            return 'happy', 0.7

        # Sad keywords
        if any(word in text_lower for word in ['sad', 'sorry', 'unfortunately', 'bad', 'wrong', 'problem', '😢']):
            return 'sad', 0.6

        # Curious/questioning
        if '?' in text or any(word in text_lower for word in ['how', 'what', 'why', 'when', 'where', 'wonder']):
            return 'curious', 0.7

        # Thinking
        if any(word in text_lower for word in ['think', 'maybe', 'perhaps', 'consider', 'might']):
            return 'thinking', 0.6

        return 'neutral', 0.5

    def express(self, emotion, intensity=0.5):
        """Express emotion through robot movement."""
        intensity = max(0.0, min(1.0, intensity))

        if emotion == 'excited':
            # Quick wiggle
            for i in range(2):
                yaw = 15 * intensity if i % 2 == 0 else -15 * intensity
                self.robot.set_target_antenna_joint_positions([-2.0 * intensity, 2.0 * intensity])
                pose = create_head_pose(yaw=yaw, degrees=True)
                self.robot.goto_target(head=pose, duration=0.2)
                time.sleep(0.25)

        elif emotion == 'happy':
            self.robot.set_target_antenna_joint_positions([-1.5 * intensity, 1.5 * intensity])
            pose = create_head_pose(pitch=8 * intensity, degrees=True)
            self.robot.goto_target(head=pose, duration=0.8)

        elif emotion == 'sad':
            self.robot.set_target_antenna_joint_positions([0.4 * intensity, -0.4 * intensity])
            pose = create_head_pose(pitch=-12 * intensity, degrees=True)
            self.robot.goto_target(head=pose, duration=1.2)

        elif emotion == 'curious':
            self.robot.set_target_antenna_joint_positions([-1.2 * intensity, 0.2])
            pose = create_head_pose(yaw=12 * intensity, pitch=-8 * intensity, degrees=True)
            self.robot.goto_target(head=pose, duration=0.8)

        elif emotion == 'thinking':
            self.robot.set_target_antenna_joint_positions([-0.5 * intensity, 0.5 * intensity])
            pose = create_head_pose(yaw=15 * intensity, pitch=-10 * intensity, degrees=True)
            self.robot.goto_target(head=pose, duration=1.0)

        else:  # neutral
            self.robot.set_target_antenna_joint_positions([0.0, 0.0])
            pose = create_head_pose()
            self.robot.goto_target(head=pose, duration=0.8)

        time.sleep(1.0)

    def respond(self, user_message):
        """Generate and express a response."""
        # Show thinking
        print("  🤔 Thinking...", end='', flush=True)
        self.express('thinking', 0.4)

        # Generate response
        response = self.generate_response(user_message)
        print("\r  " + " " * 20 + "\r", end='')  # Clear "Thinking..."

        # Detect emotion and express it
        emotion, intensity = self.detect_emotion(response)

        # Store in history
        self.conversation_history.append({
            'user': user_message,
            'robot': response,
            'emotion': emotion,
            'timestamp': datetime.now().isoformat()
        })

        # Express emotion
        self.express(emotion, intensity)

        return response, emotion


def main():
    print("=" * 60)
    print("🤖 Interactive Chat with Reachy Mini")
    print("=" * 60)
    print("\nInitializing...")

    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in response.json().get("models", [])]

        if not models:
            print("❌ No Ollama models found!")
            print("Install one: ollama pull gemma3:27b")
            return

        # Prefer faster models for interactive chat
        preferred = ["gemma3:27b", "deepseek-r1:8b", "llama3", "gemma"]
        model = next((m for p in preferred for m in models if p in m), models[0])

        print(f"✓ Using model: {model}")

    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        return

    with ReachyMini() as reachy_mini:
        robot = ConversationalRobot(reachy_mini, model=model)

        # Initial greeting
        print("\n" + "=" * 60)
        robot.express('happy', 0.7)
        print("🤖 Reachy: Hi! I'm Reachy Mini. Let's chat!")
        print("   (Type 'quit' or 'exit' to end)")
        print("=" * 60)

        while True:
            # Get user input
            print()
            try:
                user_input = input("👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                # Say goodbye
                robot.express('sad', 0.5)
                print("🤖 Reachy: Goodbye! It was nice chatting with you!")
                time.sleep(2.0)
                break

            # Generate and display response
            response, emotion = robot.respond(user_input)
            print(f"🤖 Reachy [{emotion}]: {response}")

        # Return to neutral
        robot.express('neutral', 0.5)

        # Show conversation stats
        print("\n" + "=" * 60)
        print(f"Conversation Summary:")
        print(f"  Messages exchanged: {len(robot.conversation_history)}")
        if robot.conversation_history:
            emotions = [msg['emotion'] for msg in robot.conversation_history]
            print(f"  Most common emotion: {max(set(emotions), key=emotions.count)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
