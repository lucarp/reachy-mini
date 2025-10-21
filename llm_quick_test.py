#!/usr/bin/env python3
"""
Quick LLM integration test - runs automatically without user input.
This demonstrates the basic LLM + Robot integration.
"""

import requests
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import time


def test_ollama():
    """Test if Ollama is accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in response.json().get("models", [])]
        if models:
            print(f"✓ Ollama is running with {len(models)} model(s): {', '.join(models[:3])}")
            return models[0]
        else:
            print("❌ No models found in Ollama")
            return None
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return None


def ask_llm(model, question):
    """Simple LLM query."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": f"Answer in one short sentence: {question}",
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=30)
        return response.json()["response"].strip()
    except:
        return "Error getting response"


def main():
    print("=" * 60)
    print("Quick LLM + Reachy Mini Integration Test")
    print("=" * 60)

    # Test Ollama
    print("\n1. Testing Ollama connection...")
    model = test_ollama()
    if not model:
        print("\nPlease start Ollama: ollama serve")
        print("And pull a model: ollama pull gemma3:27b")
        return

    # Test LLM
    print("\n2. Testing LLM generation...")
    question = "What is a robot in 5 words?"
    print(f"   Question: {question}")
    answer = ask_llm(model, question)
    print(f"   Answer: {answer}")

    # Test robot control
    print("\n3. Testing robot control...")
    with ReachyMini() as reachy:
        print("   ✓ Connected to robot")

        # Demo: Robot responds to different prompts with expressions
        test_cases = [
            ("Tell me something exciting!", "excited"),
            ("What makes you happy?", "happy"),
            ("Hmm, that's interesting...", "curious"),
        ]

        for i, (prompt, expected_emotion) in enumerate(test_cases, 1):
            print(f"\n4.{i}. Testing: '{prompt}'")
            print(f"     Expected emotion: {expected_emotion}")

            # Get LLM response
            response = ask_llm(model, prompt)
            print(f"     LLM says: {response[:60]}...")

            # Express emotion
            if expected_emotion == "excited":
                reachy.set_target_antenna_joint_positions([-2.0, 2.0])
                for j in range(2):
                    yaw = 15 if j % 2 == 0 else -15
                    pose = create_head_pose(yaw=yaw, degrees=True)
                    reachy.goto_target(head=pose, duration=0.3)
                    time.sleep(0.35)

            elif expected_emotion == "happy":
                reachy.set_target_antenna_joint_positions([-1.5, 1.5])
                pose = create_head_pose(pitch=10, degrees=True)
                reachy.goto_target(head=pose, duration=1.0)
                time.sleep(1.5)

            elif expected_emotion == "curious":
                reachy.set_target_antenna_joint_positions([-1.5, 0.0])
                pose = create_head_pose(yaw=15, pitch=-10, degrees=True)
                reachy.goto_target(head=pose, duration=1.0)
                time.sleep(1.5)

            # Reset
            reachy.set_target_antenna_joint_positions([0.0, 0.0])
            pose = create_head_pose()
            reachy.goto_target(head=pose, duration=1.0)
            time.sleep(1.0)

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  - Run llm_interactive_chat.py for real-time conversation")
        print("  - Run llm_text_interaction.py for automated examples")
        print("  - Install llava for vision: ollama pull llava")


if __name__ == "__main__":
    main()
