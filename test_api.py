#!/usr/bin/env python3
"""Quick test script for Phase 1 API."""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    print()


def test_chat(session_id="test_session"):
    """Test chat endpoint."""
    print("💬 Testing chat endpoint...")

    payload = {
        "session_id": session_id,
        "message": "Hello! Can you introduce yourself?"
    }

    response = requests.post(
        f"{BASE_URL}/api/chat/message",
        json=payload
    )

    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Agent: {data.get('agent')}")
        print(f"   Response: {data.get('response')}")
    else:
        print(f"   Error: {response.text}")
    print()


def test_robot_control():
    """Test robot control endpoints."""
    print("🤖 Testing robot control...")

    # Test get current pose
    print("   Getting current pose...")
    response = requests.get(f"{BASE_URL}/api/robot/current_pose")
    if response.status_code == 200:
        print(f"   Current pose: {response.json()}")
    else:
        print(f"   Error: {response.text}")

    # Test express emotion
    print("   Expressing happiness...")
    response = requests.post(
        f"{BASE_URL}/api/robot/express_emotion",
        json={"emotion": "happy"}
    )
    if response.status_code == 200:
        print(f"   Result: {response.json()['status']}")
    else:
        print(f"   Error: {response.text}")

    time.sleep(2)

    # Test move head
    print("   Moving head...")
    response = requests.post(
        f"{BASE_URL}/api/robot/move_head",
        json={"pitch": 10, "yaw": 20, "roll": 0, "duration": 1.5}
    )
    if response.status_code == 200:
        print(f"   Result: {response.json()['status']}")
    else:
        print(f"   Error: {response.text}")

    print()


def test_vision():
    """Test vision endpoints."""
    print("👁️  Testing vision endpoints...")

    # Test take photo
    print("   Taking photo...")
    response = requests.post(
        f"{BASE_URL}/api/vision/take_photo",
        json={"save_path": "./test_photo.jpg"}
    )
    if response.status_code == 200:
        print(f"   Photo taken: {response.json()}")
    else:
        print(f"   Error: {response.text}")

    # Test analyze scene
    print("   Analyzing scene...")
    response = requests.post(
        f"{BASE_URL}/api/vision/analyze_scene",
        json={"question": "What objects can you see?"}
    )
    if response.status_code == 200:
        print(f"   Analysis: {response.json().get('analysis', 'N/A')[:100]}...")
    else:
        print(f"   Error: {response.text}")

    print()


def test_sessions():
    """Test session endpoints."""
    print("📋 Testing session management...")

    # List sessions
    response = requests.get(f"{BASE_URL}/api/sessions/list")
    if response.status_code == 200:
        sessions = response.json()["sessions"]
        print(f"   Total sessions: {len(sessions)}")
        if sessions:
            print(f"   Latest session: {sessions[0]['session_id']}")
    else:
        print(f"   Error: {response.text}")

    print()


def test_conversational_flow():
    """Test a conversational flow with multiple interactions."""
    print("🔄 Testing conversational flow...")

    session_id = f"conv_test_{int(time.time())}"

    messages = [
        "Hello! Can you see me?",
        "What do you see in front of you?",
        "Can you express happiness?",
        "Now look to your left",
    ]

    for i, message in enumerate(messages, 1):
        print(f"\n   [{i}] User: {message}")

        response = requests.post(
            f"{BASE_URL}/api/chat/message",
            json={"session_id": session_id, "message": message}
        )

        if response.status_code == 200:
            data = response.json()
            print(f"   [{i}] {data['agent']}: {data['response'][:100]}...")
        else:
            print(f"   [{i}] Error: {response.text}")

        time.sleep(1)

    # Get history
    print(f"\n   Getting conversation history...")
    response = requests.get(f"{BASE_URL}/api/sessions/{session_id}/history")
    if response.status_code == 200:
        history = response.json()["history"]
        print(f"   Total messages in history: {len(history)}")

    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 Testing Reachy Mini Agentic AI - Phase 1")
    print("=" * 60)
    print()

    try:
        test_health()
        test_chat()
        test_robot_control()
        test_vision()
        test_sessions()
        test_conversational_flow()

        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server.")
        print("   Make sure the server is running: python -m src.main")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
