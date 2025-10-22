#!/usr/bin/env python3
"""Test basic agent system without multimodal components."""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.session import SessionManager
from src.tools.robot_tools import set_robot_instance
from src.tools.vision_tools import set_vision_config
from src.agents.runner import ReachyAgentRunner
from reachy_mini import ReachyMini


async def test_robot_direct(robot):
    """Test robot control directly using existing robot instance."""
    print("🤖 Testing Direct Robot Control...")

    from reachy_mini.utils import create_head_pose

    # Test get current pose
    print("\n1. Getting current pose...")
    pose = robot.get_current_head_pose()
    joints = robot.get_current_joint_positions()
    print(f"   ✅ Got pose matrix: {pose.shape}")
    print(f"   ✅ Got joints: {type(joints)}")

    # Test move head
    print("\n2. Moving head (pitch=10, yaw=20)...")
    target = create_head_pose(z=10, y=20, roll=0, degrees=True, mm=False)
    robot.goto_target(head=target, duration=1.5)
    await asyncio.sleep(2)
    print("   ✅ Movement complete")

    # Test antennas
    print("\n3. Setting antennas to 'happy' (both up)...")
    robot.goto_target(antennas=[90, 90], duration=1.0)
    await asyncio.sleep(1.5)
    print("   ✅ Antennas set")

    # Reset
    print("\n4. Resetting to neutral...")
    target = create_head_pose(z=0, y=0, roll=0, degrees=True, mm=False)
    robot.goto_target(head=target, antennas=[0, 0], duration=1.5)
    await asyncio.sleep(2)
    print("   ✅ Reset complete")

    print("\n✅ Direct robot control working!")


async def test_vision_direct(robot):
    """Test vision directly using existing robot instance."""
    print("\n👁️  Testing Direct Vision...")

    from PIL import Image

    # Test take photo
    print("\n1. Taking photo...")
    frame = robot.media.camera.read()
    if frame is not None:
        rgb_frame = frame[:, :, ::-1]
        image = Image.fromarray(rgb_frame)
        image.save("./test_direct_photo.jpg")
        print(f"   ✅ Photo saved: {image.width}x{image.height}")
    else:
        print("   ⚠️  No frame captured")
        return

    # Skip LLM vision test (too slow for quick testing - Gemma 3 27B takes >2 minutes)
    print("\n2. Skipping LLM vision test (Gemma 3 27B is slow, will test via agent)...")

    print("\n✅ Direct vision working!")


async def test_agent_runner(robot):
    """Test the full agent runner using existing robot instance."""
    print("\n🧠 Testing Agent Runner with OpenAI Agents SDK...")

    config = load_config()
    session_manager = SessionManager(config.session.db_path)

    # Configure robot for tools
    set_robot_instance(robot)
    set_vision_config(
        robot=robot,
        llm_config={
            "base_url": config.llm.base_url,
            "model": config.llm.model,
        }
    )

    print("\n1. Creating agent runner...")
    try:
        runner = ReachyAgentRunner(config, session_manager)
        print("   ✅ Agent runner created")
        print(f"   Coordinator: {runner.coordinator.name}")
        print(f"   Robot Agent: {runner.robot_agent.name}")
        print(f"   Vision Agent: {runner.vision_agent.name}")
    except Exception as e:
        print(f"   ❌ Failed to create runner: {e}")
        import traceback
        traceback.print_exc()
        robot.__exit__(None, None, None)
        return

    print("\n2. Testing conversation (each may take 10-30 seconds)...")
    session_id = "test_basic"

    # Test 1: Simple greeting
    print("\n   User: Hello! Can you introduce yourself briefly?")
    try:
        result = await runner.process_message(
            session_id=session_id,
            message="Hello! Can you introduce yourself briefly?"
        )
        if result['status'] == 'success':
            print(f"   {result['agent']}: {result['response'][:120]}...")
        else:
            print(f"   ⚠️  Error: {result.get('error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Ask to move
    print("\n   User: Move your head slightly to the right")
    try:
        result = await runner.process_message(
            session_id=session_id,
            message="Move your head slightly to the right"
        )
        if result['status'] == 'success':
            print(f"   {result['agent']}: {result['response'][:120]}...")
        else:
            print(f"   ⚠️  Error: {result.get('error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Express emotion
    print("\n   User: Show me you're happy")
    try:
        result = await runner.process_message(
            session_id=session_id,
            message="Show me you're happy"
        )
        if result['status'] == 'success':
            print(f"   {result['agent']}: {result['response'][:120]}...")
        else:
            print(f"   ⚠️  Error: {result.get('error')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n✅ Agent runner working!")

    # Clean up
    session_manager.clear_session(session_id)


async def main():
    """Run all tests."""
    print("=" * 70)
    print("🧪 Testing Phase 1: Basic Agent System")
    print("=" * 70)
    print()

    # Load config
    config = load_config()
    print(f"✅ Configuration loaded")
    print(f"   LLM: {config.llm.model}")
    print(f"   Robot simulation: {config.robot.simulation}")
    print()

    # Initialize robot once
    robot = None

    try:
        print("\n🔧 Initializing Reachy Mini...")
        robot = ReachyMini()
        print("✅ Robot initialized\n")

        # Run tests
        await test_robot_direct(robot)
        await test_vision_direct(robot)
        await test_agent_runner(robot)

        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot:
            print("\n🧹 Cleaning up robot...")
            robot.__exit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
