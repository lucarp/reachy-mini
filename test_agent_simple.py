#!/usr/bin/env python3
"""Simple synchronous test of agent system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.session import SessionManager
from src.tools.robot_tools import set_robot_instance
from src.tools.vision_tools import set_vision_config
from src.agents.runner import ReachyAgentRunner
from reachy_mini import ReachyMini
from agents import Runner

print("=" * 70)
print("🧪 Simple Agent System Test")
print("=" * 70)
print()

# Load config
config = load_config()
print(f"✅ Configuration loaded")
print(f"   LLM: {config.llm.model}")
print()

# Initialize robot
print("🤖 Initializing robot...")
robot = ReachyMini()
print("✅ Robot initialized")
print()

# Set robot instance for tools
set_robot_instance(robot)
set_vision_config(
    robot=robot,
    llm_config={
        "base_url": config.llm.base_url,
        "model": config.llm.model,
    }
)
print("✅ Tools configured")
print()

# Create session manager
session_manager = SessionManager(config.session.db_path)
print("✅ Session manager created")
print()

# Create agent runner
print("🧠 Creating agent runner...")
try:
    runner = ReachyAgentRunner(config, session_manager)
    print(f"✅ Agent runner created")
    print(f"   Coordinator: {runner.coordinator.name}")
    print(f"   Robot Agent: {runner.robot_agent.name}")
    print(f"   Vision Agent: {runner.vision_agent.name}")
    print()
except Exception as e:
    print(f"❌ Failed to create runner: {e}")
    import traceback
    traceback.print_exc()
    robot.__exit__(None, None, None)
    sys.exit(1)

# Test with Runner.run_sync directly
print("💬 Testing direct Runner.run_sync (bypassing runner class)...")
print()

test_message = "Hello! Just say 'hi' back in one word."
print(f"   User: {test_message}")
print(f"   (This may take 10-30 seconds with Gemma 3 27B...)")
print()

try:
    result = Runner.run_sync(
        starting_agent=runner.coordinator,
        input=test_message,
        max_turns=3,
    )

    response = result.final_output if hasattr(result, 'final_output') else str(result)
    print(f"   Assistant: {response}")
    print()
    print("✅ Agent system working!")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
print()
print("🧹 Cleaning up...")
robot.__exit__(None, None, None)
session_manager.clear_session("test")

print()
print("=" * 70)
print("✅ Test complete!")
print("=" * 70)
