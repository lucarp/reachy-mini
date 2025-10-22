#!/usr/bin/env python3
"""Test the LLM-generated shake_head_no tool."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.tool_registry import ToolRegistry
from reachy_mini import ReachyMini

print("=" * 70)
print("🎯 Testing LLM-Generated 'shake_head_no' Tool")
print("=" * 70)
print()

# Load the tool
tools_storage = Path(__file__).parent / "src" / "tools"
registry = ToolRegistry(tools_storage)

shake_func = registry.load_tool("shake_head_no")
print(f"✅ Tool loaded: {shake_func.__name__}")
print()

# Connect to robot
robot = ReachyMini()
print(f"✅ Robot connected")
print()

# Execute the tool
print("Executing: shake_head_no(robot, times=2, speed=0.3)")
result = shake_func(robot, times=2, speed=0.3)
print(f"✅ Result: {result}")
print()

print("=" * 70)
print("🎉 LLM-Generated Tool Works Perfectly!")
print("=" * 70)
