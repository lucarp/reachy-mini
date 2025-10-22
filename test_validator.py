#!/usr/bin/env python3
"""Test ToolValidator functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.tool_validator import ToolValidator

print("=" * 70)
print("🧪 ToolValidator Test Suite")
print("=" * 70)
print()

validator = ToolValidator(strict_mode=True)

# Test 1: Valid tool
print("Test 1: Valid Tool")
print("-" * 70)
valid_code = '''
import numpy as np
from typing import Tuple

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points.

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point

    Returns:
        Euclidean distance between the points
    """
    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx**2 + dy**2)
'''

result = validator.validate(valid_code, "calculate_distance")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
print()

# Test 2: Forbidden import (os)
print("Test 2: Forbidden Import (os)")
print("-" * 70)
invalid_code_import = '''
import os

def delete_files() -> None:
    """Delete all files (DANGEROUS!)."""
    os.system("rm -rf /")
'''

result = validator.validate(invalid_code_import, "delete_files")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print()

# Test 3: Forbidden function (exec)
print("Test 3: Forbidden Function (exec)")
print("-" * 70)
invalid_code_exec = '''
def run_code(code: str) -> None:
    """Execute arbitrary code (DANGEROUS!)."""
    exec(code)
'''

result = validator.validate(invalid_code_exec, "run_code")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print()

# Test 4: Missing type hints
print("Test 4: Missing Type Hints")
print("-" * 70)
invalid_code_types = '''
def add_numbers(a, b):
    """Add two numbers."""
    return a + b
'''

result = validator.validate(invalid_code_types, "add_numbers")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print()

# Test 5: Missing docstring
print("Test 5: Missing Docstring")
print("-" * 70)
invalid_code_docs = '''
def multiply(a: int, b: int) -> int:
    return a * b
'''

result = validator.validate(invalid_code_docs, "multiply")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print()

# Test 6: Subprocess usage
print("Test 6: Forbidden Pattern (subprocess)")
print("-" * 70)
invalid_code_subprocess = '''
import subprocess

def run_command(cmd: str) -> None:
    """Run shell command (DANGEROUS!)."""
    subprocess.run(cmd, shell=True)
'''

result = validator.validate(invalid_code_subprocess, "run_command")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print()

# Test 7: Valid robot control tool
print("Test 7: Valid Robot Control Tool")
print("-" * 70)
valid_robot_code = '''
from reachy_mini import ReachyMini
from typing import Tuple

def get_robot_status(robot: ReachyMini) -> Tuple[bool, str]:
    """Get current robot status.

    Args:
        robot: ReachyMini instance

    Returns:
        Tuple of (is_ready, status_message)
    """
    try:
        joints = robot.get_current_joint_positions()
        if joints:
            return True, "Robot ready"
        return False, "Robot not responding"
    except Exception as e:
        return False, f"Error: {e}"
'''

result = validator.validate(valid_robot_code, "get_robot_status")
print(f"Valid: {result.is_valid}")
print(f"Errors: {result.errors}")
print(f"Warnings: {result.warnings}")
print(f"Functions: {result.metadata.get('functions', [])}")
print()

# Summary
print("=" * 70)
print("✅ ToolValidator Test Complete!")
print("=" * 70)
