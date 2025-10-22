#!/usr/bin/env python3
"""Test ToolTester functionality."""

import sys
import asyncio
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.tool_tester import ToolTester, TestCase

print("=" * 70)
print("🧪 ToolTester Test Suite")
print("=" * 70)
print()

tester = ToolTester(default_timeout=5.0)


async def run_tests():
    """Run all tests."""

    # Test 1: Simple math function
    print("Test 1: Simple Math Function")
    print("-" * 70)
    code1 = '''
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
    return float(np.sqrt(dx**2 + dy**2))
'''

    test_cases1 = [
        TestCase(
            name="distance_origin_to_1_1",
            input={"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            expected_output=np.sqrt(2.0),
        ),
        TestCase(
            name="distance_same_point",
            input={"x1": 5.0, "y1": 5.0, "x2": 5.0, "y2": 5.0},
            expected_output=0.0,
        ),
    ]

    results1 = await tester.test_tool(code1, test_cases1, "calculate_distance")
    for result in results1:
        print(f"  {result.test_name}: {'✅ PASS' if result.passed else '❌ FAIL'}")
        if result.error:
            print(f"    Error: {result.error}")
        print(f"    Output: {result.output}")
        print(f"    Time: {result.execution_time:.3f}s")
    print()

    # Test 2: Robot interaction (mock)
    print("Test 2: Robot Interaction (Mock)")
    print("-" * 70)
    code2 = '''
from typing import Tuple

def get_robot_status(robot) -> Tuple[bool, str]:
    """Get current robot status.

    Args:
        robot: ReachyMini instance

    Returns:
        Tuple of (is_ready, status_message)
    """
    try:
        joints = robot.get_current_joint_positions()
        if joints:
            return True, f"Robot ready with {len(joints)} joints"
        return False, "Robot not responding"
    except Exception as e:
        return False, f"Error: {e}"
'''

    test_cases2 = [
        TestCase(
            name="robot_status_check",
            input={"robot": tester._create_safe_namespace()['robot']},
            expected_output=(True, "Robot ready with 4 joints"),
        ),
    ]

    results2 = await tester.test_tool(code2, test_cases2, "get_robot_status")
    for result in results2:
        print(f"  {result.test_name}: {'✅ PASS' if result.passed else '❌ FAIL'}")
        if result.error:
            print(f"    Error: {result.error}")
        print(f"    Output: {result.output}")
    print()

    # Test 3: Timeout handling
    print("Test 3: Timeout Handling")
    print("-" * 70)
    code3 = '''
import time

def slow_function(delay: float) -> str:
    """A slow function that should timeout.

    Args:
        delay: Sleep duration in seconds

    Returns:
        Success message
    """
    time.sleep(delay)
    return "Completed"
'''

    test_cases3 = [
        TestCase(
            name="timeout_test",
            input={"delay": 10.0},  # Will timeout (default 5s)
            timeout=2.0,  # Override to 2s
        ),
    ]

    results3 = await tester.test_tool(code3, test_cases3, "slow_function")
    for result in results3:
        print(f"  {result.test_name}: {'✅ PASS' if result.passed else '❌ FAIL'}")
        if result.error:
            print(f"    Error: {result.error}")
        print(f"    Expected timeout, got: {result.error}")
    print()

    # Test 4: Error handling
    print("Test 4: Error Handling")
    print("-" * 70)
    code4 = '''
def divide_numbers(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of division
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''

    test_cases4 = [
        TestCase(
            name="valid_division",
            input={"a": 10.0, "b": 2.0},
            expected_output=5.0,
        ),
        TestCase(
            name="divide_by_zero",
            input={"a": 10.0, "b": 0.0},
            should_raise=ValueError,
        ),
    ]

    results4 = await tester.test_tool(code4, test_cases4, "divide_numbers")
    for result in results4:
        print(f"  {result.test_name}: {'✅ PASS' if result.passed else '❌ FAIL'}")
        if result.error:
            print(f"    {result.error}")
        else:
            print(f"    Output: {result.output}")
    print()

    # Summary
    all_results = results1 + results2 + results3 + results4
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)

    print("=" * 70)
    print(f"✅ ToolTester Test Complete: {passed}/{total} tests passed")
    print("=" * 70)


# Run async tests
asyncio.run(run_tests())
