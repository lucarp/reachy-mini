#!/usr/bin/env python3
"""Integration test for Phase 2: Code generation pipeline."""

import sys
import shutil
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.tool_validator import ToolValidator
from src.validation.tool_tester import ToolTester, TestCase
from src.validation.tool_registry import ToolRegistry
from src.agents.code_agent import CodeGenerationPipeline

print("=" * 70)
print("🧪 Phase 2 Integration Test")
print("=" * 70)
print()

# Setup
test_storage = Path("./test_integration_storage")
if test_storage.exists():
    shutil.rmtree(test_storage)

validator = ToolValidator(strict_mode=True)
tester = ToolTester(default_timeout=5.0)
registry = ToolRegistry(test_storage / "tools")
pipeline = CodeGenerationPipeline(validator, tester, registry)


async def test_successful_tool():
    """Test a valid tool that passes all checks."""
    print("=" * 70)
    print("Test 1: Valid Tool (Full Pipeline)")
    print("=" * 70)
    print()

    code = '''
import numpy as np
from typing import Tuple

def calculate_3d_distance(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float
) -> float:
    """Calculate Euclidean distance between two 3D points.

    Args:
        x1: X coordinate of first point
        y1: Y coordinate of first point
        z1: Z coordinate of first point
        x2: X coordinate of second point
        y2: Y coordinate of second point
        z2: Z coordinate of second point

    Returns:
        Euclidean distance between the points
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return float(np.sqrt(dx**2 + dy**2 + dz**2))
'''

    test_cases = [
        TestCase(
            name="origin_to_1_1_1",
            input={"x1": 0.0, "y1": 0.0, "z1": 0.0, "x2": 1.0, "y2": 1.0, "z2": 1.0},
            expected_output=np.sqrt(3.0),
        ),
        TestCase(
            name="same_point",
            input={"x1": 5.0, "y1": 5.0, "z1": 5.0, "x2": 5.0, "y2": 5.0, "z2": 5.0},
            expected_output=0.0,
        ),
    ]

    result = await pipeline.generate_and_register_tool(
        name="calculate_3d_distance",
        code=code,
        description="Calculate distance between 3D points",
        test_cases=test_cases,
    )

    print(f"Status: {result['status']}")
    print(f"Validation: {result['validation']['passed']}")
    print(f"Testing: {result['testing']['passed'] if result['testing'] else 'N/A'}")
    print(f"Version: {result['version']}")
    print()

    if result['status'] == 'success':
        print("✅ Tool successfully registered!")
        # Try to load and execute
        func = registry.load_tool("calculate_3d_distance")
        if func:
            dist = func(0, 0, 0, 3, 4, 0)
            print(f"   Loaded and tested: distance((0,0,0), (3,4,0)) = {dist}")
            print(f"   Expected: 5.0, Got: {dist}, Match: {dist == 5.0}")
    else:
        print(f"❌ Failed: {result['status']}")
        print(f"   Errors: {result['errors']}")
    print()


async def test_validation_failure():
    """Test a tool that fails validation (forbidden import)."""
    print("=" * 70)
    print("Test 2: Validation Failure (Forbidden Import)")
    print("=" * 70)
    print()

    code = '''
import os

def dangerous_tool() -> None:
    """This should fail validation."""
    os.system("echo 'dangerous'")
'''

    result = await pipeline.generate_and_register_tool(
        name="dangerous_tool",
        code=code,
        description="Dangerous tool (should fail)",
    )

    print(f"Status: {result['status']}")
    print(f"Expected: validation_failed, Got: {result['status']}")
    print(f"Match: {result['status'] == 'validation_failed'}")
    print(f"Errors:")
    for error in result['errors']:
        print(f"  - {error}")
    print()


async def test_testing_failure():
    """Test a tool that passes validation but fails tests."""
    print("=" * 70)
    print("Test 3: Testing Failure (Wrong Implementation)")
    print("=" * 70)
    print()

    code = '''
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    # Bug: adding instead of multiplying!
    return a + b
'''

    test_cases = [
        TestCase(
            name="test_2_times_3",
            input={"a": 2.0, "b": 3.0},
            expected_output=6.0,  # Expects multiplication
        ),
    ]

    result = await pipeline.generate_and_register_tool(
        name="multiply_numbers",
        code=code,
        description="Multiply two numbers",
        test_cases=test_cases,
    )

    print(f"Status: {result['status']}")
    print(f"Expected: testing_failed, Got: {result['status']}")
    print(f"Match: {result['status'] == 'testing_failed'}")
    print(f"Validation passed: {result['validation']['passed']}")
    print(f"Testing passed: {result['testing']['passed']}")
    print(f"Test failures:")
    for test_result in result['testing']['results']:
        if not test_result['passed']:
            print(f"  - {test_result['test']}: {test_result['error']}")
    print()


async def test_version_management():
    """Test tool versioning."""
    print("=" * 70)
    print("Test 4: Version Management")
    print("=" * 70)
    print()

    # Register first version
    code_v1 = '''
def greet(name: str) -> str:
    """Greet someone.

    Args:
        name: Person's name

    Returns:
        Greeting message
    """
    return f"Hello, {name}!"
'''

    result_v1 = await pipeline.generate_and_register_tool(
        name="greet",
        code=code_v1,
        description="Simple greeting",
    )

    print(f"Version 1: {result_v1['version']}")

    # Register second version
    code_v2 = '''
def greet(name: str, enthusiastic: bool = False) -> str:
    """Greet someone with optional enthusiasm.

    Args:
        name: Person's name
        enthusiastic: Add exclamation marks if True

    Returns:
        Greeting message
    """
    if enthusiastic:
        return f"Hello, {name}!!!"
    return f"Hello, {name}!"
'''

    result_v2 = await pipeline.generate_and_register_tool(
        name="greet",
        code=code_v2,
        description="Enhanced greeting with enthusiasm",
    )

    print(f"Version 2: {result_v2['version']}")

    # Check versions
    versions = registry.list_versions("greet")
    print(f"All versions: {versions}")
    print(f"Expected: [1, 2], Got: {versions}, Match: {versions == [1, 2]}")
    print()

    # Load and test both versions
    greet_v1 = registry.load_tool("greet", version=1)
    greet_v2 = registry.load_tool("greet", version=2)

    if greet_v1:
        msg_v1 = greet_v1("World")
        print(f"v1: greet('World') = '{msg_v1}'")

    if greet_v2:
        msg_v2 = greet_v2("World", enthusiastic=True)
        print(f"v2: greet('World', enthusiastic=True) = '{msg_v2}'")
    print()


async def run_all_tests():
    """Run all integration tests."""
    await test_successful_tool()
    await test_validation_failure()
    await test_testing_failure()
    await test_version_management()

    # Summary
    print("=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print()

    all_tools = registry.list_tools()
    print(f"Total tools registered: {len(all_tools)}")
    for tool in all_tools:
        print(f"  - {tool.name} v{tool.version}")
        print(f"    Description: {tool.description}")
        print(f"    Validation: {'✅' if tool.validation_passed else '❌'}")
        print(f"    Tests: {'✅' if tool.test_passed else '⏭️'}")
    print()

    # Cleanup
    print("🧹 Cleaning up test storage...")
    shutil.rmtree(test_storage)

    print()
    print("=" * 70)
    print("✅ Phase 2 Integration Test Complete!")
    print("=" * 70)


# Run async tests
import numpy as np  # Need this for test cases
asyncio.run(run_all_tests())
