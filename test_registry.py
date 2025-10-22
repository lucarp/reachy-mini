#!/usr/bin/env python3
"""Test ToolRegistry functionality."""

import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.validation.tool_registry import ToolRegistry, ToolMetadata

print("=" * 70)
print("🧪 ToolRegistry Test Suite")
print("=" * 70)
print()

# Create test storage directory
test_storage = Path("./test_tool_storage")
if test_storage.exists():
    shutil.rmtree(test_storage)

registry = ToolRegistry(test_storage / "tools")

# Test 1: Register a generated tool
print("Test 1: Register Generated Tool")
print("-" * 70)

tool_code_v1 = '''
def add_numbers(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b
'''

metadata_v1 = ToolMetadata(
    name="add_numbers",
    version=1,
    source="generated",
    created_at="2025-01-01T00:00:00",
    description="Simple addition tool",
    validation_passed=True,
    test_passed=True,
)

v1 = registry.register_tool("add_numbers", tool_code_v1, metadata_v1)
print(f"✅ Registered 'add_numbers' v{v1}")
print()

# Test 2: Register second version
print("Test 2: Register Updated Version")
print("-" * 70)

tool_code_v2 = '''
def add_numbers(a: float, b: float, c: float = 0.0) -> float:
    """Add two or three numbers.

    Args:
        a: First number
        b: Second number
        c: Third number (optional)

    Returns:
        Sum of numbers
    """
    return a + b + c
'''

v2 = registry.register_tool("add_numbers", tool_code_v2)
print(f"✅ Registered 'add_numbers' v{v2}")
print()

# Test 3: List versions
print("Test 3: List Versions")
print("-" * 70)
versions = registry.list_versions("add_numbers")
print(f"Versions: {versions}")
print(f"Expected: [1, 2]")
print(f"Match: {versions == [1, 2]}")
print()

# Test 4: Get latest version code
print("Test 4: Get Latest Version Code")
print("-" * 70)
latest_code = registry.get_tool_code("add_numbers")
print(f"Got code: {len(latest_code)} characters")
print(f"Contains 'c: float = 0.0': {'c: float = 0.0' in latest_code}")
print()

# Test 5: Get specific version
print("Test 5: Get Specific Version Code")
print("-" * 70)
v1_code = registry.get_tool_code("add_numbers", version=1)
print(f"Got v1 code: {len(v1_code)} characters")
print(f"Does NOT contain 'c: float': {'c: float' not in v1_code}")
print()

# Test 6: Load and execute tool
print("Test 6: Load and Execute Tool")
print("-" * 70)
add_func_v2 = registry.load_tool("add_numbers", version=2)
if add_func_v2:
    result = add_func_v2(10, 20, 5)
    print(f"✅ Loaded v2")
    print(f"   add_numbers(10, 20, 5) = {result}")
    print(f"   Expected: 35, Got: {result}, Match: {result == 35}")
else:
    print("❌ Failed to load tool")
print()

add_func_v1 = registry.load_tool("add_numbers", version=1)
if add_func_v1:
    result = add_func_v1(10, 20)
    print(f"✅ Loaded v1")
    print(f"   add_numbers(10, 20) = {result}")
    print(f"   Expected: 30, Got: {result}, Match: {result == 30}")
else:
    print("❌ Failed to load tool")
print()

# Test 7: Get metadata
print("Test 7: Get Metadata")
print("-" * 70)
meta = registry.get_metadata("add_numbers", version=1)
if meta:
    print(f"✅ Got metadata for v1")
    print(f"   Name: {meta.name}")
    print(f"   Version: {meta.version}")
    print(f"   Description: {meta.description}")
    print(f"   Validation: {meta.validation_passed}")
else:
    print("❌ Failed to get metadata")
print()

# Test 8: Update metadata
print("Test 8: Update Metadata")
print("-" * 70)
registry.update_metadata("add_numbers", 2, description="Enhanced addition with 3 params", test_passed=True)
meta_v2 = registry.get_metadata("add_numbers", version=2)
if meta_v2:
    print(f"✅ Updated metadata for v2")
    print(f"   Description: {meta_v2.description}")
    print(f"   Test passed: {meta_v2.test_passed}")
else:
    print("❌ Failed to get metadata")
print()

# Test 9: List all tools
print("Test 9: List All Tools")
print("-" * 70)
all_tools = registry.list_tools()
print(f"Total tools: {len(all_tools)}")
for tool in all_tools:
    print(f"  - {tool.name} v{tool.version} ({tool.source})")
print()

# Test 10: List generated tools only
print("Test 10: List Generated Tools Only")
print("-" * 70)
generated_tools = registry.list_tools(source="generated")
print(f"Generated tools: {len(generated_tools)}")
for tool in generated_tools:
    print(f"  - {tool.name} v{tool.version}")
print()

# Test 11: Delete a version
print("Test 11: Delete Tool Version")
print("-" * 70)
success = registry.delete_tool("add_numbers", 1)
print(f"Deleted v1: {success}")
remaining_versions = registry.list_versions("add_numbers")
print(f"Remaining versions: {remaining_versions}")
print(f"Expected: [2], Got: {remaining_versions}, Match: {remaining_versions == [2]}")
print()

# Cleanup
print("=" * 70)
print("🧹 Cleaning up test storage...")
shutil.rmtree(test_storage)
print("=" * 70)
print("✅ ToolRegistry Test Complete!")
print("=" * 70)
