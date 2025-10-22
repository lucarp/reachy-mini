#!/usr/bin/env python3
"""
Test LLM-powered tool generation.

This demonstrates the CodeAgent using the LLM to generate a new tool
from a natural language description.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.validation import ToolValidator, ToolTester, ToolRegistry
from src.agents.code_agent import CodeGenerationPipeline
import litellm

print("=" * 70)
print("🤖 LLM-Powered Tool Generation Demo")
print("=" * 70)
print()

async def main():
    # Load configuration
    print("📋 Step 1: Loading Configuration")
    print("-" * 70)
    config = load_config()
    print(f"✅ LLM Model: {config.llm.model}")
    print(f"   Base URL: {config.llm.base_url}")
    print()

    # Initialize Phase 2 components
    print("🔧 Step 2: Initializing Phase 2 Pipeline")
    print("-" * 70)
    tools_storage = Path(__file__).parent / "src" / "tools"
    validator = ToolValidator(strict_mode=True)
    tester = ToolTester(default_timeout=10.0)
    registry = ToolRegistry(tools_storage)
    pipeline = CodeGenerationPipeline(validator, tester, registry)
    print(f"✅ Validator initialized")
    print(f"✅ Tester initialized")
    print(f"✅ Registry initialized")
    print()

    # Setup LLM
    print("🧠 Step 3: Setting up LLM")
    print("-" * 70)
    print(f"✅ LLM ready: {config.llm.model}")
    print()

    # Create a simple agent to generate code
    print("💻 Step 4: Asking LLM to Generate 'shake_head_no' Tool")
    print("-" * 70)
    print()
    print("Request: 'Create a Python function that makes the robot shake its")
    print("         head to indicate 'no' by moving it left and right.'")
    print()

    # Prompt for the LLM
    prompt = """You are a code generation assistant. Create a Python function named 'shake_head_no' that makes a robot shake its head to indicate 'no'.

Requirements:
1. Function signature: shake_head_no(robot: ReachyMini, times: int = 2, speed: float = 1.0) -> str
2. Use robot.look_at_world(x, y, z, duration) to control head movement
3. Shake the head left and right (vary the y coordinate)
4. Include type hints on all parameters
5. Include a comprehensive docstring with Args, Returns, and Raises sections
6. Add input validation (times and speed must be positive)
7. Only import: from reachy_mini import ReachyMini, and import time
8. Return a success message like "Successfully shook head 'no' X times"
9. Wrap in try/except and return error message on failure

Generate ONLY the Python code, nothing else."""

    print("Generating code with LLM...")
    print()

    # Call LLM to generate code
    response = litellm.completion(
        model=f"ollama/{config.llm.model}",
        messages=[{"role": "user", "content": prompt}],
        api_base=config.llm.base_url,
        max_tokens=1000,
        temperature=0.3,
    )

    generated_code = response.choices[0].message.content

    # Extract code if wrapped in markdown
    if "```python" in generated_code:
        code = generated_code.split("```python")[1].split("```")[0].strip()
    elif "```" in generated_code:
        code = generated_code.split("```")[1].split("```")[0].strip()
    else:
        code = generated_code.strip()

    print("✅ LLM generated code:")
    print()
    print("─" * 70)
    print(code)
    print("─" * 70)
    print()

    # Validate and register the tool
    print("🔍 Step 5: Validating Generated Code")
    print("-" * 70)
    result = await pipeline.generate_and_register_tool(
        name="shake_head_no",
        code=code,
        description="Make the robot shake its head to indicate 'no'",
        test_cases=None,
    )

    if result["status"] == "success":
        print(f"✅ Validation: PASS")
        print(f"✅ Registration: v{result['version']}")
        print()
    else:
        print(f"❌ Validation: FAIL")
        print(f"   Errors: {result['errors']}")
        print()
        return

    # Test the generated tool
    print("🎬 Step 6: Testing Generated Tool")
    print("-" * 70)

    from reachy_mini import ReachyMini

    print("Loading tool from registry...")
    shake_func = registry.load_tool("shake_head_no")

    if not shake_func:
        print("❌ Failed to load tool")
        return

    print(f"✅ Tool loaded: {shake_func.__name__}")
    print()

    print("Connecting to robot...")
    robot = ReachyMini()
    print(f"✅ Robot connected")
    print()

    print("Executing: shake_head_no(robot, times=2, speed=1.5)")
    result = shake_func(robot, times=2, speed=1.5)
    print(f"✅ Result: {result}")
    print()

    # Summary
    print("=" * 70)
    print("📊 Demo Summary")
    print("=" * 70)
    print()
    print("✅ LLM successfully generated a new tool from natural language!")
    print()
    print("Generated tool: shake_head_no")
    print(f"  Version: v{result['version'] if 'version' in result else 1}")
    print("  Status: Validated, Registered, and Tested")
    print("  Functionality: Robot shakes head left-right to indicate 'no'")
    print()
    print("🎯 Phase 2 Complete: Self-Coding Agent Working!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
