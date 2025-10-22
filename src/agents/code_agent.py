"""Code generation agent for creating new tools."""

import logging
from typing import List, Optional, Dict, Any
from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from ..utils.config import Config
from ..validation.tool_validator import ToolValidator
from ..validation.tool_tester import ToolTester, TestCase
from ..validation.tool_registry import ToolRegistry, ToolMetadata

logger = logging.getLogger(__name__)


class CodeGenerationPipeline:
    """Pipeline for generating, validating, testing, and registering tools."""

    def __init__(
        self,
        validator: ToolValidator,
        tester: ToolTester,
        registry: ToolRegistry,
    ):
        """Initialize the pipeline.

        Args:
            validator: Tool validator
            tester: Tool tester
            registry: Tool registry
        """
        self.validator = validator
        self.tester = tester
        self.registry = registry

    async def generate_and_register_tool(
        self,
        name: str,
        code: str,
        description: str,
        test_cases: Optional[List[TestCase]] = None,
    ) -> Dict[str, Any]:
        """Generate, validate, test, and register a tool.

        Args:
            name: Tool name
            code: Generated code
            description: Tool description
            test_cases: Test cases to run

        Returns:
            Dictionary with status and details
        """
        result = {
            "status": "pending",
            "name": name,
            "validation": None,
            "testing": None,
            "version": None,
            "errors": [],
        }

        # Step 1: Validate
        logger.info(f"Validating tool '{name}'...")
        validation_result = self.validator.validate(code, name)
        result["validation"] = {
            "passed": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
        }

        if not validation_result.is_valid:
            result["status"] = "validation_failed"
            result["errors"] = validation_result.errors
            logger.warning(f"Tool '{name}' failed validation")
            return result

        # Step 2: Test (if test cases provided)
        if test_cases:
            logger.info(f"Testing tool '{name}'...")
            test_results = await self.tester.test_tool(code, test_cases, name)

            passed = all(r.passed for r in test_results)
            result["testing"] = {
                "passed": passed,
                "results": [
                    {
                        "test": r.test_name,
                        "passed": r.passed,
                        "error": r.error,
                    }
                    for r in test_results
                ],
            }

            if not passed:
                result["status"] = "testing_failed"
                result["errors"] = [r.error for r in test_results if not r.passed]
                logger.warning(f"Tool '{name}' failed testing")
                return result

        # Step 3: Register
        logger.info(f"Registering tool '{name}'...")
        metadata = ToolMetadata(
            name=name,
            version=1,  # Will be overridden by registry
            source="generated",
            created_at="",  # Will be set by registry
            author="code_agent",
            description=description,
            validation_passed=True,
            test_passed=test_cases is not None,
        )

        version = self.registry.register_tool(name, code, metadata)
        result["version"] = version
        result["status"] = "success"

        logger.info(f"✅ Tool '{name}' v{version} registered successfully")
        return result


# Tool generation functions for the CodeAgent
def create_code_generation_tools(pipeline: CodeGenerationPipeline) -> List:
    """Create tool generation functions.

    Args:
        pipeline: Code generation pipeline

    Returns:
        List of function tools
    """

    @function_tool
    async def generate_simple_tool(
        name: str,
        description: str,
        code: str,
    ) -> str:
        """Generate and register a simple tool without tests.

        Args:
            name: Tool name (snake_case)
            description: Tool description
            code: Python code for the tool

        Returns:
            Success message or error details
        """
        result = await pipeline.generate_and_register_tool(
            name=name,
            code=code,
            description=description,
        )

        if result["status"] == "success":
            return f"✅ Tool '{name}' v{result['version']} registered successfully!"
        else:
            errors = "\n".join(f"  - {e}" for e in result["errors"])
            return f"❌ Failed to register tool '{name}':\n{errors}"

    # Note: generate_tool_with_tests removed due to strict schema limitations
    # Use generate_simple_tool and manually test instead

    @function_tool
    def list_generated_tools() -> str:
        """List all generated tools in the registry.

        Returns:
            List of generated tools
        """
        tools = pipeline.registry.list_tools(source="generated")
        if not tools:
            return "No generated tools found."

        result = "Generated tools:\n"
        for tool in tools:
            result += f"  - {tool.name} v{tool.version}: {tool.description}\n"
            result += f"    Validation: {'✅' if tool.validation_passed else '❌'}\n"
            result += f"    Tests: {'✅' if tool.test_passed else '⏭️'}\n"

        return result

    @function_tool
    def get_tool_code(name: str, version: Optional[int] = None) -> str:
        """Get the code for a generated tool.

        Args:
            name: Tool name
            version: Version number (latest if None)

        Returns:
            Tool code or error message
        """
        code = pipeline.registry.get_tool_code(name, version)
        if code is None:
            return f"❌ Tool '{name}' not found"
        return code

    return [
        generate_simple_tool,
        list_generated_tools,
        get_tool_code,
    ]


def create_code_agent(
    config: Config,
    pipeline: CodeGenerationPipeline,
    handoff_back: Optional[Agent] = None,
) -> Agent:
    """Create the code generation agent.

    Args:
        config: Configuration object
        pipeline: Code generation pipeline
        handoff_back: Handoff to coordinator

    Returns:
        Configured Agent instance
    """
    model = LitellmModel(
        model=f"ollama/{config.llm.model}",
        base_url=config.llm.base_url,
    )

    # Code generation tools
    tools = create_code_generation_tools(pipeline)

    # Build handoffs list
    handoffs = [handoff_back] if handoff_back else []

    # Enhanced instructions for code generation
    instructions = """You are a code generation specialist that creates new Python tools for the robot.

## Your Responsibilities

1. **Generate high-quality Python code** following these requirements:
   - All parameters must have type hints
   - Comprehensive docstrings (Google style)
   - Input validation
   - Error handling
   - Only use whitelisted imports: numpy, typing, reachy_mini, cv2, PIL
   - NEVER use: exec, eval, os, sys, subprocess, file operations

2. **Code Template**:
```python
from typing import [Type1, Type2]
import numpy as np  # if needed

def tool_name(param1: Type1, param2: Type2) -> ReturnType:
    \"\"\"Brief description.

    Detailed description of what this tool does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When validation fails
    \"\"\"
    # Input validation
    if param1 < 0:
        raise ValueError("param1 must be non-negative")

    # Implementation
    result = param1 + param2

    return result
```

3. **Testing**: Provide test cases with diverse inputs including edge cases

4. **Safety**: All code is automatically validated and tested in a sandbox before registration

## Tools Available

- `generate_simple_tool`: For tools without automated tests
- `generate_tool_with_tests`: For tools with test cases (recommended)
- `list_generated_tools`: See what tools have been created
- `get_tool_code`: View code of existing tools

## Example Workflow

User: "Create a tool that calculates the distance between two 3D points"

You:
1. Generate the code following the template
2. Create test cases with various point pairs
3. Call `generate_tool_with_tests` with code and tests
4. Report success or work with validation/test failures to improve

Always provide clear feedback on what was created or what went wrong."""

    # Use config for agent name and merge instructions
    final_instructions = config.agents.code_generation.instructions + "\n\n" + instructions

    agent = Agent(
        name=config.agents.code_generation.name,
        model=model,
        instructions=final_instructions,
        tools=tools,
        handoffs=handoffs,
    )

    logger.info(f"Created code agent: {agent.name} with {len(tools)} tools")
    return agent
