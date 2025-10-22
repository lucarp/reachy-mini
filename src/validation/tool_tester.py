"""Tool testing system with sandboxed execution."""

import asyncio
import logging
import traceback
import io
import sys
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case for a tool."""

    name: str
    input: Dict[str, Any]
    expected_output: Optional[Any] = None
    should_raise: Optional[type] = None
    timeout: float = 10.0


@dataclass
class TestResult:
    """Result of tool testing."""

    passed: bool
    test_name: str
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    stdout: str = ""
    stderr: str = ""


class MockReachyMini:
    """Mock robot for safe testing without hardware.

    Simulates ReachyMini interface for testing generated tools.
    """

    def __init__(self):
        """Initialize mock robot."""
        self._head_pose = np.eye(4)  # 4x4 identity matrix
        self._joint_positions = (0.0, 0.0, 0.0, 0.0)  # (pitch, yaw, left_antenna, right_antenna)
        self._antennas = [0.0, 0.0]
        self._camera_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def get_current_head_pose(self) -> np.ndarray:
        """Get current head pose.

        Returns:
            4x4 transformation matrix
        """
        return self._head_pose.copy()

    def get_current_joint_positions(self) -> tuple:
        """Get current joint positions.

        Returns:
            Tuple of (pitch, yaw, left_antenna, right_antenna)
        """
        return self._joint_positions

    def goto_target(
        self,
        head: Optional[Dict[str, float]] = None,
        antennas: Optional[List[float]] = None,
        duration: float = 1.0,
        wait: bool = True,
    ) -> bool:
        """Move robot to target position.

        Args:
            head: Head target angles (pitch, yaw, roll)
            antennas: Antenna angles [left, right]
            duration: Movement duration
            wait: Wait for completion

        Returns:
            True if successful
        """
        if head:
            # Simulate head movement
            pitch = head.get('pitch', 0.0)
            yaw = head.get('yaw', 0.0)
            roll = head.get('roll', 0.0)
            self._joint_positions = (pitch, yaw, self._joint_positions[2], self._joint_positions[3])

        if antennas:
            # Simulate antenna movement
            if len(antennas) == 2:
                self._antennas = antennas
                self._joint_positions = (
                    self._joint_positions[0],
                    self._joint_positions[1],
                    antennas[0],
                    antennas[1],
                )

        return True

    def read_camera(self) -> tuple[bool, np.ndarray]:
        """Read camera frame (mock).

        Returns:
            Tuple of (success, frame)
        """
        return True, self._camera_frame.copy()

    class camera:
        """Mock camera object."""

        @staticmethod
        def read() -> tuple[bool, np.ndarray]:
            """Read camera frame.

            Returns:
                Tuple of (success, frame)
            """
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)


class ToolTester:
    """Tests generated tools in a sandboxed environment.

    Features:
    - Isolated execution namespace
    - Timeout enforcement
    - Mock robot objects
    - Stdout/stderr capture
    - Test case validation
    """

    def __init__(self, default_timeout: float = 10.0):
        """Initialize the tester.

        Args:
            default_timeout: Default timeout in seconds for tests
        """
        self.default_timeout = default_timeout

    async def test_tool(
        self,
        code: str,
        test_cases: List[TestCase],
        tool_name: str = "generated_tool",
    ) -> List[TestResult]:
        """Test a tool with multiple test cases.

        Args:
            code: Python code defining the tool
            test_cases: List of test cases to run
            tool_name: Name of the tool being tested

        Returns:
            List of TestResult objects
        """
        results = []

        # Create safe namespace
        namespace = self._create_safe_namespace()

        # Track names before execution to identify new definitions
        names_before = set(namespace.keys())

        # Execute code to define functions
        try:
            exec(code, namespace)
        except Exception as e:
            logger.error(f"Failed to execute tool code: {e}")
            return [
                TestResult(
                    passed=False,
                    test_name="code_execution",
                    error=f"Failed to execute code: {str(e)}",
                )
            ]

        # Find newly defined names (these are the tool functions)
        new_names = set(namespace.keys()) - names_before

        # Run each test case
        for test_case in test_cases:
            result = await self._run_test_case(namespace, test_case, tool_name, new_names)
            results.append(result)

        return results

    async def _run_test_case(
        self,
        namespace: Dict[str, Any],
        test_case: TestCase,
        tool_name: str,
        new_names: set,
    ) -> TestResult:
        """Run a single test case.

        Args:
            namespace: Execution namespace
            test_case: Test case to run
            tool_name: Name of the tool

        Returns:
            TestResult
        """
        import time

        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        start_time = time.time()
        result = TestResult(passed=False, test_name=test_case.name)

        try:
            # Find the function to test from newly defined names
            func = None
            import types
            import typing

            for name in new_names:
                if name.startswith('_'):
                    continue
                obj = namespace[name]

                # Skip types, classes, and modules
                if isinstance(obj, type):
                    continue
                if isinstance(obj, types.ModuleType):
                    continue
                # Skip typing generics
                if hasattr(typing, '__all__') and name in typing.__all__:
                    continue
                if hasattr(obj, '__module__') and obj.__module__ == 'typing':
                    continue

                # Check if it's a function
                if callable(obj):
                    func = obj
                    break

            if func is None:
                raise ValueError(f"No callable function found in tool code. New names: {new_names}")

            # Execute with timeout
            output = await asyncio.wait_for(
                self._execute_function(func, test_case.input),
                timeout=test_case.timeout,
            )

            # Validate output
            if test_case.should_raise:
                result.passed = False
                result.error = f"Expected {test_case.should_raise.__name__} to be raised"
            elif test_case.expected_output is not None:
                if output == test_case.expected_output:
                    result.passed = True
                    result.output = output
                else:
                    result.passed = False
                    result.error = f"Expected {test_case.expected_output}, got {output}"
            else:
                # No expected output, just check it didn't crash
                result.passed = True
                result.output = output

        except asyncio.TimeoutError:
            result.passed = False
            result.error = f"Timeout after {test_case.timeout}s"

        except Exception as e:
            if test_case.should_raise and isinstance(e, test_case.should_raise):
                result.passed = True
                result.error = f"Correctly raised {type(e).__name__}"
            else:
                result.passed = False
                result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            result.execution_time = time.time() - start_time
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

        return result

    async def _execute_function(self, func: Callable, inputs: Dict[str, Any]) -> Any:
        """Execute a function with given inputs.

        Args:
            func: Function to execute
            inputs: Dictionary of parameter name -> value

        Returns:
            Function output
        """
        # Check if function is async
        if asyncio.iscoroutinefunction(func):
            return await func(**inputs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**inputs))

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Create a safe execution namespace with allowed imports.

        Returns:
            Namespace dictionary
        """
        # Import builtins to get the real __import__
        import builtins

        namespace = {
            '__builtins__': {
                # Safe builtins only
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round,
                'sorted': sorted,
                'enumerate': enumerate,
                'zip': zip,
                'isinstance': isinstance,
                'issubclass': issubclass,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'RuntimeError': RuntimeError,
                'Exception': Exception,
                # Required for imports
                '__import__': builtins.__import__,
                '__name__': '__main__',
                '__doc__': None,
            },
            # Mock robot
            'ReachyMini': MockReachyMini,
            'robot': MockReachyMini(),
            # Safe libraries
            'numpy': np,
            'np': np,
        }

        return namespace

    def create_simple_test_cases(
        self,
        function_name: str,
        input_output_pairs: List[tuple[Dict[str, Any], Any]],
    ) -> List[TestCase]:
        """Create simple test cases from input/output pairs.

        Args:
            function_name: Name of the function
            input_output_pairs: List of (input_dict, expected_output) tuples

        Returns:
            List of TestCase objects
        """
        test_cases = []
        for i, (inputs, expected) in enumerate(input_output_pairs):
            test_cases.append(
                TestCase(
                    name=f"{function_name}_test_{i+1}",
                    input=inputs,
                    expected_output=expected,
                )
            )
        return test_cases
