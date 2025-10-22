"""Tool validation system for ensuring generated code safety."""

import ast
import re
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of tool validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class ToolValidator:
    """Validates generated tool code for safety and correctness.

    Uses AST-based analysis to ensure:
    - Only whitelisted imports are used
    - No dangerous operations (exec, eval, subprocess, etc.)
    - Proper function signatures with type hints
    - Comprehensive docstrings
    - No resource-intensive operations
    """

    # Allowed imports (whitelist)
    ALLOWED_IMPORTS = {
        # Core Python
        'typing', 'dataclasses', 'datetime', 'time', 'math', 'json', 'pathlib',
        # Scientific computing
        'numpy', 'np',
        # Computer vision
        'cv2', 'PIL', 'Image',
        # Robot SDK
        'reachy_mini', 'ReachyMini',
        # Async
        'asyncio',
        # HTTP (controlled)
        'requests',
    }

    # Forbidden imports (blacklist)
    FORBIDDEN_IMPORTS = {
        'os', 'sys', 'subprocess', 'shutil', 'multiprocessing', 'threading',
        'socket', 'pickle', 'shelve', '__builtin__', 'builtins', 'importlib',
        'ctypes', 'pty', 'rlcompleter', 'code', 'codeop', 'pdb',
    }

    # Forbidden function calls
    FORBIDDEN_CALLS = {
        'exec', 'eval', 'compile', '__import__', 'open', 'input',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'globals', 'locals', 'vars', 'dir',
    }

    # Forbidden patterns (regex)
    FORBIDDEN_PATTERNS = [
        r'\bexec\s*\(',
        r'\beval\s*\(',
        r'\b__import__\s*\(',
        r'\bcompile\s*\(',
        r'\bopen\s*\(',
        r'\bsubprocess\.',
        r'\bos\.',
        r'\bsys\.',
        r'\b__.*__\s*\(',  # Dunder methods (except in class definitions)
    ]

    def __init__(self, strict_mode: bool = True):
        """Initialize the validator.

        Args:
            strict_mode: If True, enforce strict validation (type hints, docstrings)
        """
        self.strict_mode = strict_mode

    def validate(self, code: str, tool_name: str = "generated_tool") -> ValidationResult:
        """Validate generated tool code.

        Args:
            code: Python code to validate
            tool_name: Name of the tool being validated

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}

        # Step 1: Pattern-based validation
        pattern_errors = self._validate_patterns(code)
        errors.extend(pattern_errors)

        # Step 2: AST-based validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
            )

        # Step 3: Import validation
        import_errors, import_warnings = self._validate_imports(tree)
        errors.extend(import_errors)
        warnings.extend(import_warnings)

        # Step 4: Function call validation
        call_errors = self._validate_calls(tree)
        errors.extend(call_errors)

        # Step 5: Function signature validation
        if self.strict_mode:
            sig_errors, sig_warnings, functions = self._validate_signatures(tree)
            errors.extend(sig_errors)
            warnings.extend(sig_warnings)
            metadata['functions'] = functions

        # Step 6: Docstring validation
        if self.strict_mode:
            doc_errors = self._validate_docstrings(tree)
            errors.extend(doc_errors)

        # Step 7: Resource usage validation
        resource_warnings = self._validate_resource_usage(tree)
        warnings.extend(resource_warnings)

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"✅ Tool '{tool_name}' passed validation")
        else:
            logger.warning(f"❌ Tool '{tool_name}' failed validation: {len(errors)} errors")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def _validate_patterns(self, code: str) -> List[str]:
        """Validate code against forbidden regex patterns.

        Args:
            code: Python code to validate

        Returns:
            List of error messages
        """
        errors = []
        for pattern in self.FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, code, re.MULTILINE)
            if matches:
                errors.append(f"Forbidden pattern detected: {pattern}")
        return errors

    def _validate_imports(self, tree: ast.AST) -> tuple[List[str], List[str]]:
        """Validate imports against whitelist/blacklist.

        Args:
            tree: AST tree

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module in self.FORBIDDEN_IMPORTS:
                        errors.append(f"Forbidden import: {alias.name}")
                    elif module not in self.ALLOWED_IMPORTS:
                        warnings.append(f"Unknown import (not in whitelist): {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module in self.FORBIDDEN_IMPORTS:
                        errors.append(f"Forbidden import from: {node.module}")
                    elif module not in self.ALLOWED_IMPORTS:
                        warnings.append(f"Unknown import from (not in whitelist): {node.module}")

        return errors, warnings

    def _validate_calls(self, tree: ast.AST) -> List[str]:
        """Validate function calls against forbidden list.

        Args:
            tree: AST tree

        Returns:
            List of error messages
        """
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None

                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr

                if func_name in self.FORBIDDEN_CALLS:
                    errors.append(f"Forbidden function call: {func_name}()")

        return errors

    def _validate_signatures(self, tree: ast.AST) -> tuple[List[str], List[str], List[str]]:
        """Validate function signatures (type hints required in strict mode).

        Args:
            tree: AST tree

        Returns:
            Tuple of (errors, warnings, function_names)
        """
        errors = []
        warnings = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)

                # Check for type hints on arguments
                for arg in node.args.args:
                    if arg.annotation is None:
                        errors.append(
                            f"Function '{node.name}': parameter '{arg.arg}' missing type hint"
                        )

                # Check for return type hint
                if node.returns is None and node.name not in ['__init__', '__str__', '__repr__']:
                    warnings.append(f"Function '{node.name}': missing return type hint")

        return errors, warnings, functions

    def _validate_docstrings(self, tree: ast.AST) -> List[str]:
        """Validate docstrings are present.

        Args:
            tree: AST tree

        Returns:
            List of error messages
        """
        errors = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    errors.append(f"Function '{node.name}': missing docstring")
                elif len(docstring.strip()) < 10:
                    errors.append(f"Function '{node.name}': docstring too short")

        return errors

    def _validate_resource_usage(self, tree: ast.AST) -> List[str]:
        """Validate resource usage patterns (warnings only).

        Args:
            tree: AST tree

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for infinite loops (basic detection)
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if condition is a constant True
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    warnings.append("Potential infinite loop detected (while True)")

            if isinstance(node, ast.For):
                # Check for very large ranges
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
                        if len(node.iter.args) > 0:
                            if isinstance(node.iter.args[0], ast.Constant):
                                if node.iter.args[0].value > 1_000_000:
                                    warnings.append(f"Large iteration detected: range({node.iter.args[0].value})")

        return warnings

    def get_function_signatures(self, code: str) -> List[Dict[str, Any]]:
        """Extract function signatures from code.

        Args:
            code: Python code

        Returns:
            List of function signature dictionaries
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        signatures = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sig = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node),
                }
                signatures.append(sig)

        return signatures
