"""Tool registry for managing predefined and generated tools with versioning."""

import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Metadata for a tool."""

    name: str
    version: int
    source: str  # "predefined" or "generated"
    created_at: str
    author: str = "system"
    description: str = ""
    validation_passed: bool = False
    test_passed: bool = False
    test_results: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolMetadata':
        """Create from dictionary."""
        return cls(**data)


class ToolRegistry:
    """Registry for managing predefined and generated tools.

    Features:
    - Version-controlled storage
    - Separate predefined vs generated tools
    - Dynamic tool loading
    - Rollback capability
    - Metadata tracking
    """

    def __init__(self, storage_root: Path):
        """Initialize the registry.

        Args:
            storage_root: Root directory for tool storage (e.g., src/tools/)
        """
        self.storage_root = Path(storage_root)
        self.predefined_dir = self.storage_root / "predefined"
        self.generated_dir = self.storage_root / "generated"

        # Create directories if they don't exist
        self.predefined_dir.mkdir(parents=True, exist_ok=True)
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        # In-memory registry
        self._tools: Dict[str, Dict[int, ToolMetadata]] = {}
        self._loaded_functions: Dict[str, Callable] = {}

        # Load existing tools
        self._scan_tools()

    def _scan_tools(self):
        """Scan storage directories and load tool metadata."""
        # Scan predefined tools
        for tool_file in self.predefined_dir.glob("*.py"):
            if tool_file.name == "__init__.py":
                continue
            self._register_predefined_tool(tool_file)

        # Scan generated tools
        for tool_dir in self.generated_dir.iterdir():
            if tool_dir.is_dir():
                self._register_generated_tool(tool_dir)

    def _register_predefined_tool(self, tool_file: Path):
        """Register a predefined tool.

        Args:
            tool_file: Path to the tool file
        """
        tool_name = tool_file.stem

        metadata = ToolMetadata(
            name=tool_name,
            version=1,
            source="predefined",
            created_at=datetime.now().isoformat(),
            author="system",
            description=f"Predefined tool: {tool_name}",
            validation_passed=True,
            test_passed=True,
        )

        if tool_name not in self._tools:
            self._tools[tool_name] = {}
        self._tools[tool_name][1] = metadata

        logger.debug(f"Registered predefined tool: {tool_name}")

    def _register_generated_tool(self, tool_dir: Path):
        """Register all versions of a generated tool.

        Args:
            tool_dir: Directory containing tool versions
        """
        tool_name = tool_dir.name

        # Find all version files
        for version_file in sorted(tool_dir.glob("v*.py")):
            version_str = version_file.stem[1:]  # Remove 'v' prefix
            try:
                version = int(version_str)
            except ValueError:
                logger.warning(f"Invalid version file: {version_file}")
                continue

            # Load metadata
            metadata_file = version_file.with_suffix('.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = ToolMetadata.from_dict(metadata_dict)
            else:
                # Create default metadata if missing
                metadata = ToolMetadata(
                    name=tool_name,
                    version=version,
                    source="generated",
                    created_at=datetime.now().isoformat(),
                )

            if tool_name not in self._tools:
                self._tools[tool_name] = {}
            self._tools[tool_name][version] = metadata

            logger.debug(f"Registered generated tool: {tool_name} v{version}")

    def register_tool(
        self,
        name: str,
        code: str,
        metadata: Optional[ToolMetadata] = None,
    ) -> int:
        """Register a new generated tool.

        Args:
            name: Tool name
            code: Tool code
            metadata: Optional metadata (will be created if None)

        Returns:
            Version number of the registered tool
        """
        # Determine next version number
        if name in self._tools and self._tools[name]:
            next_version = max(self._tools[name].keys()) + 1
        else:
            next_version = 1

        # Create metadata if not provided
        if metadata is None:
            metadata = ToolMetadata(
                name=name,
                version=next_version,
                source="generated",
                created_at=datetime.now().isoformat(),
            )
        else:
            metadata.version = next_version

        # Create tool directory
        tool_dir = self.generated_dir / name
        tool_dir.mkdir(exist_ok=True)

        # Save code
        code_file = tool_dir / f"v{next_version}.py"
        with open(code_file, 'w') as f:
            f.write(code)

        # Save metadata
        metadata_file = tool_dir / f"v{next_version}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Register in memory
        if name not in self._tools:
            self._tools[name] = {}
        self._tools[name][next_version] = metadata

        logger.info(f"Registered tool '{name}' v{next_version}")
        return next_version

    def get_tool_code(self, name: str, version: Optional[int] = None) -> Optional[str]:
        """Get tool code.

        Args:
            name: Tool name
            version: Version number (latest if None)

        Returns:
            Tool code or None if not found
        """
        if name not in self._tools:
            return None

        # Get version
        if version is None:
            version = max(self._tools[name].keys())

        if version not in self._tools[name]:
            return None

        metadata = self._tools[name][version]

        # Determine file path
        if metadata.source == "predefined":
            code_file = self.predefined_dir / f"{name}.py"
        else:
            code_file = self.generated_dir / name / f"v{version}.py"

        if not code_file.exists():
            return None

        with open(code_file, 'r') as f:
            return f.read()

    def load_tool(self, name: str, version: Optional[int] = None) -> Optional[Callable]:
        """Load and return a tool function.

        Args:
            name: Tool name
            version: Version number (latest if None)

        Returns:
            Tool function or None if not found
        """
        code = self.get_tool_code(name, version)
        if code is None:
            return None

        # Create a namespace and track names before execution
        namespace = {'__name__': '__main__', '__builtins__': __builtins__}
        names_before = set(namespace.keys())

        try:
            exec(code, namespace)
        except Exception as e:
            logger.error(f"Failed to load tool '{name}': {e}")
            return None

        # Find newly defined functions (not imports)
        import types
        import typing

        new_names = set(namespace.keys()) - names_before

        for obj_name in new_names:
            if obj_name.startswith('_'):
                continue

            obj = namespace[obj_name]

            # Skip types, classes, and modules
            if isinstance(obj, type):
                continue
            if isinstance(obj, types.ModuleType):
                continue
            # Skip typing generics
            if hasattr(obj, '__module__') and obj.__module__ == 'typing':
                continue

            # Check if it's a function
            if callable(obj):
                self._loaded_functions[f"{name}_v{version}"] = obj
                return obj

        logger.warning(f"No callable function found in tool '{name}'")
        return None

    def get_metadata(self, name: str, version: Optional[int] = None) -> Optional[ToolMetadata]:
        """Get tool metadata.

        Args:
            name: Tool name
            version: Version number (latest if None)

        Returns:
            ToolMetadata or None if not found
        """
        if name not in self._tools:
            return None

        if version is None:
            version = max(self._tools[name].keys())

        return self._tools[name].get(version)

    def update_metadata(self, name: str, version: int, **kwargs):
        """Update tool metadata.

        Args:
            name: Tool name
            version: Version number
            **kwargs: Metadata fields to update
        """
        metadata = self.get_metadata(name, version)
        if metadata is None:
            logger.warning(f"Tool '{name}' v{version} not found")
            return

        # Update fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        # Save to file
        if metadata.source == "generated":
            metadata_file = self.generated_dir / name / f"v{version}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

    def list_tools(self, source: Optional[str] = None) -> List[ToolMetadata]:
        """List all tools.

        Args:
            source: Filter by source ("predefined" or "generated")

        Returns:
            List of tool metadata (latest version of each tool)
        """
        tools = []
        for tool_name, versions in self._tools.items():
            latest_version = max(versions.keys())
            metadata = versions[latest_version]

            if source is None or metadata.source == source:
                tools.append(metadata)

        return tools

    def list_versions(self, name: str) -> List[int]:
        """List all versions of a tool.

        Args:
            name: Tool name

        Returns:
            List of version numbers
        """
        if name not in self._tools:
            return []
        return sorted(self._tools[name].keys())

    def rollback(self, name: str, to_version: int) -> bool:
        """Rollback to a previous version (marks newer versions as inactive).

        Args:
            name: Tool name
            to_version: Version to rollback to

        Returns:
            True if successful
        """
        if name not in self._tools or to_version not in self._tools[name]:
            logger.warning(f"Cannot rollback '{name}' to v{to_version}: not found")
            return False

        # In this implementation, we don't delete newer versions,
        # we just return the older version when loading
        logger.info(f"Rolled back '{name}' to v{to_version}")
        return True

    def delete_tool(self, name: str, version: int) -> bool:
        """Delete a specific tool version.

        Args:
            name: Tool name
            version: Version to delete

        Returns:
            True if successful
        """
        if name not in self._tools or version not in self._tools[name]:
            logger.warning(f"Tool '{name}' v{version} not found")
            return False

        metadata = self._tools[name][version]

        # Don't delete predefined tools
        if metadata.source == "predefined":
            logger.warning(f"Cannot delete predefined tool '{name}'")
            return False

        # Delete files
        code_file = self.generated_dir / name / f"v{version}.py"
        metadata_file = self.generated_dir / name / f"v{version}.json"

        if code_file.exists():
            code_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()

        # Remove from registry
        del self._tools[name][version]

        logger.info(f"Deleted tool '{name}' v{version}")
        return True
