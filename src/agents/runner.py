"""Agent runner for executing conversations with session management."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from agents import Agent, Runner
from ..utils.config import Config
from ..utils.session import SessionManager
from .coordinator import create_coordinator_agent
from .robot_agent import create_robot_agent
from .vision_agent import create_vision_agent
from .code_agent import create_code_agent, CodeGenerationPipeline
from ..validation.tool_validator import ToolValidator
from ..validation.tool_tester import ToolTester
from ..validation.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ReachyAgentRunner:
    """Runner for managing agent conversations with session persistence."""

    def __init__(self, config: Config, session_manager: SessionManager):
        """Initialize the agent runner.

        Args:
            config: Configuration object
            session_manager: Session manager for conversation history
        """
        self.config = config
        self.session_manager = session_manager

        # Create agents (including Phase 2 code agent)
        self.coordinator, self.robot_agent, self.vision_agent, self.code_agent = self._create_agents()

        logger.info("ReachyAgentRunner initialized with Phase 2 code generation")

    def _create_agents(self) -> tuple[Agent, Agent, Agent, Agent]:
        """Create all agents with proper handoffs.

        Returns:
            Tuple of (coordinator, robot_agent, vision_agent, code_agent)
        """
        # Initialize Phase 2 components
        logger.info("Initializing Phase 2 validation pipeline...")

        # Tool storage (use project root / src / tools)
        tools_storage = Path(__file__).parent.parent / "tools"

        validator = ToolValidator(strict_mode=True)
        tester = ToolTester(default_timeout=10.0)
        registry = ToolRegistry(tools_storage)

        pipeline = CodeGenerationPipeline(validator, tester, registry)

        logger.info(f"Phase 2 pipeline initialized (storage: {tools_storage})")

        # Create specialist agents first (without handoff back yet)
        robot_agent = create_robot_agent(
            config=self.config,
            handoff_back=None,  # Will be set after coordinator is created
        )

        vision_agent = create_vision_agent(
            config=self.config,
            handoff_back=None,  # Will be set after coordinator is created
        )

        code_agent = create_code_agent(
            config=self.config,
            pipeline=pipeline,
            handoff_back=None,  # Will be set after coordinator is created
        )

        # Create coordinator with handoffs to all specialists
        coordinator = create_coordinator_agent(
            config=self.config,
            handoffs=[robot_agent, vision_agent, code_agent],
        )

        # Note: In OpenAI Agents SDK, handoffs work automatically
        # The specialist agents can naturally complete and return control
        # No explicit "handoff back" is needed

        return coordinator, robot_agent, vision_agent, code_agent

    async def process_message(
        self,
        session_id: str,
        message: str,
    ) -> Dict[str, Any]:
        """Process a user message and return the response.

        Args:
            session_id: Session identifier
            message: User message

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Ensure session exists
            self.session_manager.create_session(session_id)

            # Add user message to history
            self.session_manager.add_message(
                session_id=session_id,
                role="user",
                content=message,
            )

            # Run agent using Runner.run() static method
            result = await Runner.run(
                starting_agent=self.coordinator,
                input=message,
                max_turns=10,
            )

            # Extract final response
            final_message = result.final_output if hasattr(result, 'final_output') else str(result)

            # Add assistant response to history
            self.session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=final_message,
                metadata={
                    "agent": self.coordinator.name,
                },
            )

            return {
                "status": "success",
                "response": final_message,
                "agent": self.coordinator.name,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
            }

    async def process_stream(
        self,
        session_id: str,
        message: str,
    ):
        """Process a user message and stream the response.

        Args:
            session_id: Session identifier
            message: User message

        Yields:
            Response chunks
        """
        try:
            # Ensure session exists
            self.session_manager.create_session(session_id)

            # Add user message
            self.session_manager.add_message(
                session_id=session_id,
                role="user",
                content=message,
            )

            # Stream response using Runner.run_streamed()
            full_response = ""

            async for event in Runner.run_streamed(
                starting_agent=self.coordinator,
                input=message,
                max_turns=10,
            ):
                # Handle different event types
                if hasattr(event, 'content') and event.content:
                    full_response += event.content
                    yield {
                        "type": "chunk",
                        "content": event.content,
                        "agent": self.coordinator.name,
                    }

            # Add full response to history
            if full_response:
                self.session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata={"agent": self.coordinator.name},
                )

            yield {
                "type": "done",
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
            }

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages
        """
        return self.session_manager.get_history(session_id)

    def clear_session(self, session_id: str):
        """Clear a session's history.

        Args:
            session_id: Session identifier
        """
        self.session_manager.clear_session(session_id)
        logger.info(f"Cleared session: {session_id}")
