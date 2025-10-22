"""Agent runner for executing conversations with session management."""

import logging
from typing import Dict, Any, Optional, List
from agents import Agent, Runner, Handoff
from ..utils.config import Config
from ..utils.session import SessionManager
from .coordinator import create_coordinator_agent
from .robot_agent import create_robot_agent
from .vision_agent import create_vision_agent

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

        # Create agents
        self.coordinator, self.robot_agent, self.vision_agent = self._create_agents()

        # Create runner
        self.runner = Runner(
            starting_agent=self.coordinator,
            agents=[self.coordinator, self.robot_agent, self.vision_agent],
        )

        logger.info("ReachyAgentRunner initialized")

    def _create_agents(self) -> tuple[Agent, Agent, Agent]:
        """Create all agents with proper handoffs.

        Returns:
            Tuple of (coordinator, robot_agent, vision_agent)
        """
        # Create handoff placeholders
        coordinator_handoff = Handoff(target_name="ReachyCoordinator")
        robot_handoff = Handoff(target_name="RobotControl")
        vision_handoff = Handoff(target_name="VisionAnalyst")

        # Create specialist agents
        robot_agent = create_robot_agent(
            config=self.config,
            handoff_back=coordinator_handoff,
        )

        vision_agent = create_vision_agent(
            config=self.config,
            handoff_back=coordinator_handoff,
        )

        # Create coordinator with handoffs to specialists
        coordinator = create_coordinator_agent(
            config=self.config,
            handoffs=[robot_handoff, vision_handoff],
        )

        return coordinator, robot_agent, vision_agent

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

            # Get conversation history
            history = self.session_manager.get_history(
                session_id=session_id,
                max_messages=self.config.session.max_history,
            )

            # Format history for agent
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]

            # Run agent
            response = await self.runner.run(messages=messages)

            # Extract final response
            final_message = response.messages[-1].content if response.messages else ""
            final_agent = response.messages[-1].agent if response.messages else self.coordinator.name

            # Add assistant response to history
            self.session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=final_message,
                metadata={
                    "agent": final_agent,
                    "message_count": len(response.messages),
                },
            )

            return {
                "status": "success",
                "response": final_message,
                "agent": final_agent,
                "session_id": session_id,
                "message_count": len(response.messages),
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

            # Get history
            history = self.session_manager.get_history(
                session_id=session_id,
                max_messages=self.config.session.max_history,
            )

            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in history
            ]

            # Stream response
            full_response = ""
            final_agent = self.coordinator.name

            async for chunk in self.runner.stream(messages=messages):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    final_agent = chunk.agent if hasattr(chunk, "agent") else final_agent
                    yield {
                        "type": "chunk",
                        "content": chunk.content,
                        "agent": final_agent,
                    }

            # Add full response to history
            if full_response:
                self.session_manager.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response,
                    metadata={"agent": final_agent},
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
