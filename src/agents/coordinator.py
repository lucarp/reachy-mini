"""Main coordinator agent that manages all interactions."""

import logging
from typing import List
from agents import Agent, Handoff
from agents.extensions.models.litellm_model import LitellmModel
from ..utils.config import Config

logger = logging.getLogger(__name__)


def create_coordinator_agent(
    config: Config,
    handoffs: List[Handoff],
    tools: List = None,
) -> Agent:
    """Create the main coordinator agent.

    The coordinator is responsible for:
    - Understanding user requests
    - Delegating to specialist agents (robot control, vision)
    - Maintaining conversation context
    - Providing natural, friendly responses

    Args:
        config: Configuration object
        handoffs: List of handoffs to specialist agents
        tools: Optional list of direct tools

    Returns:
        Configured Agent instance
    """
    # Create LiteLLM model for Ollama
    model = LitellmModel(
        model=f"ollama/{config.llm.model}",
        base_url=config.llm.base_url,
    )

    agent = Agent(
        name=config.agents.coordinator.name,
        model=model,
        instructions=config.agents.coordinator.instructions,
        handoffs=handoffs,
        tools=tools or [],
    )

    logger.info(f"Created coordinator agent: {agent.name}")
    return agent
