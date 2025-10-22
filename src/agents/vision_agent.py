"""Vision analysis specialist agent."""

import logging
from typing import List
from agents import Agent, Handoff
from agents.extensions.models.litellm_model import LitellmModel
from ..utils.config import Config
from ..tools.vision_tools import (
    take_photo,
    analyze_scene,
    detect_objects,
    describe_view,
)

logger = logging.getLogger(__name__)


def create_vision_agent(
    config: Config,
    handoff_back: Handoff,
) -> Agent:
    """Create the vision analysis specialist agent.

    This agent is responsible for:
    - Analyzing camera feed
    - Describing what the robot sees
    - Detecting objects
    - Answering questions about the visual scene

    Args:
        config: Configuration object
        handoff_back: Handoff to return to coordinator

    Returns:
        Configured Agent instance
    """
    model = LitellmModel(
        model=f"ollama/{config.llm.model}",
        api_base=config.llm.base_url,
        temperature=0.7,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
    )

    # Vision tools
    tools = [
        take_photo,
        analyze_scene,
        detect_objects,
        describe_view,
    ]

    agent = Agent(
        name=config.agents.vision.name,
        model=model,
        instructions=config.agents.vision.instructions,
        tools=tools,
        handoffs=[handoff_back],
    )

    logger.info(f"Created vision agent: {agent.name} with {len(tools)} tools")
    return agent
