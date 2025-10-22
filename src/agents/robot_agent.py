"""Robot control specialist agent."""

import logging
from typing import List
from agents import Agent, Handoff
from agents.extensions.models.litellm_model import LitellmModel
from ..utils.config import Config
from ..tools.robot_tools import (
    move_head,
    set_antennas,
    get_current_pose,
    look_at_object,
    express_emotion,
)

logger = logging.getLogger(__name__)


def create_robot_agent(
    config: Config,
    handoff_back: Handoff,
) -> Agent:
    """Create the robot control specialist agent.

    This agent is responsible for:
    - Executing physical movements safely
    - Managing head and antenna positions
    - Expressing emotions through movement
    - Reporting movement results

    Args:
        config: Configuration object
        handoff_back: Handoff to return to coordinator

    Returns:
        Configured Agent instance
    """
    model = LitellmModel(
        model=f"ollama/{config.llm.model}",
        api_base=config.llm.base_url,
        temperature=0.5,  # Lower temperature for precise control
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
    )

    # Robot control tools
    tools = [
        move_head,
        set_antennas,
        get_current_pose,
        look_at_object,
        express_emotion,
    ]

    agent = Agent(
        name=config.agents.robot_control.name,
        model=model,
        instructions=config.agents.robot_control.instructions,
        tools=tools,
        handoffs=[handoff_back],
    )

    logger.info(f"Created robot agent: {agent.name} with {len(tools)} tools")
    return agent
