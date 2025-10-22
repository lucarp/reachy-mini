"""Robot control specialist agent."""

import logging
from typing import List, Optional
from agents import Agent
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
    handoff_back: Optional[Agent] = None,
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
        base_url=config.llm.base_url,
    )

    # Robot control tools
    tools = [
        move_head,
        set_antennas,
        get_current_pose,
        look_at_object,
        express_emotion,
    ]

    # Build handoffs list (if handoff_back is provided)
    handoffs = [handoff_back] if handoff_back else []

    agent = Agent(
        name=config.agents.robot_control.name,
        model=model,
        instructions=config.agents.robot_control.instructions,
        tools=tools,
        handoffs=handoffs,
    )

    logger.info(f"Created robot agent: {agent.name} with {len(tools)} tools")
    return agent
