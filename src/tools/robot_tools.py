"""Robot control tools using @function_tool decorator."""

import logging
from typing import Dict, Any, Optional, Literal
from agents import function_tool
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

logger = logging.getLogger(__name__)

# Global robot instance (will be set during initialization)
_robot: Optional[ReachyMini] = None


def set_robot_instance(robot: ReachyMini):
    """Set the global robot instance for tools to use.

    Args:
        robot: ReachyMini instance
    """
    global _robot
    _robot = robot
    logger.info("Robot instance set for tools")


def get_robot() -> ReachyMini:
    """Get the global robot instance.

    Returns:
        ReachyMini instance

    Raises:
        RuntimeError: If robot not initialized
    """
    if _robot is None:
        raise RuntimeError("Robot instance not initialized. Call set_robot_instance() first.")
    return _robot


@function_tool
async def move_head(
    pitch: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    duration: float = 2.0,
) -> Dict[str, Any]:
    """Move Reachy's head to a specified orientation.

    Args:
        pitch: Pitch angle in degrees (up/down), range: -45 to 45
        yaw: Yaw angle in degrees (left/right), range: -90 to 90
        roll: Roll angle in degrees (tilt), range: -30 to 30
        duration: Movement duration in seconds

    Returns:
        Dictionary with status and final position
    """
    try:
        robot = get_robot()

        # Create target pose
        target_pose = create_head_pose(
            z=pitch,
            y=yaw,
            roll=roll,
            degrees=True,
            mm=False
        )

        # Execute movement
        robot.goto_target(head=target_pose, duration=duration)

        # Get final position
        final_pose = robot.get_current_head_pose()

        return {
            "status": "success",
            "action": "move_head",
            "target": {"pitch": pitch, "yaw": yaw, "roll": roll},
            "final_position": {
                "pitch": final_pose.z,
                "yaw": final_pose.y,
                "roll": final_pose.roll,
            },
            "duration": duration,
        }

    except Exception as e:
        logger.error(f"Error moving head: {e}")
        return {
            "status": "error",
            "action": "move_head",
            "error": str(e),
        }


@function_tool
async def set_antennas(
    left: Literal["up", "down", "middle"] = "middle",
    right: Literal["up", "down", "middle"] = "middle",
    duration: float = 1.0,
) -> Dict[str, Any]:
    """Set the position of Reachy's antenna ears.

    Args:
        left: Left antenna position (up/down/middle)
        right: Right antenna position (up/down/middle)
        duration: Movement duration in seconds

    Returns:
        Dictionary with status and final antenna positions
    """
    try:
        robot = get_robot()

        # Map positions to angles
        position_map = {
            "up": 90,
            "middle": 0,
            "down": -90,
        }

        left_angle = position_map[left]
        right_angle = position_map[right]

        # Set antennas
        robot.goto_target(
            antennas={"left": left_angle, "right": right_angle},
            duration=duration
        )

        return {
            "status": "success",
            "action": "set_antennas",
            "left": left,
            "right": right,
            "angles": {"left": left_angle, "right": right_angle},
            "duration": duration,
        }

    except Exception as e:
        logger.error(f"Error setting antennas: {e}")
        return {
            "status": "error",
            "action": "set_antennas",
            "error": str(e),
        }


@function_tool
async def get_current_pose() -> Dict[str, Any]:
    """Get the current position of Reachy's head and antennas.

    Returns:
        Dictionary with current head pose and antenna positions
    """
    try:
        robot = get_robot()

        head_pose = robot.get_current_head_pose()
        joint_positions = robot.get_current_joint_positions()

        return {
            "status": "success",
            "head": {
                "pitch": head_pose.z,
                "yaw": head_pose.y,
                "roll": head_pose.roll,
            },
            "antennas": {
                "left": joint_positions.get("l_antenna", 0),
                "right": joint_positions.get("r_antenna", 0),
            },
        }

    except Exception as e:
        logger.error(f"Error getting current pose: {e}")
        return {
            "status": "error",
            "action": "get_current_pose",
            "error": str(e),
        }


@function_tool
async def look_at_object(
    direction: Literal["left", "right", "up", "down", "center"],
    intensity: Literal["small", "medium", "large"] = "medium",
) -> Dict[str, Any]:
    """Make Reachy look in a specific direction to focus on an object.

    Args:
        direction: Direction to look (left/right/up/down/center)
        intensity: How far to look (small/medium/large)

    Returns:
        Dictionary with status and head movement
    """
    # Define movement angles for each direction and intensity
    movements = {
        "left": {"small": (0, 20, 0), "medium": (0, 45, 0), "large": (0, 70, 0)},
        "right": {"small": (0, -20, 0), "medium": (0, -45, 0), "large": (0, -70, 0)},
        "up": {"small": (15, 0, 0), "medium": (30, 0, 0), "large": (40, 0, 0)},
        "down": {"small": (-15, 0, 0), "medium": (-30, 0, 0), "large": (-40, 0, 0)},
        "center": {"small": (0, 0, 0), "medium": (0, 0, 0), "large": (0, 0, 0)},
    }

    pitch, yaw, roll = movements[direction][intensity]

    result = await move_head(pitch=pitch, yaw=yaw, roll=roll, duration=1.5)

    result["action"] = "look_at_object"
    result["direction"] = direction
    result["intensity"] = intensity

    return result


@function_tool
async def express_emotion(
    emotion: Literal["happy", "sad", "curious", "surprised", "neutral"],
) -> Dict[str, Any]:
    """Express an emotion using head and antenna movements.

    Args:
        emotion: Emotion to express

    Returns:
        Dictionary with status and movements performed
    """
    try:
        robot = get_robot()

        # Define emotion expressions
        expressions = {
            "happy": {
                "antennas": {"left": "up", "right": "up"},
                "head": {"pitch": 10, "yaw": 0, "roll": 0},
            },
            "sad": {
                "antennas": {"left": "down", "right": "down"},
                "head": {"pitch": -20, "yaw": 0, "roll": 0},
            },
            "curious": {
                "antennas": {"left": "up", "right": "middle"},
                "head": {"pitch": 5, "yaw": 20, "roll": 10},
            },
            "surprised": {
                "antennas": {"left": "up", "right": "up"},
                "head": {"pitch": 15, "yaw": 0, "roll": 0},
            },
            "neutral": {
                "antennas": {"left": "middle", "right": "middle"},
                "head": {"pitch": 0, "yaw": 0, "roll": 0},
            },
        }

        expression = expressions[emotion]

        # Set antennas
        await set_antennas(
            left=expression["antennas"]["left"],
            right=expression["antennas"]["right"],
            duration=1.0,
        )

        # Move head
        await move_head(
            pitch=expression["head"]["pitch"],
            yaw=expression["head"]["yaw"],
            roll=expression["head"]["roll"],
            duration=1.5,
        )

        return {
            "status": "success",
            "action": "express_emotion",
            "emotion": emotion,
            "expression": expression,
        }

    except Exception as e:
        logger.error(f"Error expressing emotion: {e}")
        return {
            "status": "error",
            "action": "express_emotion",
            "error": str(e),
        }
