"""Tools for robot control and vision analysis."""

from .robot_tools import (
    move_head,
    set_antennas,
    get_current_pose,
    look_at_object,
    express_emotion,
)
from .vision_tools import (
    take_photo,
    analyze_scene,
    detect_objects,
    describe_view,
)

__all__ = [
    # Robot tools
    "move_head",
    "set_antennas",
    "get_current_pose",
    "look_at_object",
    "express_emotion",
    # Vision tools
    "take_photo",
    "analyze_scene",
    "detect_objects",
    "describe_view",
]
