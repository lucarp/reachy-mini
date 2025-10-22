"""Robot control endpoints."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from ...tools.robot_tools import (
    move_head,
    set_antennas,
    get_current_pose,
    look_at_object,
    express_emotion,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class MoveHeadRequest(BaseModel):
    """Move head request."""
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    duration: float = 2.0


class SetAntennasRequest(BaseModel):
    """Set antennas request."""
    left: Literal["up", "down", "middle"] = "middle"
    right: Literal["up", "down", "middle"] = "middle"
    duration: float = 1.0


class LookAtObjectRequest(BaseModel):
    """Look at object request."""
    direction: Literal["left", "right", "up", "down", "center"]
    intensity: Literal["small", "medium", "large"] = "medium"


class ExpressEmotionRequest(BaseModel):
    """Express emotion request."""
    emotion: Literal["happy", "sad", "curious", "surprised", "neutral"]


@router.post("/move_head")
async def api_move_head(request: MoveHeadRequest):
    """Move robot's head."""
    try:
        result = await move_head(
            pitch=request.pitch,
            yaw=request.yaw,
            roll=request.roll,
            duration=request.duration,
        )
        return result
    except Exception as e:
        logger.error(f"Error moving head: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set_antennas")
async def api_set_antennas(request: SetAntennasRequest):
    """Set antenna positions."""
    try:
        result = await set_antennas(
            left=request.left,
            right=request.right,
            duration=request.duration,
        )
        return result
    except Exception as e:
        logger.error(f"Error setting antennas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current_pose")
async def api_get_current_pose():
    """Get current robot pose."""
    try:
        result = await get_current_pose()
        return result
    except Exception as e:
        logger.error(f"Error getting pose: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/look_at")
async def api_look_at_object(request: LookAtObjectRequest):
    """Look in a specific direction."""
    try:
        result = await look_at_object(
            direction=request.direction,
            intensity=request.intensity,
        )
        return result
    except Exception as e:
        logger.error(f"Error looking at object: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/express_emotion")
async def api_express_emotion(request: ExpressEmotionRequest):
    """Express an emotion."""
    try:
        result = await express_emotion(emotion=request.emotion)
        return result
    except Exception as e:
        logger.error(f"Error expressing emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
