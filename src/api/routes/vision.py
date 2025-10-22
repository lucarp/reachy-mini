"""Vision analysis endpoints."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ...tools.vision_tools import (
    take_photo,
    analyze_scene,
    detect_objects,
    describe_view,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class TakePhotoRequest(BaseModel):
    """Take photo request."""
    save_path: Optional[str] = None


class AnalyzeSceneRequest(BaseModel):
    """Analyze scene request."""
    question: Optional[str] = None


class DescribeViewRequest(BaseModel):
    """Describe view request."""
    focus: Optional[str] = None


@router.post("/take_photo")
async def api_take_photo(request: TakePhotoRequest):
    """Take a photo from robot's camera."""
    try:
        result = await take_photo(save_path=request.save_path)
        return result
    except Exception as e:
        logger.error(f"Error taking photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze_scene")
async def api_analyze_scene(request: AnalyzeSceneRequest):
    """Analyze the current scene."""
    try:
        result = await analyze_scene(question=request.question)
        return result
    except Exception as e:
        logger.error(f"Error analyzing scene: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect_objects")
async def api_detect_objects():
    """Detect objects in the current view."""
    try:
        result = await detect_objects()
        return result
    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/describe_view")
async def api_describe_view(request: DescribeViewRequest):
    """Get a description of the current view."""
    try:
        result = await describe_view(focus=request.focus)
        return result
    except Exception as e:
        logger.error(f"Error describing view: {e}")
        raise HTTPException(status_code=500, detail=str(e))
