"""Vision analysis tools using @function_tool decorator."""

import base64
import logging
from typing import Dict, Any, Optional, List
from io import BytesIO
from PIL import Image
import numpy as np
import requests
from agents import function_tool
from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

# Global instances
_robot: Optional[ReachyMini] = None
_llm_config: Optional[Dict[str, str]] = None


def set_vision_config(robot: ReachyMini, llm_config: Dict[str, str]):
    """Set global configuration for vision tools.

    Args:
        robot: ReachyMini instance
        llm_config: Dictionary with base_url and model
    """
    global _robot, _llm_config
    _robot = robot
    _llm_config = llm_config
    logger.info("Vision tools configured")


def get_robot() -> ReachyMini:
    """Get the global robot instance."""
    if _robot is None:
        raise RuntimeError("Robot instance not initialized for vision tools")
    return _robot


def get_llm_config() -> Dict[str, str]:
    """Get LLM configuration."""
    if _llm_config is None:
        raise RuntimeError("LLM config not initialized for vision tools")
    return _llm_config


def _capture_image() -> Image.Image:
    """Capture image from robot's camera.

    Returns:
        PIL Image
    """
    robot = get_robot()
    frame = robot.media.camera.read()

    if frame is None:
        raise ValueError("Failed to capture image from camera")

    # Convert BGR to RGB (OpenCV uses BGR)
    rgb_frame = frame[:, :, ::-1]
    return Image.fromarray(rgb_frame)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string.

    Args:
        image: PIL Image

    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _query_vision_llm(image_b64: str, prompt: str) -> str:
    """Query vision LLM with image and prompt.

    Args:
        image_b64: Base64 encoded image
        prompt: Text prompt

    Returns:
        LLM response
    """
    config = get_llm_config()

    payload = {
        "model": config["model"],
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }

    try:
        response = requests.post(
            f"{config['base_url']}/api/generate",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except Exception as e:
        logger.error(f"Error querying vision LLM: {e}")
        raise


@function_tool
async def take_photo(save_path: Optional[str] = None) -> Dict[str, Any]:
    """Take a photo from Reachy's camera.

    Args:
        save_path: Optional path to save the image

    Returns:
        Dictionary with status and image info
    """
    try:
        image = _capture_image()

        result = {
            "status": "success",
            "action": "take_photo",
            "width": image.width,
            "height": image.height,
        }

        if save_path:
            image.save(save_path)
            result["saved_to"] = save_path
            logger.info(f"Photo saved to {save_path}")

        return result

    except Exception as e:
        logger.error(f"Error taking photo: {e}")
        return {
            "status": "error",
            "action": "take_photo",
            "error": str(e),
        }


@function_tool
async def analyze_scene(
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze the current scene from Reachy's camera using vision LLM.

    Args:
        question: Optional specific question about the scene.
                 If not provided, gives a general description.

    Returns:
        Dictionary with analysis results
    """
    try:
        # Capture image
        image = _capture_image()
        image_b64 = _image_to_base64(image)

        # Create prompt
        if question:
            prompt = question
        else:
            prompt = "Describe what you see in this image in detail. Include objects, people, colors, and spatial relationships."

        # Query vision LLM
        analysis = _query_vision_llm(image_b64, prompt)

        return {
            "status": "success",
            "action": "analyze_scene",
            "question": question or "general description",
            "analysis": analysis,
        }

    except Exception as e:
        logger.error(f"Error analyzing scene: {e}")
        return {
            "status": "error",
            "action": "analyze_scene",
            "error": str(e),
        }


@function_tool
async def detect_objects() -> Dict[str, Any]:
    """Detect and list objects visible in the current camera view.

    Returns:
        Dictionary with detected objects
    """
    try:
        image = _capture_image()
        image_b64 = _image_to_base64(image)

        prompt = (
            "List all objects you can see in this image. "
            "Format your response as a simple comma-separated list of object names."
        )

        response = _query_vision_llm(image_b64, prompt)

        # Parse comma-separated list
        objects = [obj.strip() for obj in response.split(",")]

        return {
            "status": "success",
            "action": "detect_objects",
            "objects": objects,
            "count": len(objects),
        }

    except Exception as e:
        logger.error(f"Error detecting objects: {e}")
        return {
            "status": "error",
            "action": "detect_objects",
            "error": str(e),
        }


@function_tool
async def describe_view(
    focus: Optional[str] = None,
) -> Dict[str, Any]:
    """Get a natural language description of what Reachy sees.

    Args:
        focus: Optional aspect to focus on (e.g., "colors", "people", "text")

    Returns:
        Dictionary with description
    """
    try:
        image = _capture_image()
        image_b64 = _image_to_base64(image)

        if focus:
            prompt = f"Describe what you see in this image, focusing particularly on: {focus}"
        else:
            prompt = "Describe what you see in this image as if you were explaining it to someone who can't see it."

        description = _query_vision_llm(image_b64, prompt)

        return {
            "status": "success",
            "action": "describe_view",
            "focus": focus,
            "description": description,
        }

    except Exception as e:
        logger.error(f"Error describing view: {e}")
        return {
            "status": "error",
            "action": "describe_view",
            "error": str(e),
        }
