"""Chat endpoints for text and voice interaction."""

import logging
import json
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ...api.main import app_state

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response model."""
    status: str
    response: str
    agent: str
    session_id: str
    message_count: Optional[int] = None


@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """Send a text message to the agent.

    Args:
        request: Chat request with session_id and message

    Returns:
        Agent response
    """
    agent_runner = app_state.get("agent_runner")

    if not agent_runner:
        raise HTTPException(status_code=503, detail="Agent runner not initialized")

    try:
        result = await agent_runner.process_message(
            session_id=request.session_id,
            message=request.message,
        )

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_message(request: ChatRequest):
    """Stream agent response.

    Args:
        request: Chat request

    Returns:
        Streaming response
    """
    agent_runner = app_state.get("agent_runner")

    if not agent_runner:
        raise HTTPException(status_code=503, detail="Agent runner not initialized")

    async def generate():
        try:
            async for chunk in agent_runner.process_stream(
                session_id=request.session_id,
                message=request.message,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time bidirectional communication.

    Protocol:
    - Client sends: {"type": "message", "session_id": str, "message": str}
    - Server sends: {"type": "chunk"|"done"|"error", ...}
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    agent_runner = app_state.get("agent_runner")

    if not agent_runner:
        await websocket.send_json({
            "type": "error",
            "error": "Agent runner not initialized"
        })
        await websocket.close()
        return

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data.get("type") == "message":
                session_id = data.get("session_id")
                message = data.get("message")

                if not session_id or not message:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Missing session_id or message"
                    })
                    continue

                # Process and stream response
                async for chunk in agent_runner.process_stream(
                    session_id=session_id,
                    message=message,
                ):
                    await websocket.send_json(chunk)

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
