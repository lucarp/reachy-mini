"""Session management endpoints."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from ...api.main import app_state

logger = logging.getLogger(__name__)

router = APIRouter()


class ClearSessionRequest(BaseModel):
    """Clear session request."""
    session_id: str


@router.get("/list")
async def list_sessions():
    """List all sessions."""
    session_manager = app_state.get("session_manager")

    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    try:
        sessions = session_manager.get_all_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    agent_runner = app_state.get("agent_runner")

    if not agent_runner:
        raise HTTPException(status_code=503, detail="Agent runner not initialized")

    try:
        history = agent_runner.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_session(request: ClearSessionRequest):
    """Clear a session's history."""
    agent_runner = app_state.get("agent_runner")

    if not agent_runner:
        raise HTTPException(status_code=503, detail="Agent runner not initialized")

    try:
        agent_runner.clear_session(request.session_id)
        return {"status": "success", "session_id": request.session_id}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session_manager = app_state.get("session_manager")

    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")

    try:
        session_manager.delete_session(session_id)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
