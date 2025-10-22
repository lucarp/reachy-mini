"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..utils.config import Config, load_config
from ..utils.session import SessionManager
from ..agents.runner import ReachyAgentRunner
from ..tools.robot_tools import set_robot_instance
from ..tools.vision_tools import set_vision_config
from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)


# Global instances
app_state = {
    "config": None,
    "robot": None,
    "session_manager": None,
    "agent_runner": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Reachy Mini Agentic AI API")

    # Load configuration
    config = load_config()
    app_state["config"] = config

    # Initialize robot
    robot = ReachyMini()
    app_state["robot"] = robot
    logger.info("Robot initialized")

    # Set robot instance for tools
    set_robot_instance(robot)
    set_vision_config(
        robot=robot,
        llm_config={
            "base_url": config.llm.base_url,
            "model": config.llm.model,
        }
    )

    # Initialize session manager
    session_manager = SessionManager(config.session.db_path)
    app_state["session_manager"] = session_manager
    logger.info("Session manager initialized")

    # Initialize agent runner
    agent_runner = ReachyAgentRunner(config, session_manager)
    app_state["agent_runner"] = agent_runner
    logger.info("Agent runner initialized")

    yield

    # Shutdown
    logger.info("Shutting down Reachy Mini Agentic AI API")

    # Clean up robot
    if app_state["robot"]:
        app_state["robot"].__exit__(None, None, None)

    # Clean up sessions
    if app_state["session_manager"]:
        app_state["session_manager"].cleanup_old_sessions(
            days=config.session.cleanup_after_days
        )


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Reachy Mini Agentic AI",
        description="Multimodal agentic AI system for Reachy Mini robot",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    config = load_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and include routers
    from .routes import chat, robot, vision, sessions

    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    app.include_router(robot.router, prefix="/api/robot", tags=["robot"])
    app.include_router(vision.router, prefix="/api/vision", tags=["vision"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Reachy Mini Agentic AI",
            "version": "1.0.0",
            "status": "online",
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "robot": app_state["robot"] is not None,
            "agent_runner": app_state["agent_runner"] is not None,
        }

    return app
