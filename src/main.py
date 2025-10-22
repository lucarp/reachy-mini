"""Main entry point for Reachy Mini Agentic AI."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config


def setup_logging(config):
    """Set up logging configuration."""
    # Create logs directory
    log_file = Path(config.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def main():
    """Main entry point."""
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("Starting Reachy Mini Agentic AI")

    # Start API server
    import uvicorn
    from src.api.main import create_app

    app = create_app()

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level=config.logging.level.lower(),
    )


if __name__ == "__main__":
    main()
