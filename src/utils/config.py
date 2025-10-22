"""Configuration management using Pydantic v2 and YAML."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "gemma3:27b"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30


class SpeechToTextConfig(BaseModel):
    """WhisperX configuration."""
    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 1
    language: str = "en"
    vad_filter: bool = True


class SynthesisConfig(BaseModel):
    """Piper synthesis parameters."""
    volume: float = 1.0
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8


class TextToSpeechConfig(BaseModel):
    """Piper TTS configuration."""
    voice: str = "en_US-lessac-medium"
    voice_path: Optional[str] = None
    sample_rate: int = 22050
    streaming: bool = True
    synthesis_config: SynthesisConfig = Field(default_factory=SynthesisConfig)


class CameraConfig(BaseModel):
    """Camera configuration."""
    width: int = 640
    height: int = 480
    fps: int = 30


class WorkspaceLimits(BaseModel):
    """Robot workspace limits in degrees."""
    pitch: List[float] = Field(default=[-45, 45])
    yaw: List[float] = Field(default=[-90, 90])
    roll: List[float] = Field(default=[-30, 30])

    @field_validator("pitch", "yaw", "roll")
    @classmethod
    def validate_limits(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("Limits must be [min, max]")
        if v[0] >= v[1]:
            raise ValueError("Min must be less than max")
        return v


class SafetyConfig(BaseModel):
    """Robot safety configuration."""
    max_head_speed: float = 50.0  # degrees/second
    collision_check: bool = True
    workspace_limits: WorkspaceLimits = Field(default_factory=WorkspaceLimits)


class RobotConfig(BaseModel):
    """Robot configuration."""
    simulation: bool = True
    scene: str = "minimal"
    camera: CameraConfig = Field(default_factory=CameraConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class AgentConfig(BaseModel):
    """Individual agent configuration."""
    name: str
    instructions: str


class AgentsConfig(BaseModel):
    """All agents configuration."""
    coordinator: AgentConfig
    robot_control: AgentConfig
    vision: AgentConfig
    code_generation: AgentConfig


class WebSocketConfig(BaseModel):
    """WebSocket configuration."""
    ping_interval: int = 30
    ping_timeout: int = 10


class APIConfig(BaseModel):
    """FastAPI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)


class SessionConfig(BaseModel):
    """Session management configuration."""
    db_path: str = "./data/sessions.db"
    max_history: int = 50
    cleanup_after_days: int = 7


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/reachy_agent.log"


class Config(BaseModel):
    """Main configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    speech_to_text: SpeechToTextConfig = Field(default_factory=SpeechToTextConfig)
    text_to_speech: TextToSpeechConfig = Field(default_factory=TextToSpeechConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    agents: AgentsConfig
    api: APIConfig = Field(default_factory=APIConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def expand_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in config dict."""
    if isinstance(config_dict, dict):
        return {k: expand_env_vars(v) for k, v in config_dict.items()}
    elif isinstance(config_dict, list):
        return [expand_env_vars(item) for item in config_dict]
    elif isinstance(config_dict, str):
        # Expand ${VAR:-default} pattern
        if config_dict.startswith("${") and config_dict.endswith("}"):
            content = config_dict[2:-1]
            if ":-" in content:
                var_name, default = content.split(":-", 1)
                return os.getenv(var_name, default)
            else:
                return os.getenv(content, config_dict)
        return config_dict
    else:
        return config_dict


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file with environment variable expansion.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        raw_config = yaml.safe_load(f)

    # Expand environment variables
    expanded_config = expand_env_vars(raw_config)

    # Validate and create Config object
    config = Config(**expanded_config)

    # Create necessary directories
    Path(config.session.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.logging.file).parent.mkdir(parents=True, exist_ok=True)

    return config
