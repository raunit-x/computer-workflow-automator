"""Configuration and settings for workflow automation."""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""
    
    # API Keys (loaded from .env file)
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    
    # Recording settings
    screenshot_interval_ms: int = 500  # Screenshot every 500ms
    audio_sample_rate: int = 44100
    audio_channels: int = 1
    audio_chunk_size: int = 1024
    
    # Storage paths
    recordings_dir: Path = Path("./recordings")
    workflows_dir: Path = Path("./workflows")
    
    # Claude settings
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384
    
    # Execution settings
    action_delay_ms: int = 100  # Delay between actions for stability
    screenshot_delay_ms: int = 500  # Wait after action before screenshot
    
    def __post_init__(self):
        """Load API keys from environment after initialization."""
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", self.anthropic_api_key)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", self.openai_api_key)
        
        # Ensure directories exist
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> Config:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

