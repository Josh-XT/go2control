"""
Go2 Control — Configuration loader

Loads settings from config.yaml with environment variable overrides (GO2_* prefix).
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("go2.config")

CONFIG_FILE = Path(__file__).parent / "config.yaml"


@dataclass
class ConnectionConfig:
    """Connection settings for the Go2 robot."""

    method: str = "webrtc"  # webrtc or dds
    robot_ip: str = "192.168.12.1"  # Default AP mode IP
    connection_mode: str = "LocalAP"  # LocalAP, LocalSTA, Remote
    serial_number: str = ""  # Required for Remote mode
    dds_domain_id: int = 0
    dds_interface: str = "eth0"


@dataclass
class SafetyConfig:
    """Safety limits for velocity commands."""

    max_vx: float = 1.5  # m/s forward
    max_vy: float = 0.8  # m/s lateral
    max_vyaw: float = 2.0  # rad/s yaw
    api_command_timeout: float = 5.0  # seconds
    watchdog_timeout: float = 2.0  # seconds
    ramp_rate: float = 3.0  # m/s² acceleration limit


@dataclass
class AudioConfig:
    """Audio settings."""

    default_volume: int = 5  # 0-10
    enable_mic: bool = True
    sample_rate: int = 48000
    channels: int = 2


@dataclass
class VideoConfig:
    """Camera settings."""

    enable: bool = True
    jpeg_quality: int = 80
    max_fps: int = 10  # For snapshot endpoint, not stream


@dataclass
class AppConfig:
    """Top-level application config."""

    simulation: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


def load_config() -> AppConfig:
    """Load config from YAML file with GO2_* env var overrides."""
    raw = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            raw = yaml.safe_load(f) or {}

    config = AppConfig(
        simulation=_env_or(raw, "simulation", False, bool),
        host=_env_or(raw, "host", "0.0.0.0", str),
        port=_env_or(raw, "port", 8000, int),
        connection=ConnectionConfig(
            method=_env_or(raw.get("connection", {}), "method", "webrtc", str),
            robot_ip=_env_or(
                raw.get("connection", {}), "robot_ip", "192.168.12.1", str
            ),
            connection_mode=_env_or(
                raw.get("connection", {}), "connection_mode", "LocalAP", str
            ),
            serial_number=_env_or(
                raw.get("connection", {}), "serial_number", "", str
            ),
            dds_domain_id=_env_or(
                raw.get("connection", {}), "dds_domain_id", 0, int
            ),
            dds_interface=_env_or(
                raw.get("connection", {}), "dds_interface", "eth0", str
            ),
        ),
        safety=SafetyConfig(
            max_vx=_env_or(raw.get("safety", {}), "max_vx", 1.5, float),
            max_vy=_env_or(raw.get("safety", {}), "max_vy", 0.8, float),
            max_vyaw=_env_or(raw.get("safety", {}), "max_vyaw", 2.0, float),
            api_command_timeout=_env_or(
                raw.get("safety", {}), "api_command_timeout", 5.0, float
            ),
            watchdog_timeout=_env_or(
                raw.get("safety", {}), "watchdog_timeout", 2.0, float
            ),
            ramp_rate=_env_or(raw.get("safety", {}), "ramp_rate", 3.0, float),
        ),
        audio=AudioConfig(
            default_volume=_env_or(
                raw.get("audio", {}), "default_volume", 5, int
            ),
            enable_mic=_env_or(raw.get("audio", {}), "enable_mic", True, bool),
            sample_rate=_env_or(raw.get("audio", {}), "sample_rate", 48000, int),
            channels=_env_or(raw.get("audio", {}), "channels", 2, int),
        ),
        video=VideoConfig(
            enable=_env_or(raw.get("video", {}), "enable", True, bool),
            jpeg_quality=_env_or(raw.get("video", {}), "jpeg_quality", 80, int),
            max_fps=_env_or(raw.get("video", {}), "max_fps", 10, int),
        ),
    )

    logger.info("Config loaded (simulation=%s)", config.simulation)
    return config


def _env_or(section: dict, key: str, default, cast):
    """Check GO2_KEY env var, then YAML section, then default."""
    env_key = f"GO2_{key.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        if cast is bool:
            return env_val.lower() in ("1", "true", "yes")
        return cast(env_val)
    return cast(section.get(key, default))
