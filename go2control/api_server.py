#!/usr/bin/env python3
"""
Unitree Go2 Pro Robot Control API Server

REST + WebSocket API for controlling the Unitree Go2 Pro robot dog.

Features:
- Full SportClient movement control (walk, run, tricks, poses)
- Camera image capture and streaming
- Speaker control (volume, playback, megaphone)
- Microphone audio capture for transcription
- Velocity commands with safety ramping and watchdog
- Command sequencing / macros
- WebSocket real-time telemetry streaming
- Structured event logging

Communication:
- WebRTC (preferred): Works wirelessly via unitree_webrtc_connect
- DDS: Wired Ethernet via unitree_sdk2py (192.168.123.x)

Usage:
    python3 api_server.py
"""

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from config import AppConfig, load_config
from sequence_library import SequenceLibrary

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("go2api")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RobotMode(str, Enum):
    IDLE = "idle"
    DAMPING = "damping"
    STANDING = "standing"
    WALKING = "walking"
    RUNNING = "running"
    SITTING = "sitting"
    LYING_DOWN = "lying_down"


class MoveCommand(BaseModel):
    """Velocity command for robot movement."""

    vx: float = Field(0.0, description="Forward velocity m/s (neg = backward)")
    vy: float = Field(0.0, description="Lateral velocity m/s (pos = left)")
    vyaw: float = Field(0.0, description="Yaw rate rad/s (pos = CCW)")
    duration: Optional[float] = Field(
        None, ge=0.0, description="Auto-stop after N seconds"
    )


class EulerCommand(BaseModel):
    """Body orientation command."""

    roll: float = Field(0.0, ge=-0.75, le=0.75, description="Roll rad")
    pitch: float = Field(0.0, ge=-0.75, le=0.75, description="Pitch rad")
    yaw: float = Field(0.0, ge=-0.6, le=0.6, description="Yaw rad")


class VolumeCommand(BaseModel):
    """Speaker volume command."""

    level: int = Field(5, ge=0, le=10, description="Volume 0-10")


class SpeedLevelCommand(BaseModel):
    """Speed level command."""

    level: int = Field(0, ge=-1, le=1, description="-1=slow, 0=normal, 1=fast")


class SequenceStep(BaseModel):
    """One step in a command sequence."""

    action: str = Field(..., description="Step type: move, action, wait, euler")
    params: dict = Field(default_factory=dict)
    duration: float = Field(1.0, ge=0.0, description="Step duration (seconds)")


class SequenceCommand(BaseModel):
    """A sequence of steps to execute."""

    name: str = Field("unnamed", description="Sequence name")
    steps: list[SequenceStep] = Field(..., min_length=1)
    loop: bool = Field(False, description="Loop continuously")


class RobotStatus(BaseModel):
    """Full robot status."""

    mode: str = "idle"
    speed_level: int = 0
    battery_percent: int = 0
    connected: bool = False
    api_control_active: bool = False
    current_velocity: dict = Field(
        default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
    )
    target_velocity: dict = Field(
        default_factory=lambda: {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
    )
    uptime_seconds: float = 0.0
    active_sequence: Optional[str] = None
    volume: int = 5
    video_enabled: bool = True
    mic_enabled: bool = True


# ---------------------------------------------------------------------------
# Available actions — maps to Go2 SportClient methods
# These are all real SDK methods from unitree_sdk2py.go2.sport.sport_client
# ---------------------------------------------------------------------------

SPORT_ACTIONS: dict[str, dict] = {
    # State transitions
    "damp": {"id": 1001, "desc": "Emergency stop — all motors enter damping"},
    "balance_stand": {"id": 1002, "desc": "Stand with active balance"},
    "stop_move": {"id": 1003, "desc": "Stop current action, restore defaults"},
    "stand_up": {"id": 1004, "desc": "Stand at normal height (0.33m)"},
    "stand_down": {"id": 1005, "desc": "Lie down with joints locked"},
    "recovery_stand": {"id": 1006, "desc": "Recover from overturned to standing"},
    "sit": {"id": 1009, "desc": "Sit down animation"},
    "rise_sit": {"id": 1010, "desc": "Stand up from sitting"},
    # Tricks and animations
    "hello": {"id": 1016, "desc": "Wave hello animation"},
    "stretch": {"id": 1017, "desc": "Stretch animation"},
    "content": {"id": 1020, "desc": "Happy/content animation"},
    "dance1": {"id": 1022, "desc": "Dance routine 1"},
    "dance2": {"id": 1023, "desc": "Dance routine 2"},
    "pose": {"id": 1028, "desc": "Strike a pose"},
    "scrape": {"id": 1029, "desc": "New Year greeting animation"},
    "front_flip": {"id": 1030, "desc": "Front flip"},
    "front_jump": {"id": 1031, "desc": "Jump forward"},
    "front_pounce": {"id": 1032, "desc": "Pounce forward"},
    "heart": {"id": 1036, "desc": "Heart gesture"},
    # Gait modes
    "static_walk": {"id": 1061, "desc": "Static walking gait"},
    "trot_run": {"id": 1062, "desc": "Trotting/running gait"},
    "economic_gait": {"id": 1063, "desc": "Power-saving gait"},
    # Advanced
    "left_flip": {"id": 2041, "desc": "Left flip"},
    "back_flip": {"id": 2043, "desc": "Back flip"},
    "hand_stand": {"id": 2044, "desc": "Handstand mode"},
    "free_walk": {"id": 2045, "desc": "Free walk mode"},
    "cross_step": {"id": 2051, "desc": "Cross step mode"},
    "switch_joystick": {"id": 1027, "desc": "Toggle remote controller response"},
}


# ---------------------------------------------------------------------------
# Robot Connection Manager
# ---------------------------------------------------------------------------


class RobotConnection:
    """
    Manages the connection to the Go2 robot via WebRTC or DDS.

    In simulation mode, all calls succeed with dummy data.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._connected = False
        self._conn = None  # WebRTC connection object
        self._sport_client = None  # DDS sport client
        self._video_client = None
        self._vui_client = None
        self._volume = config.audio.default_volume
        self._last_frame: Optional[bytes] = None  # Last captured JPEG
        self._mic_buffer: list[bytes] = []
        self._video_task: Optional[asyncio.Task] = None
        self._audio_task: Optional[asyncio.Task] = None

    @property
    def connected(self) -> bool:
        return self._connected or self.config.simulation

    async def connect(self):
        """Establish connection to the robot."""
        if self.config.simulation:
            self._connected = True
            logger.info("Simulation mode — no real robot connection")
            return

        method = self.config.connection.method
        if method == "webrtc":
            await self._connect_webrtc()
        elif method == "dds":
            self._connect_dds()
        else:
            raise ValueError(f"Unknown connection method: {method}")

    async def _connect_webrtc(self):
        """Connect via unitree_webrtc_connect."""
        try:
            from unitree_webrtc_connect.webrtc_driver import (
                UnitreeWebRTCConnection,
                WebRTCConnectionMethod,
            )

            mode_map = {
                "LocalAP": WebRTCConnectionMethod.LocalAP,
                "LocalSTA": WebRTCConnectionMethod.LocalSTA,
                "Remote": WebRTCConnectionMethod.Remote,
            }
            mode = mode_map.get(
                self.config.connection.connection_mode,
                WebRTCConnectionMethod.LocalAP,
            )

            kwargs = {"ip": self.config.connection.robot_ip}
            if mode == WebRTCConnectionMethod.Remote:
                kwargs["serial_number"] = self.config.connection.serial_number

            self._conn = UnitreeWebRTCConnection(mode, **kwargs)
            await self._conn.connect()
            self._connected = True

            # Start video reception
            if self.config.video.enable:
                self._conn.video.switchVideoChannel(True)
                self._conn.video.add_track_callback(self._on_video_frame)

            # Start audio reception
            if self.config.audio.enable_mic:
                self._conn.audio.switchAudioChannel(True)
                self._conn.audio.add_track_callback(self._on_audio_frame)

            logger.info(
                "WebRTC connected to Go2 at %s (%s)",
                self.config.connection.robot_ip,
                self.config.connection.connection_mode,
            )

        except ImportError:
            raise RuntimeError(
                "unitree_webrtc_connect not installed. "
                "Install with: pip install unitree_webrtc_connect"
            )
        except Exception as e:
            logger.error("WebRTC connection failed: %s", e)
            raise

    def _connect_dds(self):
        """Connect via DDS (unitree_sdk2py)."""
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.go2.video.video_client import VideoClient
            from unitree_sdk2py.go2.vui.vui_client import VuiClient

            ChannelFactoryInitialize(
                self.config.connection.dds_domain_id,
                self.config.connection.dds_interface,
            )

            self._sport_client = SportClient()
            self._sport_client.SetTimeout(10.0)
            self._sport_client.Init()

            self._video_client = VideoClient()
            self._video_client.SetTimeout(3.0)
            self._video_client.Init()

            self._vui_client = VuiClient()
            self._vui_client.SetTimeout(3.0)
            self._vui_client.Init()

            self._connected = True
            logger.info("DDS connected via %s", self.config.connection.dds_interface)

        except ImportError:
            raise RuntimeError(
                "unitree_sdk2py not installed. "
                "Requires CycloneDDS and unitree_sdk2py."
            )

    async def disconnect(self):
        """Disconnect from the robot."""
        if self._video_task:
            self._video_task.cancel()
        if self._audio_task:
            self._audio_task.cancel()
        if self._conn:
            try:
                await self._conn.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Disconnected from Go2")

    # -- Video --

    async def _on_video_frame(self, track):
        """Callback for WebRTC video frames."""
        try:
            while True:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                try:
                    import cv2

                    _, buf = cv2.imencode(
                        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.config.video.jpeg_quality]
                    )
                    self._last_frame = buf.tobytes()
                except ImportError:
                    # Fallback without cv2
                    self._last_frame = img.tobytes()
        except Exception as e:
            logger.debug("Video frame loop ended: %s", e)

    def get_camera_image(self) -> Optional[bytes]:
        """Get the latest camera frame as JPEG bytes."""
        if self.config.simulation:
            # Generate a small placeholder image
            img = np.zeros((120, 160, 3), dtype=np.uint8)
            img[:, :] = (40, 40, 40)  # Dark gray
            try:
                import cv2

                cv2.putText(
                    img, "SIM", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2
                )
                _, buf = cv2.imencode(".jpg", img)
                return buf.tobytes()
            except ImportError:
                return None
        if self._video_client:
            # DDS path
            try:
                code, data = self._video_client.GetImageSample()
                if code == 0:
                    return bytes(data)
            except Exception as e:
                logger.error("DDS camera capture failed: %s", e)
            return None
        return self._last_frame

    # -- Audio --

    async def _on_audio_frame(self, frame):
        """Callback for WebRTC audio frames."""
        try:
            audio_data = frame.to_ndarray()
            pcm = audio_data.astype(np.int16).tobytes()
            self._mic_buffer.append(pcm)
            # Keep last ~10 seconds at 48kHz stereo 16-bit
            max_chunks = 500
            if len(self._mic_buffer) > max_chunks:
                self._mic_buffer = self._mic_buffer[-max_chunks:]
        except Exception as e:
            logger.debug("Audio frame error: %s", e)

    def get_mic_audio(self, seconds: float = 5.0) -> Optional[bytes]:
        """Get recent mic audio as raw PCM bytes."""
        if self.config.simulation:
            # Return silence
            sr = self.config.audio.sample_rate
            ch = self.config.audio.channels
            samples = int(sr * seconds * ch)
            return np.zeros(samples, dtype=np.int16).tobytes()
        if not self._mic_buffer:
            return None
        # Estimate how many chunks for requested duration
        chunk_duration = 0.02  # ~20ms per WebRTC frame
        n_chunks = min(int(seconds / chunk_duration), len(self._mic_buffer))
        return b"".join(self._mic_buffer[-n_chunks:])

    # -- Speaker --

    async def set_volume(self, level: int) -> bool:
        """Set speaker volume (0-10)."""
        level = max(0, min(10, level))
        self._volume = level
        if self.config.simulation:
            return True
        if self._vui_client:
            try:
                self._vui_client.SetVolume(level)
                return True
            except Exception as e:
                logger.error("DDS set volume failed: %s", e)
                return False
        if self._conn:
            try:
                from unitree_webrtc_connect.constants import RTC_TOPIC

                await self._conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["VUI_REQ"],
                    {"api_id": 1001, "parameter": {"volume": level}},
                )
                return True
            except Exception as e:
                logger.error("WebRTC set volume failed: %s", e)
                return False
        return False

    async def get_volume(self) -> int:
        """Get current speaker volume."""
        if self._vui_client and not self.config.simulation:
            try:
                code, level = self._vui_client.GetVolume()
                if code == 0:
                    self._volume = level
            except Exception:
                pass
        return self._volume

    # -- Sport commands --

    async def execute_sport_action(self, action_name: str) -> bool:
        """Execute a named sport action."""
        if action_name not in SPORT_ACTIONS:
            return False

        action = SPORT_ACTIONS[action_name]
        api_id = action["id"]

        if self.config.simulation:
            logger.info("SIM: Sport action %s (id=%d)", action_name, api_id)
            return True

        if self._sport_client:
            # DDS path — call the named method directly
            method = getattr(self._sport_client, _api_id_to_method(action_name), None)
            if method:
                try:
                    method()
                    return True
                except Exception as e:
                    logger.error("DDS sport action %s failed: %s", action_name, e)
                    return False

        if self._conn:
            # WebRTC path — publish sport command
            try:
                from unitree_webrtc_connect.constants import RTC_TOPIC

                await self._conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": api_id},
                )
                return True
            except Exception as e:
                logger.error("WebRTC sport action %s failed: %s", action_name, e)
                return False

        return False

    async def move(self, vx: float, vy: float, vyaw: float) -> bool:
        """Send velocity command."""
        if self.config.simulation:
            logger.info("SIM: Move vx=%.2f vy=%.2f vyaw=%.2f", vx, vy, vyaw)
            return True

        if self._sport_client:
            try:
                self._sport_client.Move(vx, vy, vyaw)
                return True
            except Exception as e:
                logger.error("DDS move failed: %s", e)
                return False

        if self._conn:
            try:
                from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

                await self._conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD.get("Move", 1008),
                        "parameter": {"x": vx, "y": vy, "z": vyaw},
                    },
                )
                return True
            except Exception as e:
                logger.error("WebRTC move failed: %s", e)
                return False

        return False

    async def set_euler(self, roll: float, pitch: float, yaw: float) -> bool:
        """Set body orientation."""
        if self.config.simulation:
            logger.info("SIM: Euler roll=%.2f pitch=%.2f yaw=%.2f", roll, pitch, yaw)
            return True

        if self._sport_client:
            try:
                self._sport_client.Euler(roll, pitch, yaw)
                return True
            except Exception as e:
                logger.error("DDS euler failed: %s", e)
                return False

        if self._conn:
            try:
                from unitree_webrtc_connect.constants import RTC_TOPIC

                await self._conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": 1007,
                        "parameter": {"x": roll, "y": pitch, "z": yaw},
                    },
                )
                return True
            except Exception as e:
                logger.error("WebRTC euler failed: %s", e)
                return False

        return False

    async def set_speed_level(self, level: int) -> bool:
        """Set speed level (-1=slow, 0=normal, 1=fast)."""
        if self.config.simulation:
            logger.info("SIM: Speed level %d", level)
            return True

        if self._sport_client:
            try:
                self._sport_client.SpeedLevel(level)
                return True
            except Exception as e:
                logger.error("DDS speed level failed: %s", e)
                return False

        if self._conn:
            try:
                from unitree_webrtc_connect.constants import RTC_TOPIC

                await self._conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": 1015, "parameter": level},
                )
                return True
            except Exception as e:
                logger.error("WebRTC speed level failed: %s", e)
                return False

        return False


def _api_id_to_method(action_name: str) -> str:
    """Convert snake_case action name to PascalCase method name."""
    return "".join(w.capitalize() for w in action_name.split("_"))


# ---------------------------------------------------------------------------
# Safety Manager
# ---------------------------------------------------------------------------


class SafetyManager:
    """Velocity clamping, ramping, and watchdog."""

    def __init__(self, config: AppConfig):
        self.config = config.safety
        self._current_vx = 0.0
        self._current_vy = 0.0
        self._current_vyaw = 0.0
        self._last_command_time = 0.0
        self.watchdog_tripped = False
        self._dt = 1.0 / 50.0  # 50 Hz assumed

    def feed_command(self):
        self._last_command_time = time.time()
        self.watchdog_tripped = False

    def check_watchdog(self) -> bool:
        elapsed = time.time() - self._last_command_time
        if elapsed > self.config.watchdog_timeout:
            self.watchdog_tripped = True
            return False
        return True

    def clamp(self, vx, vy, vyaw):
        vx = max(-self.config.max_vx, min(self.config.max_vx, vx))
        vy = max(-self.config.max_vy, min(self.config.max_vy, vy))
        vyaw = max(-self.config.max_vyaw, min(self.config.max_vyaw, vyaw))
        return vx, vy, vyaw

    def ramp(self, vx, vy, vyaw):
        max_delta = self.config.ramp_rate * self._dt
        self._current_vx += max(-max_delta, min(max_delta, vx - self._current_vx))
        self._current_vy += max(-max_delta, min(max_delta, vy - self._current_vy))
        self._current_vyaw += max(-max_delta, min(max_delta, vyaw - self._current_vyaw))
        return self._current_vx, self._current_vy, self._current_vyaw

    def reset(self):
        self._current_vx = 0.0
        self._current_vy = 0.0
        self._current_vyaw = 0.0

    @property
    def is_ramping(self) -> bool:
        return (
            abs(self._current_vx) > 0.01
            or abs(self._current_vy) > 0.01
            or abs(self._current_vyaw) > 0.01
        )


# ---------------------------------------------------------------------------
# Sequence Runner
# ---------------------------------------------------------------------------


class SequenceRunner:
    """Execute sequences of commands."""

    def __init__(self, controller: "RobotController"):
        self._ctrl = controller
        self._task: Optional[asyncio.Task] = None
        self._active_name: Optional[str] = None
        self._cancelled = False

    @property
    def active(self) -> Optional[str]:
        return self._active_name

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, seq: SequenceCommand) -> str:
        self.cancel()
        seq_id = str(uuid.uuid4())[:8]
        self._active_name = seq.name
        self._cancelled = False
        self._task = asyncio.create_task(self._run(seq, seq_id))
        return seq_id

    def cancel(self):
        if self._task and not self._task.done():
            self._cancelled = True
            self._task.cancel()
        self._active_name = None

    async def _run(self, seq: SequenceCommand, seq_id: str):
        try:
            while True:
                for step in seq.steps:
                    if self._cancelled:
                        return
                    await self._execute_step(step)
                    await asyncio.sleep(step.duration)
                if not seq.loop:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            await self._ctrl.stop_movement()
            self._active_name = None

    async def _execute_step(self, step: SequenceStep):
        if step.action == "move":
            vx = step.params.get("vx", 0.0)
            vy = step.params.get("vy", 0.0)
            vyaw = step.params.get("vyaw", 0.0)
            await self._ctrl.set_move_command(
                MoveCommand(vx=vx, vy=vy, vyaw=vyaw, duration=step.duration + 0.5)
            )
        elif step.action == "action":
            name = step.params.get("name", "")
            await self._ctrl.execute_action(name)
        elif step.action == "euler":
            roll = step.params.get("roll", 0.0)
            pitch = step.params.get("pitch", 0.0)
            yaw = step.params.get("yaw", 0.0)
            await self._ctrl.robot.set_euler(roll, pitch, yaw)
        elif step.action == "wait":
            await self._ctrl.stop_movement()
        else:
            logger.warning("Unknown sequence action: %s", step.action)


# ---------------------------------------------------------------------------
# RobotController — the core
# ---------------------------------------------------------------------------


class RobotController:
    """Central controller managing robot connection, safety, and commands."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.robot = RobotConnection(config)
        self.safety = SafetyManager(config)
        self.sequence_runner = SequenceRunner(self)
        self.sequence_library = SequenceLibrary()

        self.running = False
        self.start_time = time.time()

        # State
        self.current_mode = RobotMode.IDLE
        self.speed_level: int = 0
        self._api_active = False
        self._api_vx = 0.0
        self._api_vy = 0.0
        self._api_vyaw = 0.0
        self._api_cmd_expires = 0.0
        self._move_task: Optional[asyncio.Task] = None

        # WebSocket clients
        self._ws_clients: set[WebSocket] = set()

    async def start(self):
        self.running = True
        await self.robot.connect()
        logger.info("Robot controller started")

    async def stop(self):
        self.running = False
        await self.stop_movement()
        await self.robot.disconnect()
        logger.info("Robot controller stopped")

    async def set_move_command(self, cmd: MoveCommand):
        """Apply velocity command with safety limits."""
        vx, vy, vyaw = self.safety.clamp(cmd.vx, cmd.vy, cmd.vyaw)
        vx, vy, vyaw = self.safety.ramp(vx, vy, vyaw)
        self.safety.feed_command()
        self._api_active = True
        self._api_vx = vx
        self._api_vy = vy
        self._api_vyaw = vyaw

        await self.robot.move(vx, vy, vyaw)

        # Auto-stop after duration
        duration = cmd.duration or self.config.safety.api_command_timeout
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
        self._move_task = asyncio.create_task(self._auto_stop(duration))

        self.current_mode = RobotMode.WALKING
        logger.info("Move: vx=%.2f vy=%.2f vyaw=%.2f dur=%.1f", vx, vy, vyaw, duration)

    async def _auto_stop(self, duration: float):
        try:
            await asyncio.sleep(duration)
            await self.stop_movement()
        except asyncio.CancelledError:
            pass

    async def stop_movement(self):
        """Stop all movement."""
        self._api_active = False
        self._api_vx = 0.0
        self._api_vy = 0.0
        self._api_vyaw = 0.0
        self.safety.reset()
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
        await self.robot.execute_sport_action("stop_move")
        self.current_mode = RobotMode.STANDING

    async def execute_action(self, action_name: str) -> bool:
        """Execute a sport action by name."""
        ok = await self.robot.execute_sport_action(action_name)
        if ok:
            # Update mode based on action
            if action_name == "damp":
                self.current_mode = RobotMode.DAMPING
            elif action_name in ("stand_up", "balance_stand", "recovery_stand", "rise_sit"):
                self.current_mode = RobotMode.STANDING
            elif action_name in ("stand_down",):
                self.current_mode = RobotMode.LYING_DOWN
            elif action_name == "sit":
                self.current_mode = RobotMode.SITTING
            logger.info("Action: %s", action_name)
        return ok

    async def emergency_stop(self):
        """Emergency stop — damp all motors."""
        await self.stop_movement()
        self.sequence_runner.cancel()
        await self.robot.execute_sport_action("damp")
        self.current_mode = RobotMode.DAMPING
        logger.warning("EMERGENCY STOP triggered")

    def get_status(self) -> RobotStatus:
        return RobotStatus(
            mode=self.current_mode.value,
            speed_level=self.speed_level,
            battery_percent=100,  # TODO: read from robot
            connected=self.robot.connected,
            api_control_active=self._api_active,
            current_velocity={
                "vx": round(self.safety._current_vx, 3),
                "vy": round(self.safety._current_vy, 3),
                "vyaw": round(self.safety._current_vyaw, 3),
            },
            target_velocity={
                "vx": round(self._api_vx, 3),
                "vy": round(self._api_vy, 3),
                "vyaw": round(self._api_vyaw, 3),
            },
            uptime_seconds=round(time.time() - self.start_time, 1),
            active_sequence=self.sequence_runner.active,
            volume=self.robot._volume,
            video_enabled=self.config.video.enable,
            mic_enabled=self.config.audio.enable_mic,
        )

    # -- WebSocket broadcasting --

    def add_ws_client(self, ws: WebSocket):
        self._ws_clients.add(ws)

    def remove_ws_client(self, ws: WebSocket):
        self._ws_clients.discard(ws)

    async def broadcast_state(self):
        while self.running:
            if self._ws_clients:
                data = self.get_status().model_dump()
                dead = []
                for ws in self._ws_clients:
                    try:
                        await ws.send_json(data)
                    except Exception:
                        dead.append(ws)
                for ws in dead:
                    self._ws_clients.discard(ws)
            await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------

robot_controller: Optional[RobotController] = None
_ws_broadcast_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot_controller, _ws_broadcast_task

    config = load_config()
    robot_controller = RobotController(config)
    await robot_controller.start()
    _ws_broadcast_task = asyncio.create_task(robot_controller.broadcast_state())

    yield

    if _ws_broadcast_task:
        _ws_broadcast_task.cancel()
    await robot_controller.stop()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Unitree Go2 Pro Robot API",
    description=(
        "REST + WebSocket API for controlling the Unitree Go2 Pro robot dog.\n\n"
        "## Features\n"
        "- 30+ sport actions (walk, run, tricks, flips, dances)\n"
        "- Velocity movement with safety ramping\n"
        "- Camera image capture\n"
        "- Speaker volume control\n"
        "- Microphone audio capture\n"
        "- Body orientation (roll/pitch/yaw)\n"
        "- Command sequencing / macros\n"
        "- Real-time WebSocket telemetry\n"
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/status", tags=["Status"])
async def get_status():
    return robot_controller.get_status()


@app.post("/api/v1/move", tags=["Movement"])
async def move_robot(cmd: MoveCommand):
    await robot_controller.set_move_command(cmd)
    return {
        "status": "moving",
        "velocity": {"vx": cmd.vx, "vy": cmd.vy, "vyaw": cmd.vyaw},
        "duration": cmd.duration,
    }


@app.post("/api/v1/stop", tags=["Movement"])
async def stop_robot():
    await robot_controller.stop_movement()
    return {"status": "stopped"}


@app.post("/api/v1/euler", tags=["Movement"])
async def set_euler(cmd: EulerCommand):
    ok = await robot_controller.robot.set_euler(cmd.roll, cmd.pitch, cmd.yaw)
    if not ok:
        raise HTTPException(500, "Failed to set euler")
    return {
        "status": "ok",
        "euler": {"roll": cmd.roll, "pitch": cmd.pitch, "yaw": cmd.yaw},
    }


@app.post("/api/v1/speed_level", tags=["Movement"])
async def set_speed_level(cmd: SpeedLevelCommand):
    ok = await robot_controller.robot.set_speed_level(cmd.level)
    if not ok:
        raise HTTPException(500, "Failed to set speed level")
    robot_controller.speed_level = cmd.level
    return {"status": "ok", "level": cmd.level}


@app.post("/api/v1/emergency_stop", tags=["Safety"])
async def emergency_stop():
    await robot_controller.emergency_stop()
    return {"status": "emergency_stop", "mode": "damping"}


# -- Actions --


@app.get("/api/v1/actions", tags=["Actions"])
async def list_actions():
    return {
        "actions": {
            name: info["desc"] for name, info in SPORT_ACTIONS.items()
        },
        "total": len(SPORT_ACTIONS),
    }


@app.post("/api/v1/action/{action_name}", tags=["Actions"])
async def execute_action(action_name: str):
    if action_name not in SPORT_ACTIONS:
        available = list(SPORT_ACTIONS.keys())
        raise HTTPException(
            400,
            f"Unknown action '{action_name}'. Available: {available}",
        )
    ok = await robot_controller.execute_action(action_name)
    if not ok:
        raise HTTPException(500, f"Failed to execute action: {action_name}")
    return {"status": "executing", "action": action_name}


# -- Camera --


@app.get("/api/v1/camera/snapshot", tags=["Camera"])
async def camera_snapshot():
    """Get the latest camera frame as JPEG."""
    frame = robot_controller.robot.get_camera_image()
    if frame is None:
        raise HTTPException(503, "No camera frame available")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/api/v1/camera/snapshot/base64", tags=["Camera"])
async def camera_snapshot_base64():
    """Get the latest camera frame as base64 JPEG."""
    frame = robot_controller.robot.get_camera_image()
    if frame is None:
        raise HTTPException(503, "No camera frame available")
    b64 = base64.b64encode(frame).decode("utf-8")
    return {"image": f"data:image/jpeg;base64,{b64}"}


# -- Audio --


@app.get("/api/v1/audio/volume", tags=["Audio"])
async def get_volume():
    vol = await robot_controller.robot.get_volume()
    return {"volume": vol}


@app.post("/api/v1/audio/volume", tags=["Audio"])
async def set_volume(cmd: VolumeCommand):
    ok = await robot_controller.robot.set_volume(cmd.level)
    if not ok:
        raise HTTPException(500, "Failed to set volume")
    return {"volume": cmd.level}


@app.get("/api/v1/audio/mic", tags=["Audio"])
async def get_mic_audio(seconds: float = Query(5.0, ge=0.5, le=30.0)):
    """Get recent microphone audio as raw PCM (16-bit, 48kHz, stereo)."""
    audio = robot_controller.robot.get_mic_audio(seconds)
    if audio is None:
        raise HTTPException(503, "No microphone audio available")
    b64 = base64.b64encode(audio).decode("utf-8")
    return {
        "audio_base64": b64,
        "format": "pcm_s16le",
        "sample_rate": robot_controller.config.audio.sample_rate,
        "channels": robot_controller.config.audio.channels,
        "duration_seconds": seconds,
    }


# -- Sequences --


@app.get("/api/v1/sequences", tags=["Sequences"])
async def list_sequences():
    all_seqs = robot_controller.sequence_library.list()
    return {
        "sequences": {
            name: {"steps": len(data.get("steps", [])), "loop": data.get("loop", False)}
            for name, data in all_seqs.items()
        }
    }


@app.get("/api/v1/sequences/{name}", tags=["Sequences"])
async def get_sequence(name: str):
    seq = robot_controller.sequence_library.get(name)
    if not seq:
        raise HTTPException(404, f"Sequence '{name}' not found")
    return seq


@app.post("/api/v1/sequences/{name}", tags=["Sequences"])
async def save_sequence(name: str, seq: SequenceCommand):
    data = seq.model_dump()
    data["name"] = name
    ok = robot_controller.sequence_library.save(name, data)
    if not ok:
        raise HTTPException(500, "Failed to save sequence")
    return {"status": "saved", "name": name}


@app.delete("/api/v1/sequences/{name}", tags=["Sequences"])
async def delete_sequence(name: str):
    ok = robot_controller.sequence_library.delete(name)
    if not ok:
        raise HTTPException(400, f"Cannot delete '{name}' (built-in or not found)")
    return {"status": "deleted", "name": name}


@app.post("/api/v1/sequences/{name}/run", tags=["Sequences"])
async def run_sequence(name: str):
    data = robot_controller.sequence_library.get(name)
    if not data:
        raise HTTPException(404, f"Sequence '{name}' not found")
    seq = SequenceCommand(**data)
    seq_id = await robot_controller.sequence_runner.start(seq)
    return {"status": "started", "sequence": name, "id": seq_id}


@app.post("/api/v1/sequences/stop", tags=["Sequences"])
async def stop_sequence():
    robot_controller.sequence_runner.cancel()
    return {"status": "stopped"}


# -- Agent context --


@app.get("/api/v1/agent/context", tags=["Agent"])
async def get_agent_context():
    """Full context for an AI agent to understand robot capabilities."""
    status = robot_controller.get_status()
    return {
        "robot": "Unitree Go2 Pro",
        "description": (
            "Quadruped robot dog with HD camera, speaker, microphone, "
            "30+ sport actions including tricks/flips/dances, and full "
            "velocity control."
        ),
        "state": {
            "mode": status.mode,
            "speed_level": status.speed_level,
            "battery_percent": status.battery_percent,
            "connected": status.connected,
            "is_moving": status.api_control_active,
            "volume": status.volume,
        },
        "actions": {
            name: info["desc"] for name, info in SPORT_ACTIONS.items()
        },
        "action_categories": {
            "state_transitions": [
                "damp", "balance_stand", "stop_move", "stand_up",
                "stand_down", "recovery_stand", "sit", "rise_sit",
            ],
            "tricks": [
                "hello", "stretch", "content", "dance1", "dance2",
                "pose", "scrape", "front_flip", "front_jump",
                "front_pounce", "heart",
            ],
            "flips": ["front_flip", "left_flip", "back_flip"],
            "gaits": ["static_walk", "trot_run", "economic_gait"],
            "modes": [
                "free_walk", "cross_step", "hand_stand", "switch_joystick",
            ],
        },
        "movement": {
            "velocity": {
                "vx_range": [-1.5, 1.5],
                "vy_range": [-0.8, 0.8],
                "vyaw_range": [-2.0, 2.0],
            },
            "euler": {
                "roll_range": [-0.75, 0.75],
                "pitch_range": [-0.75, 0.75],
                "yaw_range": [-0.6, 0.6],
            },
            "speed_levels": {-1: "slow", 0: "normal", 1: "fast"},
        },
        "peripherals": {
            "camera": "HD wide-angle front camera (JPEG snapshot via API)",
            "speaker": "Built-in speaker, volume 0-10",
            "microphone": "Built-in mic, 48kHz stereo PCM",
        },
        "sequences": list(robot_controller.sequence_library.list().keys()),
        "tips": [
            "Use 'recovery_stand' if the robot falls over",
            "Call 'balance_stand' or 'stand_up' before sending move commands",
            "Use duration on move commands to auto-stop (safer)",
            "Camera snapshots return the robot's current view",
            "Mic audio can be sent to a transcription service",
            "The robot can do flips, dances, and tricks — use them!",
            "Set speed_level to 1 for fast mode, -1 for careful/slow mode",
        ],
    }


@app.post("/api/v1/agent/command", tags=["Agent"])
async def agent_command(body: dict):
    """
    Unified command endpoint for AI agents.

    Commands:
    - move: {"forward_speed": float, "lateral_speed": float, "turn_rate": float, "duration": float}
    - stop: {}
    - action: {"name": str}
    - sequence: {"name": str}
    - euler: {"roll": float, "pitch": float, "yaw": float}
    - speed_level: {"level": int}
    - volume: {"level": int}
    - camera: {}  — returns base64 image
    - mic: {"seconds": float}  — returns base64 audio
    - status: {}
    """
    command = body.get("command", "")
    params = body.get("parameters", {})

    try:
        if command == "move":
            cmd = MoveCommand(
                vx=float(params.get("forward_speed", 0)),
                vy=float(params.get("lateral_speed", 0)),
                vyaw=float(params.get("turn_rate", 0)),
                duration=float(params.get("duration", 2.0)),
            )
            await robot_controller.set_move_command(cmd)
            return {"success": True, "message": f"Moving: vx={cmd.vx} vy={cmd.vy} vyaw={cmd.vyaw}", "action": "move"}

        elif command == "stop":
            await robot_controller.stop_movement()
            return {"success": True, "message": "Stopped", "action": "stop"}

        elif command == "action":
            name = params.get("name", "")
            ok = await robot_controller.execute_action(name)
            if ok:
                return {"success": True, "message": f"Executing: {name}", "action": "action"}
            available = list(SPORT_ACTIONS.keys())
            return {"success": False, "message": f"Unknown action '{name}'. Available: {available}", "action": "action"}

        elif command == "sequence":
            name = params.get("name", "")
            data = robot_controller.sequence_library.get(name)
            if not data:
                available = list(robot_controller.sequence_library.list().keys())
                return {"success": False, "message": f"Sequence '{name}' not found. Available: {available}", "action": "sequence"}
            seq = SequenceCommand(**data)
            seq_id = await robot_controller.sequence_runner.start(seq)
            return {"success": True, "message": f"Sequence '{name}' started (id={seq_id})", "action": "sequence"}

        elif command == "euler":
            await robot_controller.robot.set_euler(
                float(params.get("roll", 0)),
                float(params.get("pitch", 0)),
                float(params.get("yaw", 0)),
            )
            return {"success": True, "message": "Euler set", "action": "euler"}

        elif command == "speed_level":
            level = int(params.get("level", 0))
            await robot_controller.robot.set_speed_level(level)
            robot_controller.speed_level = level
            return {"success": True, "message": f"Speed level: {level}", "action": "speed_level"}

        elif command == "volume":
            level = int(params.get("level", 5))
            await robot_controller.robot.set_volume(level)
            return {"success": True, "message": f"Volume: {level}", "action": "volume"}

        elif command == "camera":
            frame = robot_controller.robot.get_camera_image()
            if frame is None:
                return {"success": False, "message": "No camera frame available", "action": "camera"}
            b64 = base64.b64encode(frame).decode("utf-8")
            return {"success": True, "image": f"data:image/jpeg;base64,{b64}", "action": "camera"}

        elif command == "mic":
            seconds = float(params.get("seconds", 5.0))
            audio = robot_controller.robot.get_mic_audio(seconds)
            if audio is None:
                return {"success": False, "message": "No mic audio available", "action": "mic"}
            b64 = base64.b64encode(audio).decode("utf-8")
            return {
                "success": True,
                "audio_base64": b64,
                "format": "pcm_s16le",
                "sample_rate": 48000,
                "channels": 2,
                "action": "mic",
            }

        elif command == "status":
            status = robot_controller.get_status()
            return {"success": True, "status": status.model_dump(), "action": "status"}

        else:
            return {
                "success": False,
                "message": f"Unknown command '{command}'. Available: move, stop, action, sequence, euler, speed_level, volume, camera, mic, status",
                "action": command,
            }

    except Exception as e:
        return {"success": False, "message": str(e), "action": command}


# -- WebSocket --


@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    robot_controller.add_ws_client(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                cmd_type = msg.get("type")
                if cmd_type == "move":
                    cmd = MoveCommand(**msg.get("data", {}))
                    await robot_controller.set_move_command(cmd)
                elif cmd_type == "stop":
                    await robot_controller.stop_movement()
                elif cmd_type == "action":
                    await robot_controller.execute_action(msg.get("data", {}).get("name", ""))
                elif cmd_type == "emergency_stop":
                    await robot_controller.emergency_stop()
            except Exception as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        robot_controller.remove_ws_client(ws)


# -- Health & Dashboard --


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "uptime": round(time.time() - robot_controller.start_time, 1),
        "connected": robot_controller.robot.connected,
        "simulation": robot_controller.config.simulation,
    }


@app.get("/dashboard", response_class=HTMLResponse, tags=["System"])
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Dashboard not found</h1>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = load_config()
    print(f"\n  Unitree Go2 Pro Robot API Server v0.1.0")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Connection: {config.connection.method} ({config.connection.connection_mode})")
    print(f"  Robot IP: {config.connection.robot_ip}")
    print(f"  Simulation: {config.simulation}")
    print(f"  Actions: {len(SPORT_ACTIONS)} sport actions available")
    print()

    uvicorn.run(
        "api_server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )
