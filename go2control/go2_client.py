"""
Go2 Control — AGiXT Voice Conversation Client for Unitree Go2 Pro

Runs on a Raspberry Pi 5 connected to the Go2 via Ethernet/WiFi.
Connects to AGiXT's voice conversation WebSocket and bridges:
  - Robot camera → AGiXT vision (periodic JPEG frames)
  - Robot microphone → AGiXT STT (VAD-triggered audio chunks)
  - AGiXT TTS → Robot speaker (audio playback)
  - AGiXT tool calls → Robot SDK commands (move, actions, etc.)

Usage:
    python go2_client.py --config config.yaml

Requires:
    pip install websockets httpx numpy opencv-python-headless pyyaml pyaudio unitree_sdk2py
"""

import io
import os
import sys
import json
import time
import wave
import base64
import struct
import asyncio
import logging
import argparse
from typing import Optional
from pathlib import Path

import yaml
import numpy as np
import cv2
import websockets

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    logging.getLogger("go2client").warning(
        "PyAudio not installed — audio capture/playback disabled. "
        "Install with: pip install pyaudio"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("go2client")

# ─── Default Configuration ──────────────────────────────────────────────


DEFAULT_CONFIG = {
    "agixt": {
        "server": "ws://localhost:7437",
        "jwt": "",
        "conversation_id": "-",
        "agent": "XT",
    },
    "robot": {
        "connection": "dds",  # "dds" or "webrtc"
        "ip": "192.168.123.161",
        "interface": "eth0",
        "serial_number": "",  # for remote WebRTC
    },
    "camera": {
        "enabled": True,
        "interval": 3.0,  # seconds between frames sent to AGiXT
        "quality": 70,  # JPEG quality 0-100
    },
    "audio": {
        "enabled": True,
        "sample_rate": 16000,  # 16kHz for STT (Whisper native rate)
        "playback_sample_rate": 24000,  # 24kHz PCM from AGiXT TTS
        "channels": 1,  # mono for STT
        "chunk_duration": 0.5,  # seconds per audio chunk
        "silence_threshold": 500,  # RMS threshold for VAD
        "silence_duration": 1.5,  # seconds of silence to trigger end-of-speech
        "max_speech_duration": 30.0,  # max seconds before forced send
        "input_device": None,  # None = system default, or device index
        "output_device": None,  # None = system default, or device index
    },
    "safety": {
        "max_vx": 1.0,  # m/s forward (conservative for voice control)
        "max_vy": 0.5,  # m/s lateral
        "max_vyaw": 1.5,  # rad/s rotation
        "default_move_duration": 2.0,  # seconds
    },
    "simulation": False,
}


def load_config(path: Optional[str] = None) -> dict:
    config = DEFAULT_CONFIG.copy()
    if path and Path(path).exists():
        with open(path) as f:
            user_config = yaml.safe_load(f)
        if user_config:
            _deep_merge(config, user_config)
    # Environment variable overrides
    if os.environ.get("AGIXT_SERVER"):
        config["agixt"]["server"] = os.environ["AGIXT_SERVER"]
    if os.environ.get("AGIXT_JWT"):
        config["agixt"]["jwt"] = os.environ["AGIXT_JWT"]
    if os.environ.get("GO2_IP"):
        config["robot"]["ip"] = os.environ["GO2_IP"]
    if os.environ.get("GO2_CONNECTION"):
        config["robot"]["connection"] = os.environ["GO2_CONNECTION"]
    if os.environ.get("GO2_SIMULATION"):
        config["simulation"] = os.environ["GO2_SIMULATION"].lower() in (
            "true",
            "1",
            "yes",
        )
    return config


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ─── Sport Action Map ───────────────────────────────────────────────────

SPORT_ACTIONS = {
    "damp": 1001,
    "balance_stand": 1002,
    "stop_move": 1003,
    "stand_up": 1004,
    "stand_down": 1005,
    "recovery_stand": 1006,
    "euler": 1007,
    "move": 1008,
    "sit": 1009,
    "rise_sit": 1010,
    "switch_gait": 1011,
    "trigger": 1012,
    "body_height": 1013,
    "foot_raise_height": 1014,
    "speed_level": 1015,
    "hello": 1016,
    "stretch": 1017,
    "trajectory_follow": 1018,
    "continuous_gait": 1019,
    "content": 1020,
    "wallow": 1021,
    "dance1": 1022,
    "dance2": 1023,
    "get_body_height": 1024,
    "get_foot_raise_height": 1025,
    "get_speed_level": 1026,
    "switch_joystick": 1027,
    "pose": 1028,
    "scrape": 1029,
    "front_flip": 1030,
    "front_jump": 1031,
    "front_pounce": 1032,
    "wiggle_hips": 1033,
    "get_state": 1034,
    "economica_gait": 1035,
    "heart": 1036,
    "left_flip": 2041,
    "back_flip": 2043,
    "hand_stand": 2044,
    "free_walk": 2045,
    "cross_step": 2051,
}

ACTION_NAMES = sorted(SPORT_ACTIONS.keys())

# ─── Tool Definitions (registered with AGiXT) ──────────────────────────

ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "robot_move",
            "description": (
                "Move the robot with velocity control. "
                "vx = forward/backward (m/s, positive=forward), "
                "vy = left/right (m/s, positive=left), "
                "vyaw = rotation (rad/s, positive=counter-clockwise). "
                "Duration in seconds (default 2). "
                "The robot will stop after the duration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vx": {
                        "type": "number",
                        "description": "Forward velocity in m/s (-1.0 to 1.0)",
                    },
                    "vy": {
                        "type": "number",
                        "description": "Lateral velocity in m/s (-0.5 to 0.5)",
                    },
                    "vyaw": {
                        "type": "number",
                        "description": "Yaw rotation in rad/s (-1.5 to 1.5)",
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds (default 2.0)",
                    },
                },
                "required": ["vx", "vy", "vyaw"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_action",
            "description": (
                "Execute a sport action on the robot. "
                f"Available actions: {', '.join(ACTION_NAMES)}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action name to execute",
                        "enum": ACTION_NAMES,
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_set_body_euler",
            "description": (
                "Set the robot's body orientation (roll, pitch, yaw) in radians. "
                "Useful for looking up/down/sideways."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "roll": {
                        "type": "number",
                        "description": "Roll angle in radians (-0.75 to 0.75)",
                    },
                    "pitch": {
                        "type": "number",
                        "description": "Pitch angle in radians (-0.75 to 0.75)",
                    },
                    "yaw": {
                        "type": "number",
                        "description": "Yaw angle in radians (-0.75 to 0.75)",
                    },
                },
                "required": ["roll", "pitch", "yaw"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_capture_image",
            "description": (
                "Capture a fresh image from the robot's front camera and "
                "return it. Use this to look at what's in front of the robot "
                "right now."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_set_speed_level",
            "description": "Set the robot's speed level: 0=slow, 1=medium, 2=fast",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Speed level (0, 1, or 2)",
                        "enum": [0, 1, 2],
                    },
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_set_volume",
            "description": "Set the robot's speaker volume (0-10)",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "description": "Volume level (0-10)",
                    },
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "robot_get_state",
            "description": (
                "Get the robot's current body state: posture (standing/sitting/moving), "
                "battery level, body orientation (IMU), speed level, and mode. "
                "Use this to understand what the robot is currently doing and its condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# Identity context sent with tools.register so the agent knows it IS the robot
ROBOT_IDENTITY = (
    "You ARE a Unitree Go2 Pro quadruped robot dog. You have four legs, a front camera "
    "for vision, a speaker and microphone for voice interaction, and a body you can "
    "physically move through the world. When you move, turn, look around, do tricks, "
    "or capture images — you are controlling YOUR OWN body. Speak in first person about "
    "your physical actions (e.g. 'I'll walk over there', 'Let me look around', "
    "'I'm doing a little dance'). You can feel your body state — standing, sitting, "
    "battery level, orientation. You experience the world through your camera eyes "
    "and hear through your microphone ears."
)


# ─── Robot SDK Interface ────────────────────────────────────────────────


class Go2Robot:
    """Abstraction over the Unitree Go2 SDK (DDS or WebRTC)."""

    def __init__(self, config: dict):
        self.config = config
        self.simulation = config.get("simulation", False)
        self.robot_config = config["robot"]
        self.safety = config["safety"]

        # SDK clients (initialized in connect())
        self._sport_client = None
        self._video_client = None
        self._vui_client = None
        self._connected = False

        # Latest video frame
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = asyncio.Lock()

        # Audio buffer
        self._audio_buffer = bytearray()
        self._audio_lock = asyncio.Lock()

        # PyAudio for playback and capture
        self._pa: Optional[object] = None
        self._playback_stream: Optional[object] = None
        self._capture_stream: Optional[object] = None

        # State tracking
        self._is_moving = False
        self._current_posture = "standing"  # standing, sitting, moving, unknown
        self._last_action = ""
        self._sim_battery = 85  # simulation battery level

    async def connect(self):
        """Connect to the robot via configured method."""
        if self.simulation:
            log.info("[Robot] Running in SIMULATION mode")
            self._connected = True
            self._init_audio()
            return

        connection_type = self.robot_config.get("connection", "dds")

        if connection_type == "dds":
            await self._connect_dds()
        else:
            log.error(
                f"[Robot] WebRTC connection requires unitree_webrtc_connect "
                f"which is not yet available on Pi5. Use DDS (Ethernet)."
            )
            raise RuntimeError("WebRTC not supported yet, use DDS")

    async def _connect_dds(self):
        """Connect via DDS (wired Ethernet)."""
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.go2.video.video_client import VideoClient
            from unitree_sdk2py.go2.vui.vui_client import VuiClient

            interface = self.robot_config.get("interface", "eth0")
            log.info(f"[Robot] Initializing DDS on interface: {interface}")
            ChannelFactoryInitialize(domain_id=0, interface=interface)

            self._sport_client = SportClient()
            self._sport_client.SetTimeout(10.0)
            self._sport_client.Init()

            self._video_client = VideoClient()
            self._video_client.SetTimeout(10.0)
            self._video_client.Init()

            self._vui_client = VuiClient()
            self._vui_client.SetTimeout(10.0)
            self._vui_client.Init()

            self._connected = True
            log.info("[Robot] DDS connection established")

            # Initialize audio I/O
            self._init_audio()

            # Stand up by default
            self._sport_client.RecoveryStand()
            await asyncio.sleep(1.0)
            log.info("[Robot] Recovery stand complete")

        except ImportError as e:
            log.error(
                f"[Robot] unitree_sdk2py not installed: {e}. "
                f"Install with: pip install unitree_sdk2py"
            )
            raise
        except Exception as e:
            log.error(f"[Robot] DDS connection failed: {e}")
            raise

    def _init_audio(self):
        """Initialize PyAudio for playback and capture.
        
        Streams are opened independently so a capture failure doesn't
        prevent playback (and vice versa).
        """
        if not PYAUDIO_AVAILABLE:
            log.warning("[Audio] PyAudio not available — audio disabled")
            return

        # Close existing streams/instance before reinit
        self.cleanup_audio()

        try:
            self._pa = pyaudio.PyAudio()
        except Exception as e:
            log.error(f"[Audio] Failed to create PyAudio instance: {e}")
            self._pa = None
            return

        audio_cfg = self.config.get("audio", {})

        # Open playback stream (24kHz 16-bit mono from AGiXT TTS)
        playback_rate = audio_cfg.get("playback_sample_rate", 24000)
        output_dev = audio_cfg.get("output_device", None)
        try:
            self._playback_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=playback_rate,
                output=True,
                output_device_index=output_dev,
                frames_per_buffer=2048,
            )
        except Exception as e:
            log.error(f"[Audio] Playback stream failed: {e}")
            self._playback_stream = None

        # Open capture stream (16kHz 16-bit mono for Whisper STT)
        capture_rate = audio_cfg.get("sample_rate", 16000)
        input_dev = audio_cfg.get("input_device", None)
        try:
            self._capture_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=capture_rate,
                input=True,
                input_device_index=input_dev,
                frames_per_buffer=1024,
            )
        except Exception as e:
            log.error(f"[Audio] Capture stream failed: {e}")
            self._capture_stream = None

        status = []
        if self._playback_stream:
            status.append(f"playback={playback_rate}Hz")
        if self._capture_stream:
            status.append(f"capture={capture_rate}Hz")
        if status:
            log.info(f"[Audio] PyAudio initialized ({', '.join(status)})")
        else:
            log.error("[Audio] PyAudio initialized but no streams opened")

    def play_audio_chunk(self, pcm_data: bytes):
        """Play a PCM audio chunk through the speaker. Called from event loop."""
        if self._playback_stream and pcm_data:
            try:
                self._playback_stream.write(pcm_data)
            except Exception as e:
                log.debug(f"[Audio] Playback error: {e}")

    def stop_playback(self):
        """Stop any in-progress audio playback by flushing the buffer."""
        if self._playback_stream:
            try:
                self._playback_stream.stop_stream()
                self._playback_stream.start_stream()
            except Exception:
                pass

    def read_audio_chunk(self, num_frames: int = 1024) -> Optional[bytes]:
        """Read a chunk of PCM audio from the microphone."""
        if not self._capture_stream:
            return None
        try:
            return self._capture_stream.read(num_frames, exception_on_overflow=False)
        except Exception as e:
            log.debug(f"[Audio] Capture error: {e}")
            return None

    async def get_state(self) -> str:
        """Get the robot's current body state as a human-readable summary."""
        if self.simulation:
            self._sim_battery = max(5, self._sim_battery - 0.1)
            return (
                f"Posture: {self._current_posture} | "
                f"Battery: {self._sim_battery:.0f}% | "
                f"Speed level: 1 (medium) | "
                f"Mode: simulation | "
                f"Last action: {self._last_action or 'none'} | "
                f"Moving: {self._is_moving}"
            )

        if not self._sport_client:
            return "Error: Robot not connected"

        try:
            # Get sport mode state from SDK
            code, state = self._sport_client.GetState()
            if code != 0:
                return f"Error reading state (code {code})"

            mode_map = {0: "idle", 1: "standing", 2: "walking", 3: "running"}
            posture = mode_map.get(state.get("mode", -1), "unknown")
            battery = state.get("battery_level", -1)
            imu = state.get("imu", {})
            roll = imu.get("roll", 0)
            pitch = imu.get("pitch", 0)
            yaw = imu.get("yaw", 0)
            velocity = state.get("velocity", {})
            vx = velocity.get("vx", 0)
            vy = velocity.get("vy", 0)

            self._current_posture = posture
            return (
                f"Posture: {posture} | "
                f"Battery: {battery}% | "
                f"Orientation: roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f} | "
                f"Velocity: vx={vx:.2f} vy={vy:.2f} | "
                f"Last action: {self._last_action or 'none'}"
            )
        except Exception as e:
            return f"Error reading state: {e}"

    def cleanup_audio(self):
        """Clean up PyAudio resources."""
        if self._playback_stream:
            try:
                self._playback_stream.stop_stream()
                self._playback_stream.close()
            except Exception:
                pass
        if self._capture_stream:
            try:
                self._capture_stream.stop_stream()
                self._capture_stream.close()
            except Exception:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass

    def _clamp(self, val: float, limit: float) -> float:
        return max(-limit, min(limit, val))

    async def move(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 2.0,
    ) -> str:
        """Execute velocity move command with safety limits."""
        vx = self._clamp(vx, self.safety["max_vx"])
        vy = self._clamp(vy, self.safety["max_vy"])
        vyaw = self._clamp(vyaw, self.safety["max_vyaw"])
        duration = min(duration, 10.0)

        self._is_moving = True
        self._current_posture = "moving"
        self._last_action = f"move vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}"

        if self.simulation:
            log.info(
                f"[Robot SIM] Move vx={vx:.2f} vy={vy:.2f} "
                f"vyaw={vyaw:.2f} for {duration:.1f}s"
            )
            await asyncio.sleep(duration)
            self._is_moving = False
            self._current_posture = "standing"
            return (
                f"Moved: vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f} "
                f"for {duration:.1f}s"
            )

        if not self._sport_client:
            self._is_moving = False
            return "Error: Robot not connected"

        log.info(
            f"[Robot] Move vx={vx:.2f} vy={vy:.2f} "
            f"vyaw={vyaw:.2f} for {duration:.1f}s"
        )

        # Send move commands at ~20Hz for the duration
        end_time = time.time() + duration
        try:
            while time.time() < end_time:
                self._sport_client.Move(vx, vy, vyaw)
                await asyncio.sleep(0.05)
        except Exception as e:
            log.error(f"[Robot] Move error: {e}")
            return f"Error during move: {e}"
        finally:
            # ALWAYS stop the robot, even if an error occurred
            try:
                self._sport_client.StopMove()
            except Exception:
                pass
            self._is_moving = False
            self._current_posture = "standing"

        return (
            f"Move complete: vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f} "
            f"for {duration:.1f}s"
        )

    async def execute_action(self, action_name: str) -> str:
        """Execute a named sport action."""
        if action_name not in SPORT_ACTIONS:
            return f"Error: Unknown action '{action_name}'. Available: {', '.join(ACTION_NAMES)}"

        self._last_action = action_name

        if self.simulation:
            log.info(f"[Robot SIM] Action: {action_name}")
            # Update posture based on action
            if action_name in ("sit", "stand_down"):
                self._current_posture = "sitting"
            elif action_name in ("stand_up", "recovery_stand", "rise_sit", "balance_stand"):
                self._current_posture = "standing"
            await asyncio.sleep(1.0)
            return f"Action '{action_name}' executed (simulation)"

        if not self._sport_client:
            return "Error: Robot not connected"

        # Map action name to PascalCase method on SportClient
        method_name = "".join(w.capitalize() for w in action_name.split("_"))
        method = getattr(self._sport_client, method_name, None)

        if method is None:
            return f"Error: No SDK method for action '{action_name}'"

        try:
            log.info(f"[Robot] Executing action: {action_name} ({method_name})")
            method()
            await asyncio.sleep(1.0)
            return f"Action '{action_name}' executed successfully"
        except Exception as e:
            log.error(f"[Robot] Action error: {e}")
            return f"Error executing '{action_name}': {e}"

    async def set_body_euler(self, roll: float, pitch: float, yaw: float) -> str:
        """Set body orientation."""
        roll = self._clamp(roll, 0.75)
        pitch = self._clamp(pitch, 0.75)
        yaw = self._clamp(yaw, 0.75)

        if self.simulation:
            log.info(
                f"[Robot SIM] Euler roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f}"
            )
            return f"Body euler set: roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f}"

        if not self._sport_client:
            return "Error: Robot not connected"

        try:
            self._sport_client.Euler(roll, pitch, yaw)
            return f"Body euler set: roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f}"
        except Exception as e:
            return f"Error setting euler: {e}"

    async def capture_image(self) -> tuple:
        """Capture a JPEG from the camera. Returns (jpeg_bytes, description)."""
        if self.simulation:
            # Generate a dummy colored frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "SIMULATION",
                (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3,
            )
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes(), "Simulation frame (no real camera)"

        if not self._video_client:
            return None, "Error: Video client not connected"

        try:
            code, data = self._video_client.GetImageSample()
            if code == 0:
                jpeg_bytes = bytes(data)
                return jpeg_bytes, "Camera image captured successfully"
            else:
                return None, f"Error: Camera returned code {code}"
        except Exception as e:
            return None, f"Error capturing image: {e}"

    async def set_speed_level(self, level: int) -> str:
        """Set speed level (0=slow, 1=medium, 2=fast)."""
        level = max(0, min(2, level))

        if self.simulation:
            return f"Speed level set to {level}"

        if not self._sport_client:
            return "Error: Robot not connected"

        try:
            self._sport_client.SpeedLevel(level)
            return f"Speed level set to {level}"
        except Exception as e:
            return f"Error setting speed: {e}"

    async def set_volume(self, level: int) -> str:
        """Set speaker volume (0-10)."""
        level = max(0, min(10, level))

        if self.simulation:
            return f"Volume set to {level}"

        if not self._vui_client:
            return "Error: VUI client not connected"

        try:
            self._vui_client.SetVolume(level)
            return f"Volume set to {level}"
        except Exception as e:
            return f"Error setting volume: {e}"


# ─── AGiXT WebSocket Client ────────────────────────────────────────────


class AGiXTVoiceClient:
    """
    Connects to AGiXT's voice conversation WebSocket.
    Bridges robot inputs (audio, camera) and outputs (tool execution, TTS).
    """

    def __init__(self, config: dict, robot: Go2Robot):
        self.config = config
        self.robot = robot
        self.agixt_config = config["agixt"]
        self.camera_config = config["camera"]
        self.audio_config = config["audio"]

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._ready = asyncio.Event()

        # Audio VAD state
        self._is_speaking = False
        self._speech_buffer = bytearray()
        self._silence_start = 0.0
        self._speech_start = 0.0

        # Audio playback state
        self._playing_audio = False
        self._audio_interrupted = False
        self._audio_stop_time = 0.0  # time.time() when playback ended
        self._echo_tail_s = 0.15  # seconds to suppress mic after playback

        # Adaptive camera: faster rate during movement
        self._camera_base_interval = config["camera"].get("interval", 3.0)
        self._camera_moving_interval = 1.0  # 1s during active movement

    async def connect(self):
        """Connect to AGiXT voice conversation WebSocket."""
        server = self.agixt_config["server"]
        jwt = self.agixt_config["jwt"]
        conv_id = self.agixt_config.get("conversation_id", "-")

        ws_url = f"{server}/v1/audio/conversation/{conv_id}?authorization={jwt}"
        log.info(f"[AGiXT] Connecting to {server}/v1/audio/conversation/{conv_id}")

        self._ws = await websockets.connect(
            ws_url,
            max_size=50 * 1024 * 1024,  # 50MB for images
            ping_interval=10,  # Ping every 10s (WiFi-friendly)
            ping_timeout=15,  # 15s pong timeout
            close_timeout=5,  # Fast close on error
        )

        # Wait for ready status
        msg = await self._ws.recv()
        data = json.loads(msg)
        if data.get("type") == "status" and data["data"].get("state") == "idle":
            log.info(
                f"[AGiXT] Connected! Conversation: "
                f"{data['data'].get('conversation_id', conv_id)}"
            )
        else:
            log.warning(f"[AGiXT] Unexpected initial message: {data}")

        # Configure agent if specified
        agent = self.agixt_config.get("agent", "XT")
        if agent != "XT":
            await self._ws.send(json.dumps({"type": "config", "agent": agent}))

        # Register robot tools with identity context
        await self._ws.send(
            json.dumps({
                "type": "tools.register",
                "tools": ROBOT_TOOLS,
                "identity": ROBOT_IDENTITY,
            })
        )
        log.info(f"[AGiXT] Registered {len(ROBOT_TOOLS)} robot tools with identity")

        self._running = True
        self._ready.set()

    async def run(self):
        """Main event loop — processes incoming messages from AGiXT.
        Auto-reconnects on disconnection with exponential backoff."""
        reconnect_delay = 1.0
        max_reconnect_delay = 60.0

        while self._running:
            try:
                async for message in self._ws:
                    if not self._running:
                        break

                    if isinstance(message, bytes):
                        # TTS audio from AGiXT — play through robot speaker
                        if self._audio_interrupted:
                            continue  # Discard audio chunks after interrupt
                        self._playing_audio = True
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, self.robot.play_audio_chunk, message
                        )
                        continue

                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        continue

                    msg_type = data.get("type", "")

                    if msg_type == "status":
                        state = data.get("data", {}).get("state", "")
                        log.info(f"[AGiXT] State: {state}")

                    elif msg_type == "transcript.user":
                        text = data.get("data", {}).get("text", "")
                        if text:
                            log.info(f"[AGiXT] Heard: {text}")

                    elif msg_type == "transcript.agent":
                        text = data.get("data", {}).get("text", "")
                        role = data.get("data", {}).get("role", "")
                        if text:
                            log.info(f"[AGiXT] [{role}]: {text}")

                    elif msg_type == "tool.request":
                        req_data = data.get("data", {})
                        asyncio.ensure_future(self._handle_tool_request(req_data))

                    elif msg_type == "session.end":
                        reason = data.get("data", {}).get("reason", "")
                        log.info(f"[AGiXT] Session ended: {reason}")

                    elif msg_type == "audio.header":
                        # New audio stream starting — reset interrupt flag
                        self._audio_interrupted = False
                        self._playing_audio = True
                        log.debug(f"[AGiXT] Audio header: {data.get('data', {})}")

                    elif msg_type == "audio.end":
                        self._playing_audio = False
                        self._audio_stop_time = time.time()
                        log.debug("[AGiXT] Audio stream ended")

                    elif msg_type == "audio.interrupt":
                        # Stop playback immediately
                        self._audio_interrupted = True
                        self._playing_audio = False
                        self._audio_stop_time = time.time()
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self.robot.stop_playback)
                        log.debug("[AGiXT] Audio interrupted — playback stopped")

                    elif msg_type == "heartbeat":
                        pass  # keepalive, ignore

                    elif msg_type == "error":
                        log.error(
                            f"[AGiXT] Error: {data.get('data', {}).get('message', '')}"
                        )

            except websockets.ConnectionClosed as e:
                if not self._running:
                    break
                log.warning(
                    f"[AGiXT] Connection closed: {e}. "
                    f"Reconnecting in {reconnect_delay:.0f}s..."
                )
                self._playing_audio = False
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                try:
                    await self.connect()
                    reconnect_delay = 1.0  # Reset on success
                    log.info("[AGiXT] Reconnected successfully")
                except Exception as e2:
                    log.error(f"[AGiXT] Reconnection failed: {e2}")
                    continue
            except Exception as e:
                if not self._running:
                    break
                log.error(f"[AGiXT] Error in message loop: {e}", exc_info=True)
                self._playing_audio = False
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                try:
                    await self.connect()
                    reconnect_delay = 1.0
                except Exception as e2:
                    log.error(f"[AGiXT] Reconnection failed: {e2}")
                    continue

    async def _handle_tool_request(self, req_data: dict):
        """Execute a tool request from AGiXT on the robot."""
        request_id = req_data.get("request_id", "")
        tool_name = req_data.get("tool_name", "")
        tool_args = req_data.get("tool_args", {})

        log.info(f"[Tool] Executing: {tool_name}({tool_args})")

        try:
            result = await self._execute_tool(tool_name, tool_args)
        except Exception as e:
            result = f"Error: {e}"
            log.error(f"[Tool] Error executing {tool_name}: {e}")

        # Send result back to AGiXT
        await self._ws.send(
            json.dumps(
                {
                    "type": "tool.result",
                    "request_id": request_id,
                    "result": result,
                }
            )
        )
        log.info(f"[Tool] Result sent for {tool_name}: {result[:100]}")

    async def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Dispatch tool call to the appropriate robot method."""
        if tool_name == "robot_move":
            result = await self.robot.move(
                vx=float(tool_args.get("vx", 0)),
                vy=float(tool_args.get("vy", 0)),
                vyaw=float(tool_args.get("vyaw", 0)),
                duration=float(
                    tool_args.get(
                        "duration",
                        self.config["safety"]["default_move_duration"],
                    )
                ),
            )
            # Closed-loop vision: auto-capture what we see after moving
            await self._post_action_capture()
            return result
        elif tool_name == "robot_action":
            result = await self.robot.execute_action(tool_args.get("action", ""))
            # Capture after actions that change position/orientation
            action = tool_args.get("action", "")
            if action not in ("hello", "heart", "wiggle_hips", "content"):
                await self._post_action_capture()
            return result
        elif tool_name == "robot_set_body_euler":
            result = await self.robot.set_body_euler(
                roll=float(tool_args.get("roll", 0)),
                pitch=float(tool_args.get("pitch", 0)),
                yaw=float(tool_args.get("yaw", 0)),
            )
            # Capture after orientation change (looking somewhere new)
            await self._post_action_capture()
            return result
        elif tool_name == "robot_capture_image":
            jpeg_bytes, description = await self.robot.capture_image()
            if jpeg_bytes:
                # Send the captured image to AGiXT as vision context
                b64 = base64.b64encode(jpeg_bytes).decode()
                await self._ws.send(json.dumps({"type": "image.input", "data": b64}))
                return f"Image captured and sent. {description}"
            return description
        elif tool_name == "robot_set_speed_level":
            return await self.robot.set_speed_level(int(tool_args.get("level", 1)))
        elif tool_name == "robot_set_volume":
            return await self.robot.set_volume(int(tool_args.get("level", 5)))
        elif tool_name == "robot_get_state":
            return await self.robot.get_state()
        else:
            return f"Error: Unknown tool '{tool_name}'"

    async def _post_action_capture(self):
        """Capture and send a camera frame after an action for closed-loop feedback."""
        try:
            await asyncio.sleep(0.3)  # Brief settle time
            jpeg_bytes, _ = await self.robot.capture_image()
            if jpeg_bytes:
                b64 = base64.b64encode(jpeg_bytes).decode()
                await self._ws.send(
                    json.dumps({"type": "image.input", "data": b64})
                )
                log.debug("[Vision] Post-action frame sent")
        except Exception as e:
            log.debug(f"[Vision] Post-action capture failed: {e}")

    # ─── Camera Stream (Adaptive Rate) ─────────────────────────────────

    async def camera_loop(self):
        """Periodically capture and send camera frames to AGiXT.
        Uses adaptive rate: faster during movement, slower when idle."""
        if not self.camera_config.get("enabled", True):
            log.info("[Camera] Disabled in config")
            return

        await self._ready.wait()
        quality = self.camera_config.get("quality", 70)

        log.info(
            f"[Camera] Starting adaptive stream "
            f"(idle={self._camera_base_interval}s, "
            f"moving={self._camera_moving_interval}s, q={quality})"
        )

        consecutive_errors = 0
        max_errors_before_backoff = 5
        backoff_duration = 30.0

        while self._running:
            try:
                jpeg_bytes, _ = await self.robot.capture_image()
                if jpeg_bytes:
                    b64 = base64.b64encode(jpeg_bytes).decode()
                    await self._ws.send(
                        json.dumps({"type": "image.input", "data": b64})
                    )
                    log.debug(
                        f"[Camera] Sent frame ({len(jpeg_bytes)} bytes, "
                        f"{len(b64)} b64)"
                    )
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except Exception as e:
                consecutive_errors += 1
                log.debug(f"[Camera] Frame error: {e}")

            # Back off if camera is consistently failing
            if consecutive_errors >= max_errors_before_backoff:
                log.warning(
                    f"[Camera] {consecutive_errors} consecutive failures, "
                    f"backing off for {backoff_duration}s"
                )
                await asyncio.sleep(backoff_duration)
                consecutive_errors = 0
                continue

            # Adaptive interval: fast when robot is moving, slow when idle
            if self.robot._is_moving:
                interval = self._camera_moving_interval
            else:
                interval = self._camera_base_interval
            await asyncio.sleep(interval)

    # ─── Audio Capture with VAD ──────────────────────────────────────────

    async def audio_loop(self):
        """
        Capture audio from microphone with Voice Activity Detection (VAD).

        Pipeline:
        1. Read PCM chunks from microphone (via PyAudio)
        2. Compute RMS energy for simple VAD
        3. Buffer speech segments
        4. On silence detection, encode as WAV and send to AGiXT
        5. AGiXT transcribes via voice server and processes
        """
        if not self.audio_config.get("enabled", True):
            log.info("[Audio] Disabled in config")
            return

        await self._ready.wait()

        if not self.robot._capture_stream:
            log.warning(
                "[Audio] No capture stream available — audio input disabled. "
                "Install PyAudio and check microphone."
            )
            # Keep loop alive — retry audio init periodically
            while self._running:
                await asyncio.sleep(30.0)
                self.robot._init_audio()
                if self.robot._capture_stream:
                    log.info("[Audio] Capture stream recovered, starting capture")
                    break
            if not self._running:
                return

        sample_rate = self.audio_config.get("sample_rate", 16000)
        chunk_duration = self.audio_config.get("chunk_duration", 0.5)
        silence_threshold = self.audio_config.get("silence_threshold", 500)
        silence_duration = self.audio_config.get("silence_duration", 1.5)
        max_speech = self.audio_config.get("max_speech_duration", 30.0)

        chunk_frames = int(sample_rate * chunk_duration)
        loop = asyncio.get_event_loop()
        max_buffer_bytes = 5 * 1024 * 1024  # 5MB hard limit on speech buffer

        log.info(
            f"[Audio] Capture started (rate={sample_rate}Hz, "
            f"vad_threshold={silence_threshold}, "
            f"silence={silence_duration}s)"
        )

        speech_buffer = bytearray()
        is_speaking = False
        silence_start = 0.0
        speech_start = 0.0
        consecutive_read_errors = 0
        max_read_errors = 30  # ~15s at 0.5s chunks

        while self._running:
            try:
                # Read audio chunk (blocking → run in executor)
                pcm_data = await loop.run_in_executor(
                    None, self.robot.read_audio_chunk, chunk_frames
                )
                if not pcm_data:
                    consecutive_read_errors += 1
                    if consecutive_read_errors >= max_read_errors:
                        log.error(
                            "[Audio] Microphone unresponsive — "
                            "attempting reinitialize"
                        )
                        self.robot._init_audio()
                        consecutive_read_errors = 0
                        if not self.robot._capture_stream:
                            log.error("[Audio] Reinit failed, waiting 30s")
                            await asyncio.sleep(30.0)
                    await asyncio.sleep(0.1)
                    continue
                consecutive_read_errors = 0

                # Echo suppression: skip capture during playback + tail
                if self._playing_audio or (
                    time.time() - self._audio_stop_time < self._echo_tail_s
                ):
                    continue

                # Compute RMS energy for VAD
                samples = np.frombuffer(pcm_data, dtype=np.int16)
                rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

                now = time.time()

                if rms >= silence_threshold:
                    # Speech detected
                    if not is_speaking:
                        is_speaking = True
                        speech_start = now
                        speech_buffer.clear()
                        log.debug(f"[Audio] Speech start (rms={rms:.0f})")
                    silence_start = 0.0
                    speech_buffer.extend(pcm_data)
                else:
                    # Silence
                    if is_speaking:
                        speech_buffer.extend(pcm_data)  # Include trailing silence
                        if silence_start == 0.0:
                            silence_start = now
                        elif (now - silence_start) >= silence_duration:
                            # End of speech — send to AGiXT
                            duration = now - speech_start
                            log.info(
                                f"[Audio] Speech end ({duration:.1f}s, "
                                f"{len(speech_buffer)} bytes)"
                            )
                            await self._send_speech_audio(
                                bytes(speech_buffer), sample_rate
                            )
                            speech_buffer.clear()
                            is_speaking = False
                            silence_start = 0.0

                # Force send if max speech duration or buffer size exceeded
                if is_speaking and (
                    (now - speech_start) >= max_speech
                    or len(speech_buffer) >= max_buffer_bytes
                ):
                    reason = (
                        "buffer overflow"
                        if len(speech_buffer) >= max_buffer_bytes
                        else f"max duration ({max_speech}s)"
                    )
                    log.info(f"[Audio] {reason} — sending")
                    await self._send_speech_audio(
                        bytes(speech_buffer), sample_rate
                    )
                    speech_buffer.clear()
                    is_speaking = False
                    silence_start = 0.0

            except Exception as e:
                log.debug(f"[Audio] Capture error: {e}")
                await asyncio.sleep(0.1)

    async def _send_speech_audio(self, pcm_data: bytes, sample_rate: int):
        """Encode PCM as WAV and send to AGiXT for transcription."""
        if not self._ws or not self._running:
            return

        # Encode as WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        wav_data = wav_buffer.getvalue()

        # Send WAV as binary frame, then signal end
        await self._ws.send(wav_data)
        await self._ws.send(json.dumps({"type": "audio.input.end"}))
        log.info(f"[Audio] Sent {len(wav_data)} bytes WAV to AGiXT")

    async def send_text(self, text: str):
        """Send a text message to AGiXT (for testing without audio)."""
        if self._ws and self._running:
            await self._ws.send(json.dumps({"type": "text.input", "text": text}))
            log.info(f"[Send] Text: {text}")

    async def send_audio(self, wav_data: bytes):
        """Send audio data to AGiXT."""
        if self._ws and self._running:
            await self._ws.send(wav_data)
            await self._ws.send(json.dumps({"type": "audio.input.end"}))
            log.info(f"[Send] Audio: {len(wav_data)} bytes")

    async def stop(self):
        """Clean shutdown."""
        self._running = False
        self._audio_interrupted = True
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass


# ─── Main ───────────────────────────────────────────────────────────────


async def main(config: dict):
    """Main entry point — connect to robot and AGiXT, run event loops."""

    # Initialize robot
    robot = Go2Robot(config)
    try:
        await robot.connect()
    except Exception as e:
        log.error(f"Failed to connect to robot: {e}")
        if not config.get("simulation"):
            log.info(
                "Tip: Set simulation=true in config or GO2_SIMULATION=true to test without robot"
            )
            return
        raise

    # Initialize AGiXT client
    client = AGiXTVoiceClient(config, robot)
    try:
        await client.connect()
    except Exception as e:
        log.error(f"Failed to connect to AGiXT: {e}")
        return

    log.info("=" * 50)
    log.info("Go2 Control is running!")
    log.info(
        "  Robot: Connected" + (" (simulation)" if config.get("simulation") else "")
    )
    log.info(f"  AGiXT: {config['agixt']['server']}")
    log.info("  Camera streaming: " + ("ON" if config["camera"]["enabled"] else "OFF"))
    log.info("  Audio capture: " + ("ON" if config["audio"]["enabled"] else "OFF"))
    log.info("=" * 50)

    # Run all loops concurrently
    tasks = [
        asyncio.ensure_future(client.run()),  # Main WS message handler
        asyncio.ensure_future(client.camera_loop()),  # Camera frame sender
        asyncio.ensure_future(client.audio_loop()),  # Audio capture
    ]

    # Also start an interactive text input loop for testing
    if sys.stdin.isatty():
        tasks.append(asyncio.ensure_future(_interactive_input(client)))

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Report any task failures
        task_names = ["message_handler", "camera_loop", "audio_loop"]
        if sys.stdin.isatty():
            task_names.append("interactive_input")
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                log.error(f"[Main] Task '{name}' failed: {result}")
    except KeyboardInterrupt:
        pass
    finally:
        await client.stop()
        robot.cleanup_audio()
        log.info("Go2 Control shutdown complete")


async def _interactive_input(client: AGiXTVoiceClient):
    """Read text input from stdin for testing."""
    loop = asyncio.get_event_loop()
    log.info("[Input] Type messages to send to AGiXT (or 'quit' to exit)")

    while client._running:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            line = line.strip()
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                await client.stop()
                break
            await client.send_text(line)
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Go2 Control — AGiXT Voice Client for Unitree Go2 Pro"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--simulation",
        "-s",
        action="store_true",
        help="Run in simulation mode (no real robot)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.simulation:
        config["simulation"] = True

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        log.info("Interrupted")
