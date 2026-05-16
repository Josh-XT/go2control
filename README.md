# Go2 Control — AGiXT Voice Client for Unitree Go2 Pro

Voice-controlled robot dog using AGiXT's real-time voice conversation pipeline. Runs on a Raspberry Pi 5 connected to the Go2 via Ethernet (DDS SDK). The Pi bridges the robot's camera and microphone to AGiXT, which sees what the robot sees, hears what it hears, and controls it through voice commands.

## Architecture

```
┌──────────────┐    DDS/Ethernet    ┌──────────────┐    WebSocket    ┌──────────────┐
│  Unitree Go2 │◄──────────────────►│  Raspberry   │◄──────────────►│   AGiXT      │
│  Pro Robot   │  SDK commands,     │  Pi 5        │  Audio, images │   Server     │
│              │  camera, audio     │  (go2_client)│  tool calls    │  (35B + 0.8B)│
└──────────────┘                    └──────────────┘                └──────────────┘
```

**Data flows:**
- **Camera → AGiXT:** Periodic JPEG frames from Go2 camera sent as vision context
- **Microphone → AGiXT:** Audio from Go2 mic (or Pi USB mic) sent for STT
- **Identity evidence → WorkConductor:** Signed, sequence-protected face/voice evidence envelopes for server-side matching and policy decisions
- **AGiXT → Robot:** Tool calls for movement, actions, body orientation
- **AGiXT → Speaker:** TTS audio streamed back for robot to speak

## Quick Start

### 1. Install on Pi 5

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `go2control/config.yaml` or set environment variables:

```bash
export AGIXT_SERVER="ws://your-agixt-server:7437"
export AGIXT_JWT="your-jwt-token"
export GO2_IP="192.168.123.161"
export GO2_CONNECTION="dds"
export XTS_COMPANY_ID="your-company-id"
export XTS_MACHINE_ID="your-machine-id"
export XTS_EVIDENCE_KEY_ID="provisioned-key-id"
export XTS_EVIDENCE_SIGNING_SECRET="provisioned-hmac-secret"
```

Identity evidence stays server-authoritative: the robot signs bounded media envelopes and WorkConductor performs matching, liveness policy, replay checks, and allow/deny/challenge decisions. Set `AGIXT_IDENTITY_EVIDENCE=false` only for local development without a provisioned evidence signing key.

### 3. Run

```bash
# With real robot (Ethernet connected)
python go2control/go2_client.py

# Simulation mode (no robot needed)
python go2control/go2_client.py --simulation

# Custom config file
python go2control/go2_client.py --config my_config.yaml
```

### 4. Test with text input

When running interactively, type messages to send to AGiXT:
```
> Walk forward slowly
> Turn left and look around
> Do a hello trick
> What do you see in front of you?
> Walk to that ball
```

## Available Robot Commands

AGiXT can call these via the voice conversation tool bridge:

| Tool | Description |
|------|-------------|
| `robot_move` | Velocity control (vx, vy, vyaw, duration) |
| `robot_action` | Sport actions (sit, stand, hello, dance, flip, etc.) |
| `robot_set_body_euler` | Body orientation (roll, pitch, yaw) |
| `robot_capture_image` | Capture fresh camera image |
| `robot_set_speed_level` | Speed: 0=slow, 1=medium, 2=fast |
| `robot_set_volume` | Speaker volume (0-10) |

### Sport Actions

`balance_stand`, `sit`, `stand_up`, `stand_down`, `hello`, `stretch`, `dance1`, `dance2`, `heart`, `pose`, `front_flip`, `back_flip`, `left_flip`, `hand_stand`, `free_walk`, and more.

## Connection Modes

### DDS (Recommended for Pi 5)
- Connect Pi to Go2 via Ethernet cable
- Robot IP: `192.168.123.161` (default)
- Requires `unitree_sdk2py` package
- Most reliable, lowest latency

### WebRTC (Future)
- WiFi connection to Go2's access point
- Robot IP: `192.168.12.1` (AP mode)
- Requires `unitree_webrtc_connect` package (not yet Pi5 compatible)

## How "Walk to the Ball" Works

1. User says "Walk to that ball"
2. AGiXT sees the ball in the camera feed (via periodic `image.input` frames)
3. AGiXT uses vision to identify the ball's position in frame
4. AGiXT calls `robot_move` to walk forward
5. AGiXT calls `robot_capture_image` to check progress
6. AGiXT adjusts direction based on new image
7. Repeat until close enough
8. AGiXT narrates progress through TTS: "I can see the ball, walking toward it..."
