# go2control

> **⚠️ Experimental** — This project is under active development and may contain bugs or incomplete features. Use at your own risk.

REST + WebSocket API for controlling the Unitree Go2 Pro robot dog, with AGiXT AI agent integration.

## Features

- **28 Sport Actions**: Walk, run, sit, tricks, dances, flips, poses
- **Velocity Control**: Forward/backward, lateral, rotation with safety ramping
- **Camera**: HD front camera snapshot capture
- **Speaker**: Volume control, audio playback
- **Microphone**: Audio capture for transcription
- **Body Orientation**: Roll, pitch, yaw control
- **Sequences**: Pre-built and custom movement routines
- **WebSocket**: Real-time telemetry streaming
- **Dashboard**: Web-based control panel with virtual joysticks
- **AGiXT Integration**: Full AI agent context and command endpoints

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# For WebRTC connection (recommended, no jailbreak needed):
pip install unitree_webrtc_connect

# For DDS connection (wired Ethernet):
# Requires cyclonedds==0.10.2 and unitree_sdk2py

# Edit config
cd go2control
cp config.yaml config.yaml.bak
nano config.yaml  # Set simulation: false, robot_ip, connection_mode

# Run
python3 api_server.py
```

## Connection Methods

### WebRTC (Recommended)
Works wirelessly with all Go2 models (AIR/PRO/EDU). No jailbreak needed.

| Mode | Use Case | Config |
|------|----------|--------|
| `LocalAP` | Connected to robot's WiFi AP | `robot_ip: 192.168.12.1` |
| `LocalSTA` | Same WiFi network as robot | `robot_ip: <robot's IP>` |
| `Remote` | Via Unitree TURN server (requires 4G) | `serial_number: <SN>` |

### DDS (Wired)
Requires Ethernet connection to robot's internal computer at `192.168.123.161`.

## API Endpoints

### Movement
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/move` | Velocity control (vx, vy, vyaw, duration) |
| POST | `/api/v1/stop` | Stop all movement |
| POST | `/api/v1/euler` | Set body orientation |
| POST | `/api/v1/speed_level` | Set speed (-1/0/1) |
| POST | `/api/v1/emergency_stop` | DAMP all motors |

### Status
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/status` | Robot mode, velocity, battery, connections |

### Actions (28)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/actions` | List all sport actions |
| POST | `/api/v1/action/{name}` | Execute a sport action |

### Camera
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/camera/snapshot` | JPEG image |
| GET | `/api/v1/camera/snapshot/base64` | Base64 JPEG |

### Audio
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/audio/volume` | Get volume |
| POST | `/api/v1/audio/volume` | Set volume (0-10) |
| GET | `/api/v1/audio/mic?seconds=5` | Capture mic audio |

### Sequences
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/sequences` | List all sequences |
| GET | `/api/v1/sequences/{name}` | Get sequence definition |
| POST | `/api/v1/sequences/{name}` | Save a custom sequence |
| DELETE | `/api/v1/sequences/{name}` | Delete a user-saved sequence |
| POST | `/api/v1/sequences/{name}/run` | Run a named sequence |
| POST | `/api/v1/sequences/stop` | Stop active sequence |

### Agent (AGiXT)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/agent/context` | Full robot context for AI |
| POST | `/api/v1/agent/command` | Unified agent command endpoint |

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/dashboard` | Web control panel |
| WS | `/ws/telemetry` | Real-time WebSocket |

## AGiXT Extension

The AGiXT extension for this robot is in the [unitree_extensions](https://github.com/Josh-XT/unitree_extensions) repo. Install it with:

```bash
agixt env EXTENSIONS_HUB=https://github.com/Josh-XT/unitree_extensions
agixt restart
```

Then configure your agent with `GO2_API_URL=http://<go2control-host>:8000`.

## Sport Actions

### State Transitions
`damp`, `balance_stand`, `stop_move`, `stand_up`, `stand_down`, `recovery_stand`, `sit`, `rise_sit`

### Tricks & Animations
`hello`, `stretch`, `content`, `dance1`, `dance2`, `pose`, `scrape`, `heart`

### Flips & Jumps
`front_flip`, `left_flip`, `back_flip`, `front_jump`, `front_pounce`, `hand_stand`

### Gaits
`static_walk`, `trot_run`, `economic_gait`

### Modes
`free_walk`, `cross_step`, `switch_joystick`

## Project Structure

```text
go2control/
├── api_server.py        # FastAPI server
├── config.py            # YAML + env var config loader
├── config.yaml          # Default configuration
├── sequence_library.py  # Built-in + user sequence management
├── dashboard.html       # Web control panel
├── go2control.service   # systemd unit file
├── sequences/           # User-saved sequences (JSON)
└── requirements.txt     # Python dependencies
```

## Hardware

- **Model**: Unitree Go2 Pro
- **CPU**: 8-core high-performance processor
- **Camera**: HD wide-angle front camera
- **Audio**: Built-in speaker + microphone
- **Connectivity**: WiFi 6, Bluetooth 5.2, 4G (with GPS)
- **Battery**: ~2 hours runtime

## Related Repositories

- **[g1control](https://github.com/Josh-XT/g1control)** — Control server for the Unitree G1 Basic humanoid
- **[unitree_extensions](https://github.com/Josh-XT/unitree_extensions)** — AGiXT extensions for both robots
- **[AGiXT](https://github.com/Josh-XT/AGiXT)** — AI agent framework

## Contributing

This project is experimental and we welcome contributions! If you find a bug or have a suggestion:

- **Report issues** on the [GitHub Issues](https://github.com/Josh-XT/go2control/issues) page
- **Pull requests** are always welcome — if you find an issue you can fix, we'd love the help

## License

MIT — Use at your own risk. This is unofficial/experimental code.
