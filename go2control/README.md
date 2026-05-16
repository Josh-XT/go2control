# go2control

> **⚠️ Experimental** — This project is under active development and may contain bugs or incomplete features. Use at your own risk.

REST + WebSocket API for controlling the Unitree Go2 Pro robot dog, with AGiXT AI agent integration. Includes a standalone voice conversation client (`go2_client.py`) that runs on-robot or on a Raspberry Pi for real-time AI-powered voice interaction with autonomous behaviors.

## Features

- **28 Sport Actions**: Walk, run, sit, tricks, dances, flips, poses
- **Velocity Control**: Forward/backward, lateral, rotation with safety ramping
- **Camera**: HD front camera snapshot capture via DDS
- **Speaker**: Volume control, audio playback
- **Microphone**: Audio capture with VAD for voice conversations
- **Body Orientation**: Roll, pitch, yaw control
- **Sequences**: Pre-built and custom movement routines
- **WebSocket**: Real-time telemetry streaming
- **Dashboard**: Web-based control panel with virtual joysticks
- **AGiXT Integration**: Full AI agent voice conversation with tool-calling for robot control
- **Wake Word**: Configurable wake word detection (e.g., "hey robot")
- **Idle Personality**: Autonomous look-around, animations, and contextual comments when idle
- **Vision**: Periodic camera frames sent to AGiXT for visual understanding
- **Signed Identity Evidence**: HMAC-bound face/voice envelopes with monotonic sequence numbers for WorkConductor server-side identity decisions

## Quick Start

### API Server (REST + Dashboard)

```bash
pip install -r requirements.txt

# For WebRTC connection (recommended, no jailbreak needed):
pip install unitree_webrtc_connect

# For DDS connection (wired Ethernet):
# Requires cyclonedds==0.10.2 and unitree_sdk2py

# Edit config
cp config.yaml config.yaml.bak
nano config.yaml  # Set simulation: false, robot_ip, connection_mode

# Run
python3 api_server.py
```

### Voice Conversation Client (go2_client.py)

```bash
# Edit config.yaml with your AGiXT server URL and JWT
nano config.yaml

# Run
python3 go2_client.py --config config.yaml
```

See the [Raspberry Pi Deployment](#raspberry-pi-deployment) section below for running the voice client on a Pi mounted on the robot.

## Connection Methods

### WebRTC (Recommended for API Server)
Works wirelessly with all Go2 models (AIR/PRO/EDU). No jailbreak needed.

| Mode | Use Case | Config |
|------|----------|--------|
| `LocalAP` | Connected to robot's WiFi AP | `robot_ip: 192.168.12.1` |
| `LocalSTA` | Same WiFi network as robot | `robot_ip: <robot's IP>` |
| `Remote` | Via Unitree TURN server (requires 4G) | `serial_number: <SN>` |

### DDS (Wired — Required for go2_client.py)
Requires Ethernet connection to the Go2's internal computer at `192.168.123.161`. This is the only connection method for `go2_client.py` as it provides low-latency DDS access to the sport client, camera, and audio subsystems.

## Raspberry Pi Deployment

The voice conversation client (`go2_client.py`) is designed to run on a Raspberry Pi 4B or 5 physically mounted on the Go2 robot, powered from the robot's battery. This gives the robot autonomous AI voice interaction, camera-based vision, wake word detection, and idle personality behaviors.

### Tested Hardware

| Component | Details |
|-----------|---------|
| **Compute** | Raspberry Pi 4B (4GB) or Pi 5 (any RAM config) |
| **OS** | Raspberry Pi OS Bookworm (64-bit / aarch64) |
| **Power** | Go2 battery (~20V) → Buck converter → 5V GPIO |
| **Network** | Ethernet to Go2, WiFi to LAN |
| **Audio** | USB microphone + USB speaker (optional: 3.5mm speaker via headphone jack) |

### Power Wiring

The Go2's internal battery provides ~20V. Use a buck converter to step this down to 5V for the Pi.

**Recommended buck converter**: DROK B078Q1624B (5.3-32V input, 1.2-32V output, 12A max)

1. **Set output voltage first** — Before connecting to the Pi, connect the buck converter to a power supply and adjust the trim pot until the output reads **5.0-5.1V** with a multimeter
2. **Connect input** — Wire the buck converter input to the Go2's battery leads (observe polarity)
3. **Connect output to Pi GPIO** — Wire the buck converter 5V output to:
   - **Pin 4** (5V) — Red/positive wire
   - **Pin 6** (GND) — Black/negative wire
4. **Verify** — Before powering on, double-check voltage at the GPIO pins with a multimeter

> **⚠️ Important**: Do NOT use USB-C power delivery from a barrel-to-USB-C adapter — most adapters lack the required 5.1kΩ CC pull-down resistors and the Pi will refuse to boot. Wire directly to GPIO pins 4 and 6.

> **⚠️ Warning**: GPIO power bypasses the Pi's onboard voltage regulator and overcurrent protection. Ensure your buck converter output is stable at 5.0-5.1V before connecting. Incorrect voltage can permanently damage the Pi.

### Network Configuration

The Pi needs **two networks**:
- **Ethernet** (`eth0`): Static IP on the Go2's DDS subnet (`192.168.123.x`)
- **WiFi** (`wlan0`): Your LAN for AGiXT server access and SSH

#### Set static Ethernet IP

```bash
# Create a NetworkManager connection for eth0
sudo nmcli con add type ethernet con-name go2-dds ifname eth0 \
  ipv4.addresses 192.168.123.100/24 \
  ipv4.method manual \
  ipv4.never-default yes

# Activate it
sudo nmcli con up go2-dds
```

The `ipv4.never-default yes` flag is critical — it prevents the Go2 subnet from becoming the default route, which would break WiFi internet access.

#### Verify connectivity

```bash
# Ping the Go2
ping -c 3 192.168.123.161

# Should see ~0.3-0.5ms latency
```

### Software Setup

#### 1. System packages

```bash
sudo apt update && sudo apt install -y \
  python3-pip python3-venv python3-dev \
  portaudio19-dev libopencv-dev \
  cmake build-essential
```

#### 2. Build CycloneDDS from source

CycloneDDS 0.10.2 is not available as a Debian package for arm64 and must be built from source:

```bash
cd /tmp
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### 3. Python virtual environment

```bash
python3 -m venv ~/go2env
source ~/go2env/bin/activate

pip install websockets httpx numpy opencv-python-headless pyyaml pyaudio
pip install cyclonedds==0.10.2
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
```

**Note**: If you get an import error about `unitree_sdk2py.b2`, edit the SDK's `__init__.py` to remove the `b2` import:

```bash
# Find and edit the file
nano ~/go2env/lib/python3.11/site-packages/unitree_sdk2py/__init__.py
# Remove or comment out: from . import b2
```

The `b2` module is for the B2 robot and is not included in the Go2-only SDK build.

#### 4. Clone and configure

```bash
cd ~
git clone https://github.com/Josh-XT/go2control.git
cd go2control/go2control
cp config.yaml config.yaml.bak
nano config.yaml
```

Key config settings for Pi deployment:

```yaml
agixt:
  server: "wss://your-agixt-server.com"
  api_key: "your-jwt-token"
  agent_name: "XT"

robot:
  connection_mode: "dds"
  ip: "192.168.123.161"
  dds_domain_id: 0
  dds_interface: "eth0"

wake_word:
  enabled: true
  phrase: "hey robot"
  server: "http://your-wake-word-server:8091"
  listen_duration: 10.0

idle:
  enabled: true
  look_around_interval: 30.0
  animation_interval: 120.0
  comment_interval: 180.0

audio:
  capture_rate: 16000
  playback_rate: 24000

identity_evidence:
  enabled: true
  company_id: "<company-id>"
  machine_id: "<machine-id>"
  key_id: "<workconductor-evidence-key-id>"
  signing_secret: "<hmac-secret-provisioned-by-workconductor>"
  algorithm: "hmac_sha256"
  max_face_evidence_bytes: 262144
  max_voice_evidence_bytes: 524288
```

The identity evidence keys can also be provided with `XTS_COMPANY_ID`, `XTS_MACHINE_ID`, `XTS_EVIDENCE_KEY_ID`, and `XTS_EVIDENCE_SIGNING_SECRET`. The robot never submits trusted biometric scores; it sends signed media envelopes and follows WorkConductor's `identity.updated` and `identity.evidence_steering` messages.

#### 5. Install systemd service

```bash
sudo cp go2control.service /etc/systemd/system/
sudo nano /etc/systemd/system/go2control.service
# Verify paths match your setup (User, WorkingDirectory, ExecStart, venv path)
sudo systemctl daemon-reload
sudo systemctl enable go2control
sudo systemctl start go2control
```

The service file sets the required environment variables:

```ini
[Service]
Environment=CYCLONEDDS_HOME=/usr/local
Environment=LD_LIBRARY_PATH=/usr/local/lib
```

#### 6. Verify

```bash
# Check service status
sudo systemctl status go2control

# Watch logs
journalctl -u go2control -f
```

You should see:

```
[Robot] DDS connection established
[AGiXT] Connected! Conversation: <uuid>
[AGiXT] Registered 7 robot tools with identity
Go2 Control is running!
  Robot: Connected
  Camera streaming: ON
  Audio capture: ON
  Wake word: 'hey robot'
  Idle personality: ON
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `[ClientStub] send request error` | Normal DDS SDK messages during init — safe to ignore |
| ALSA/JACK warnings on startup | Normal when no USB audio device is connected |
| Pi won't boot via USB-C | Use GPIO pin 4/6 power instead (see Power Wiring above) |
| `ModuleNotFoundError: b2` | Edit `unitree_sdk2py/__init__.py` to remove `from . import b2` |
| DDS can't find interface | Verify `eth0` has IP `192.168.123.x` with `ip addr show eth0` |
| Can't ping 192.168.123.161 | Check Ethernet cable to Go2, verify static IP config |
| WebSocket won't connect | Check AGiXT server URL and JWT token in config.yaml |

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
├── api_server.py        # FastAPI REST server + dashboard
├── go2_client.py        # AGiXT voice conversation client (runs on Pi)
├── config.py            # YAML + env var config loader
├── config.yaml          # Default configuration
├── sequence_library.py  # Built-in + user sequence management
├── dashboard.html       # Web control panel
├── go2control.service   # systemd unit file
├── sequences/           # User-saved sequences (JSON)
└── requirements.txt     # Python dependencies
```

## Hardware

### Go2 Robot
- **Model**: Unitree Go2 Pro
- **CPU**: 8-core high-performance processor
- **Camera**: HD wide-angle front camera
- **Audio**: Built-in speaker + microphone
- **Connectivity**: WiFi 6, Bluetooth 5.2, 4G (with GPS)
- **Battery**: ~2 hours runtime, ~20V output

### Raspberry Pi (for on-robot deployment)
- **Model**: Raspberry Pi 4B (4GB+) or Pi 5
- **OS**: Raspberry Pi OS Bookworm (64-bit)
- **Power**: 5V from Go2 battery via buck converter to GPIO
- **Ethernet**: Direct to Go2 for DDS (192.168.123.x subnet)
- **WiFi**: LAN access for AGiXT server and SSH
- **Audio**: USB microphone + USB speaker

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
