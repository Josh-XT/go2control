"""
Pre-built sequence library for common Go2 routines.

Sequences are stored as JSON files in the sequences/ directory.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SEQUENCES_DIR = Path(__file__).parent / "sequences"

BUILTIN_SEQUENCES: dict[str, dict] = {
    "demo": {
        "name": "demo",
        "steps": [
            {"action": "action", "params": {"name": "recovery_stand"}, "duration": 2.0},
            {"action": "action", "params": {"name": "hello"}, "duration": 3.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 2.0},
            {"action": "move", "params": {"vyaw": 0.8}, "duration": 1.5},
            {"action": "move", "params": {"vx": 0.5}, "duration": 2.0},
            {"action": "action", "params": {"name": "stop_move"}, "duration": 1.0},
            {"action": "action", "params": {"name": "heart"}, "duration": 3.0},
        ],
        "loop": False,
    },
    "dance_show": {
        "name": "dance_show",
        "steps": [
            {"action": "action", "params": {"name": "balance_stand"}, "duration": 2.0},
            {"action": "action", "params": {"name": "dance1"}, "duration": 8.0},
            {"action": "action", "params": {"name": "stretch"}, "duration": 4.0},
            {"action": "action", "params": {"name": "dance2"}, "duration": 8.0},
            {"action": "action", "params": {"name": "content"}, "duration": 3.0},
            {"action": "action", "params": {"name": "heart"}, "duration": 3.0},
        ],
        "loop": False,
    },
    "tricks": {
        "name": "tricks",
        "steps": [
            {"action": "action", "params": {"name": "balance_stand"}, "duration": 2.0},
            {"action": "action", "params": {"name": "hello"}, "duration": 3.0},
            {"action": "action", "params": {"name": "pose"}, "duration": 3.0},
            {"action": "action", "params": {"name": "stretch"}, "duration": 3.0},
            {"action": "action", "params": {"name": "sit"}, "duration": 3.0},
            {"action": "action", "params": {"name": "rise_sit"}, "duration": 2.0},
            {"action": "action", "params": {"name": "scrape"}, "duration": 3.0},
        ],
        "loop": False,
    },
    "walk_square": {
        "name": "walk_square",
        "steps": [
            {"action": "action", "params": {"name": "balance_stand"}, "duration": 2.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 3.0},
            {"action": "move", "params": {"vyaw": 1.57}, "duration": 1.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 3.0},
            {"action": "move", "params": {"vyaw": 1.57}, "duration": 1.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 3.0},
            {"action": "move", "params": {"vyaw": 1.57}, "duration": 1.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 3.0},
            {"action": "action", "params": {"name": "stop_move"}, "duration": 1.0},
        ],
        "loop": False,
    },
    "patrol": {
        "name": "patrol",
        "steps": [
            {"action": "action", "params": {"name": "balance_stand"}, "duration": 2.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 5.0},
            {"action": "move", "params": {"vyaw": 1.57}, "duration": 2.0},
            {"action": "move", "params": {"vx": 0.5}, "duration": 5.0},
            {"action": "move", "params": {"vyaw": 1.57}, "duration": 2.0},
        ],
        "loop": True,
    },
    "getup_routine": {
        "name": "getup_routine",
        "steps": [
            {"action": "action", "params": {"name": "damp"}, "duration": 1.0},
            {"action": "wait", "params": {}, "duration": 1.0},
            {"action": "action", "params": {"name": "recovery_stand"}, "duration": 3.0},
            {"action": "action", "params": {"name": "balance_stand"}, "duration": 2.0},
        ],
        "loop": False,
    },
    "look_around": {
        "name": "look_around",
        "steps": [
            {"action": "action", "params": {"name": "balance_stand"}, "duration": 1.0},
            {"action": "euler", "params": {"yaw": 0.5}, "duration": 2.0},
            {"action": "euler", "params": {"yaw": -0.5}, "duration": 2.0},
            {"action": "euler", "params": {"pitch": 0.3}, "duration": 1.5},
            {"action": "euler", "params": {"pitch": -0.3}, "duration": 1.5},
            {"action": "euler", "params": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, "duration": 1.0},
        ],
        "loop": False,
    },
}


class SequenceLibrary:
    """Load, save, and list named sequences."""

    def __init__(self, sequences_dir: Optional[Path] = None):
        self._dir = sequences_dir or SEQUENCES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> dict[str, dict]:
        result = dict(BUILTIN_SEQUENCES)
        for f in self._dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                name = data.get("name", f.stem)
                result[name] = data
            except Exception as e:
                logger.warning("Failed to load sequence %s: %s", f, e)
        return result

    def get(self, name: str) -> Optional[dict]:
        if name in BUILTIN_SEQUENCES:
            return BUILTIN_SEQUENCES[name]
        path = self._dir / f"{name}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.warning("Failed to load sequence %s: %s", path, e)
        return None

    def save(self, name: str, data: dict) -> bool:
        data["name"] = name
        path = self._dir / f"{name}.json"
        try:
            path.write_text(json.dumps(data, indent=2))
            return True
        except OSError as e:
            logger.error("Failed to save sequence %s: %s", path, e)
            return False

    def delete(self, name: str) -> bool:
        if name in BUILTIN_SEQUENCES:
            return False
        path = self._dir / f"{name}.json"
        if path.exists():
            path.unlink()
            return True
        return False
