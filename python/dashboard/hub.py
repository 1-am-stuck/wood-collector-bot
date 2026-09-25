"""Thread-safe state for the dashboard.

Three things update at three different rates, so they are stored separately:

- `topology`: the graph itself -- neuron names, groups, body parts, action wiring.
  Written once when the run starts. On the full connectome this is the difference
  between a readable dashboard and 176,422 names down the socket every tick.
- `tick`: one policy decision (frame, facts, neuron rates). Control rate.
- `pose`: where the bot is and which way it faces. Stream rate, ~30 Hz.
- `voxels`: the surrounding surface blocks. Only when the bot moves out of the
  cached box, so the viewer can keep the geometry and just move the camera.
- `eye`: packed first-person RGB from `sampleEyeView`. For us and for an LLM;
  the policy never sees it.

The websocket sends pose every frame and the voxel snapshot only when `seq`
changes, which is what keeps a 30 Hz view affordable.
"""

from __future__ import annotations

import threading
from collections import deque

_lock = threading.Lock()
_latest: dict | None = None
_history: deque[dict] = deque(maxlen=240)
_pose: dict | None = None
_voxels: dict | None = None
_voxel_seq = 0
_eye: dict | None = None
_eye_seq = 0
_topology: dict | None = None


def reset() -> None:
    global _latest, _pose, _voxels, _voxel_seq, _eye, _eye_seq, _topology
    with _lock:
        _latest = None
        _pose = None
        _voxels = None
        _voxel_seq = 0
        _eye = None
        _eye_seq = 0
        _topology = None
        _history.clear()


def publish_topology(topology: dict) -> None:
    global _topology
    with _lock:
        _topology = topology


def topology() -> dict | None:
    with _lock:
        return _topology


def publish(tick: dict) -> dict:
    global _latest
    with _lock:
        _latest = tick
        _history.append(tick)
    return tick


def publish_stream(frame: dict) -> None:
    """Take a `{"stream": ...}` frame from the Node rollout."""
    global _pose, _voxels, _voxel_seq, _eye, _eye_seq
    kind = frame.get("stream")
    with _lock:
        if kind == "pose":
            _pose = frame.get("pose")
        elif kind == "voxels":
            _voxels = frame.get("voxels")
            _voxel_seq += 1
        elif kind == "eye":
            _eye = frame.get("eye")
            _eye_seq += 1


def latest() -> dict | None:
    with _lock:
        return _latest


def history() -> list[dict]:
    with _lock:
        return list(_history)


def pose() -> dict | None:
    with _lock:
        return _pose


def voxels() -> tuple[dict | None, int]:
    with _lock:
        return _voxels, _voxel_seq


def voxel_seq() -> int:
    with _lock:
        return _voxel_seq


def eye() -> dict | None:
    with _lock:
        return _eye


def eye_seq() -> int:
    with _lock:
        return _eye_seq
