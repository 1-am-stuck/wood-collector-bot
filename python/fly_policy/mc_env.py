"""JSONL Mineflayer env. Lives in a real Paper world — no synthetic stand-in."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MinecraftEnv:
    """Owns the Node rollout process.

    The rollout speaks two kinds of line on stdout: replies to our commands, and
    unsolicited `{"stream": ...}` frames carrying pose at ~30 Hz plus the voxel
    neighbourhood when it changes. A reader thread keeps draining stdout so those
    frames reach the dashboard while the control loop is sleeping between steps,
    instead of piling up in the pipe and arriving as a burst.
    """

    def __init__(self, cfg: dict, node_script: Path | None = None, on_stream=None):
        self.cfg = cfg
        self.node_script = Path(node_script) if node_script else ROOT / "js" / "sense" / "mc_rollout.js"
        self.proc: subprocess.Popen | None = None
        self._err: list[str] = []
        self._replies: queue.Queue = queue.Queue()
        self._on_stream = on_stream
        self._reader: threading.Thread | None = None
        self._alive = threading.Event()

    def start(self) -> dict:
        self.proc = subprocess.Popen(
            ["node", str(self.node_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._alive.set()
        threading.Thread(target=self._drain_err, daemon=True).start()
        self._reader = threading.Thread(target=self._drain_out, daemon=True, name="mc-stream")
        self._reader.start()
        catalog = (self.cfg.get("goal") or {}).get("catalog")
        if isinstance(catalog, (str, Path)):
            from train_config import load_goal_catalog
            catalog = load_goal_catalog(catalog)
        radius = (self.cfg.get("goal") or {}).get("explore_radius", 32)
        self._rpc({"cmd": "load_catalog", "catalog": catalog or {}, "explore_radius": radius})
        payload = {"cmd": "connect", "minecraft": self.cfg.get("minecraft") or {}}
        view = self.cfg.get("view") or {}
        if view.get("stream"):
            payload["view"] = view["stream"]
        # `mode: navigate` scores covering ground with no goal, and switches off the
        # census and the planted trees on the Node side, so it must reach `connect`.
        for key in ("seed_woods", "seed_rich", "census", "mode"):
            if key in self.cfg:
                payload[key] = self.cfg[key]
        return self._rpc(payload)

    def reset(self) -> dict:
        return self._rpc({"cmd": "reset"})

    def play_reset(self) -> dict:
        return self._rpc({"cmd": "play_reset"})

    def observe(self) -> dict:
        return self._rpc({"cmd": "observe"})

    def step(self, action: str) -> dict:
        return self._rpc({"cmd": "step", "action": action})

    def close(self) -> None:
        if not self.proc:
            return
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
        self._alive.clear()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def _drain_err(self) -> None:
        if not self.proc or not self.proc.stderr:
            return
        for line in self.proc.stderr:
            self._err.append(line.rstrip())

    def _drain_out(self) -> None:
        """Route stream frames to the viewer, command replies to the caller."""
        if not self.proc or not self.proc.stdout:
            return
        for line in self.proc.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                self._err.append(raw)
                continue
            if obj.get("stream"):
                if self._on_stream is not None:
                    try:
                        self._on_stream(obj)
                    except Exception as exc:  # a bad viewer must not stop training
                        self._err.append(f"stream handler: {exc}")
                continue
            self._replies.put(obj)
        self._replies.put({"error": "mc_rollout closed"})

    def _rpc(self, msg: dict, retries: int = 1, timeout: float = 120.0) -> dict:
        if not self.proc or self.proc.stdin is None:
            raise RuntimeError("minecraft env is not started")
        if self.proc.poll() is not None:
            raise RuntimeError("mc_rollout exited: " + "\n".join(self._err[-20:]))
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        while True:
            try:
                obj = self._replies.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeError(
                    f"mc_rollout did not answer {msg.get('cmd')} in {timeout:.0f}s: "
                    + "\n".join(self._err[-10:])
                ) from None
            if obj.get("error") == "mc_rollout closed":
                raise RuntimeError("mc_rollout closed: " + "\n".join(self._err[-20:]))
            if obj.get("error"):
                err = str(obj["error"])
                if retries and "null" in err:
                    mc = self.cfg.get("minecraft") or {}
                    try:
                        self._rpc({"cmd": "connect", "minecraft": mc}, retries=0)
                    except Exception:
                        pass
                    return self._rpc(msg, retries=retries - 1)
                raise RuntimeError(err)
            return obj
