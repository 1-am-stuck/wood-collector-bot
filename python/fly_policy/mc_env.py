"""JSONL Mineflayer env. Lives in a real Paper world — no synthetic stand-in."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MinecraftEnv:
    def __init__(self, cfg: dict, node_script: Path | None = None):
        self.cfg = cfg
        self.node_script = Path(node_script) if node_script else ROOT / "js" / "sense" / "mc_rollout.js"
        self.proc: subprocess.Popen | None = None
        self._err: list[str] = []

    def start(self) -> dict:
        self.proc = subprocess.Popen(
            ["node", str(self.node_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain_err, daemon=True).start()
        catalog = (self.cfg.get("goal") or {}).get("catalog")
        if isinstance(catalog, (str, Path)):
            from train_config import load_goal_catalog
            catalog = load_goal_catalog(catalog)
        radius = (self.cfg.get("goal") or {}).get("explore_radius", 32)
        self._rpc({"cmd": "load_catalog", "catalog": catalog or {}, "explore_radius": radius})
        return self._rpc({"cmd": "connect", "minecraft": self.cfg.get("minecraft") or {}})

    def reset(self) -> dict:
        return self._rpc({"cmd": "reset"})

    def step(self, action: str) -> dict:
        return self._rpc({"cmd": "step", "action": action})

    def close(self) -> None:
        if not self.proc:
            return
        try:
            self._rpc({"cmd": "close"})
        except Exception:
            pass
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

    def _rpc(self, msg: dict) -> dict:
        if not self.proc or self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("minecraft env is not started")
        if self.proc.poll() is not None:
            raise RuntimeError("mc_rollout exited: " + "\n".join(self._err[-20:]))
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("mc_rollout closed: " + "\n".join(self._err[-20:]))
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                self._err.append(raw)
                continue
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            return obj
