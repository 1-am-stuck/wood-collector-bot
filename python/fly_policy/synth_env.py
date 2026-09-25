"""Synthetic Minecraft-like odor/vision taxis for IL, PPO, and eval (no MC server)."""

from __future__ import annotations

import math
import random

from sense.goal_to_sense import apply_goal, reward_for


def empty_frame(n=64):
    return {
        "luminance": [0.4] * n,
        "objects": [],
        "yawRateDegPerS": 0.0,
        "pitchRateDegPerS": 0.0,
        "odor": {},
        "odorBearingDeg": None,
        "taste": {},
        "windLeft": 0.0,
        "windRight": 0.0,
        "tilt": 0.0,
        "soundLow": 0.0,
        "soundHigh": 0.0,
        "song": 0.0,
        "touchHead": 0.0,
        "touchWing": 0.0,
        "touchLegs": 0.0,
        "touchNotum": 0.0,
        "touchAbdomen": 0.0,
        "groomDust": 0.0,
        "damage": 0.0,
        "hot": 0.0,
        "cold": 0.0,
        "dry": 0.0,
        "moist": 0.0,
        "airborne": False,
        "legsOnGround": True,
        "wingbeat": 0.0,
        "objectChannels": {k: 0.0 for k in ["LC4", "LPLC2", "LC11", "LC18", "LC10a", "LC15", "HS"]},
    }


class OdorTaxisEnv:
    """Agent on a plane; an oak_log sits at a random offset. Goal = collect oak."""

    ACTIONS = [
        "forward", "back", "turn_left", "turn_right",
        "jump", "mine", "camera_up", "camera_down", "noop",
    ]

    def __init__(self, goal: dict, seed: int = 0):
        self.goal = goal
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.x, self.z = 0.0, 0.0
        self.yaw = self.rng.uniform(-math.pi, math.pi)
        ang = self.rng.uniform(-math.pi, math.pi)
        dist = self.rng.uniform(6, 14)
        self.tx = math.sin(ang) * dist
        self.tz = math.cos(ang) * dist
        self.have = False
        self.steps = 0
        return self.observe()

    def _rel(self):
        dx = self.tx - self.x
        dz = self.tz - self.z
        d = math.hypot(dx, dz)
        if d < 1e-9:
            return 0.0, 0.0
        # yaw 0 = +Z; turn_left increases yaw. Heading error >0 means turn left.
        desired = math.atan2(dx, dz)
        err = (desired - self.yaw + math.pi) % (2 * math.pi) - math.pi
        # SensoryFrame: +odorBearingDeg = fly's right (fly-brain-minecraft).
        bearing_right = -err * 180 / math.pi
        return d, bearing_right

    def observe(self):
        d, bearing = self._rel()
        conc = math.exp(-d / 6.0)
        frame = empty_frame()
        frame["odor"] = {"DM1": 0.4 * conc, "DL5": 0.7 * conc, "VA3": 0.4 * conc}
        frame["odorBearingDeg"] = bearing
        if abs(bearing) < 35 and d < 20:
            frame["objectChannels"]["LC11"] = min(1.0, 8.0 / (1 + d))
            frame["objects"] = [{
                "azimuthDeg": bearing,
                "elevationDeg": 0,
                "angularSizeDeg": max(2.0, 40 / (1 + d)),
                "expansionDegPerS": 0,
                "angularSpeedDegPerS": 0,
                "contrast": 0.8,
                "flyLike": False,
                "name": "oak_log",
            }]
        world = {
            "position": {"x": self.x, "y": 64, "z": self.z},
            "blocks": [{"name": "oak_log", "x": self.tx, "y": 64, "z": self.tz}],
        }
        frame = apply_goal(frame, self.goal, world)
        facts = {
            "inventory": {"oak_log": 1 if self.have else 0},
            "closerToQuery": 0.0,
            "distance": d,
            "bearing": bearing,
        }
        return frame, facts, world

    def step(self, action: str):
        prev = self.observe()[1]
        d0, _ = self._rel()
        if action == "forward":
            self.x += 0.8 * math.sin(self.yaw)
            self.z += 0.8 * math.cos(self.yaw)
        elif action == "back":
            self.x -= 0.5 * math.sin(self.yaw)
            self.z -= 0.5 * math.cos(self.yaw)
        elif action == "turn_left":
            self.yaw += 15 * math.pi / 180
        elif action == "turn_right":
            self.yaw -= 15 * math.pi / 180
        elif action == "mine":
            d, bearing = self._rel()
            if d < 2.5 and abs(bearing) < 50:
                self.have = True
        self.steps += 1
        frame, facts, world = self.observe()
        d1, _ = self._rel()
        facts["closerToQuery"] = (d0 - d1) * 0.3
        r = reward_for(self.goal, prev, facts)
        done = self.have or self.steps >= 120
        return frame, r, done, facts

    def expert_action(self, facts: dict) -> str:
        if facts["distance"] < 2.5 and abs(facts["bearing"]) < 50:
            return "mine"
        if facts["bearing"] > 10:
            return "turn_right"
        if facts["bearing"] < -10:
            return "turn_left"
        return "forward"


def collect_expert_jsonl(goal, n_episodes=20, seed=0):
    rows = []
    for ep in range(n_episodes):
        env = OdorTaxisEnv(goal, seed=seed + ep)
        frame, facts, _ = env.reset()
        done = False
        while not done:
            act = env.expert_action(facts)
            rows.append({"frame": frame, "action": act, "goal_id": goal.get("id"), "facts": facts})
            frame, _, done, facts = env.step(act)
    return rows
