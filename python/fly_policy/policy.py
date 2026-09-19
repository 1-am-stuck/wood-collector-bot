"""Learnable encoder + frozen connectome + decoder (ChessFly / FlyGM)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import settle
from .graph import FlyGraph
from sense.frame import frame_to_vector, vector_size


ACTIONS = [
    "forward", "back", "turn_left", "turn_right",
    "jump", "mine", "camera_up", "camera_down", "noop",
]


class FlyPolicy(nn.Module):
    def __init__(self, graph: FlyGraph, n_obs: int, n_actions: int = 9, steps: int = 5):
        super().__init__()
        self.graph = graph
        self.n = graph.n
        self.steps = steps
        self.n_obs = n_obs
        self.n_actions = n_actions
        src = torch.tensor(graph.src, dtype=torch.long)
        dst = torch.tensor(graph.dst, dtype=torch.long)
        sign = torch.tensor(graph.sign, dtype=torch.float32)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("sign", sign)
        sensory = torch.tensor(graph.sensory_idx, dtype=torch.long)
        dn = torch.tensor(graph.dn_idx, dtype=torch.long)
        self.register_buffer("sensory_idx", sensory)
        self.register_buffer("dn_idx", dn)

        self.encoder = nn.Linear(n_obs, int(sensory.numel()))
        self.log_gain = nn.Parameter(torch.zeros(graph.n_edges))
        self.gamma = nn.Parameter(torch.ones(self.n))
        self.mu = nn.Parameter(torch.zeros(self.n))
        self.log_sigma = nn.Parameter(torch.zeros(self.n))
        self.beta = nn.Parameter(torch.zeros(self.n))
        self.decoder = nn.Linear(int(dn.numel()), n_actions)
        self.value = nn.Linear(int(dn.numel()), 1)

    def encode_u(self, obs: torch.Tensor) -> torch.Tensor:
        currents = F.relu(self.encoder(obs))
        if obs.dim() == 1:
            u = obs.new_zeros(self.n)
            u[self.sensory_idx] = currents
        else:
            u = obs.new_zeros(obs.shape[0], self.n)
            u[:, self.sensory_idx] = currents
        return u

    def forward_hidden(self, obs: torch.Tensor) -> torch.Tensor:
        u = self.encode_u(obs)
        h0 = torch.zeros_like(u)
        return settle(
            h0, u, self.src, self.dst, self.log_gain, self.sign, self.n,
            steps=self.steps, gamma=self.gamma, mu=self.mu,
            sigma=torch.exp(self.log_sigma), beta=self.beta,
        )

    def forward(self, obs: torch.Tensor):
        h = self.forward_hidden(obs)
        dn = h.index_select(-1, self.dn_idx)
        logits = self.decoder(dn)
        value = self.value(dn).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, greedy: bool = False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, logits

    def frames_to_obs(self, frames) -> torch.Tensor:
        if isinstance(frames, dict):
            frames = [frames]
        vecs = [frame_to_vector(f) for f in frames]
        t = torch.tensor(vecs, dtype=torch.float32)
        return t.squeeze(0) if len(frames) == 1 else t

    def save_checkpoint(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "n_obs": self.n_obs,
            "n_actions": self.n_actions,
            "steps": self.steps,
            "n": self.n,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, graph: FlyGraph, path: str | Path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(graph, ckpt["n_obs"], ckpt["n_actions"], ckpt.get("steps", 5))
        model.load_state_dict(ckpt["state_dict"])
        return model


def default_graph_path(root: Path) -> Path:
    return root / "data" / "connectome" / "mini_male_cns.npz"


def build_policy(root: Path) -> FlyPolicy:
    from tools.build_mini_graph import build
    path = default_graph_path(root)
    if not path.exists():
        build()
    graph = FlyGraph(path)
    return FlyPolicy(graph, vector_size(64), len(ACTIONS))
