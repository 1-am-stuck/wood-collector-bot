"""Learnable encoder + frozen connectome + decoder (ChessFly / FlyGM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamics import settle, sparse_apply
from .graph import FlyGraph
from sense.frame import frame_to_vector, vector_size
from sense.routing import ACTION_POPULATIONS, DESCENDING_WHAT

ACTIONS = list(ACTION_POPULATIONS.keys())


def encoder_mask(graph: FlyGraph, n_obs: int, n_retina: int | None = None) -> torch.Tensor:
    """0/1 mask over encoder weights: a sensory unit may only read its own modality.

    Rows follow `graph.sensory_idx`. The routing comes out of the graph, which recorded
    it when the subgraph was derived, keyed by channel name and resolved here against
    the current frame layout. A unit with no routing entry reads nothing, so a graph
    built against a different frame layout goes quiet rather than silently turning the
    encoder into a dense MLP that can read any channel from any receptor.
    """
    rays = n_retina if n_retina is not None else (graph.n_retina or 64)
    routing = graph.obs_routing(rays)
    if not routing:
        raise ValueError(
            f"{graph.path} carries no sensory routing, so there is nothing to say "
            "which channel each receptor is allowed to read."
        )
    mask = torch.zeros(len(graph.sensory_idx), n_obs)
    for row, node in enumerate(graph.sensory_idx):
        for col in routing.get(graph.names[int(node)], ()):
            if 0 <= col < n_obs:
                mask[row, col] = 1.0
    return mask


def decoder_mask(graph: FlyGraph, n_actions: int) -> torch.Tensor:
    """0/1 mask over decoder weights: an action may only read its own DN / motor unit."""
    dn_names = [graph.names[int(i)] for i in graph.dn_idx]
    units = graph.action_units
    if not units:
        raise ValueError(f"{graph.path} carries no action units, so no action can be read.")
    index = {name: i for i, name in enumerate(dn_names)}
    mask = torch.zeros(n_actions, len(dn_names))
    for row, action in enumerate(ACTIONS[:n_actions]):
        for unit in units.get(action, ()):
            if unit in index:
                mask[row, index[unit]] = 1.0
    return mask


class FlyPolicy(nn.Module):
    def __init__(self, graph: FlyGraph, n_obs: int, n_actions: int = 9, steps: int = 5,
                 n_retina: int | None = None):
        super().__init__()
        self.graph = graph
        self.n = graph.n
        self.steps = steps
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_retina = n_retina if n_retina is not None else (graph.n_retina or 64)
        src = torch.tensor(graph.src, dtype=torch.long)
        dst = torch.tensor(graph.dst, dtype=torch.long)
        # Per-edge sign, expanded from each presynaptic neuron's transmitter. The
        # node-level signs are kept too, because the dashboard colours cells by
        # transmitter and an edge list cannot answer "is this neuron inhibitory".
        sign = torch.tensor(graph.edge_sign(), dtype=torch.float32)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("sign", sign)
        self.register_buffer("node_sign", torch.tensor(
            graph.sign if graph.node_signed else np.zeros(graph.n, dtype=np.float32),
            dtype=torch.float32))
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
        # Value is an RL bookkeeping head, not a fly structure, so it reads freely.
        self.value = nn.Linear(int(dn.numel()), 1)

        # Modality routing is frozen structure, like the topology: applied to the
        # weights on every forward pass so gradients never populate a blocked pair.
        self.register_buffer("enc_mask", encoder_mask(graph, n_obs, self.n_retina))
        self.register_buffer("dec_mask", decoder_mask(graph, n_actions))
        self._init_masked()

    def _init_masked(self):
        """Start at unit gain on the routed pairs.

        Default `nn.Linear` init assumes a dense fan-in of `n_obs`, but a routed
        sensory unit reads one to four channels, so those weights start far too
        small to drive the settle. Unit weights also make the untrained network
        readable: a unit's current is the sum of its own channels, and each action
        logit is its own descending rate. Symmetry between mirrored L/R units is
        broken with a little noise so they can specialise.
        """
        with torch.no_grad():
            # Sensory polarity. Most receptors are driven by their channel, but lamina
            # monopolars are driven by its absence: their photoreceptor input is
            # histaminergic, so light inhibits them and a contrast decrement is what
            # depolarises them. Those rows start at weight -1 with a bias of +1, so the
            # rectified current is `relu(1 - luminance)` -- darkness -- which is what a
            # real L2 cell reports. Getting this backwards would make the fly's escape
            # pathway respond to bright sky instead of to an approaching dark object.
            pol = torch.ones(self.enc_mask.shape[0])
            for row, node in enumerate(self.sensory_idx.tolist()):
                pol[row] = float(self.graph.polarity.get(self.graph.names[node], 1))
            jitter = 1.0 + 0.05 * torch.randn_like(self.encoder.weight)
            self.encoder.weight.copy_(self.enc_mask * jitter * pol.unsqueeze(1))
            self.encoder.bias.copy_(torch.clamp(-pol, min=0.0))
            self.decoder.weight.copy_(self.dec_mask)
            self.decoder.bias.zero_()

            # Edge gains start at the measured synapse counts where we have them, so
            # the untrained network is the connectome at its own relative strengths
            # rather than a graph with every connection equally important. LC4's 2,580
            # synapses onto DNp01 should not start out matching a 5-synapse edge.
            measured = self.graph.initial_log_gain()
            if measured is not None:
                self.log_gain.copy_(torch.from_numpy(measured))

            # Units within a population can share an input set exactly (the two
            # forward-pool DNs, the two aDNs). At identical gain they would hold
            # identical rates and receive identical gradients forever, so the
            # population could never be more than one unit wearing three labels.
            self.log_gain.add_(0.05 * torch.randn_like(self.log_gain))

    def calibrate(self, obs: torch.Tensor) -> dict:
        """One-shot homeostatic scale so quiet cells can fire without the loud ones exploding.

        Each neuron's incoming synapses already sum to one (see `initial_log_gain`).
        About a third of them are inhibitory, so net drive shrinks with every hop, and
        a photoreceptor is three to five synapses from a descending neuron. Fitting a
        *per-neuron* sigma to that tiny residual is what the previous calibrator did,
        and it ran away: most cells have near-zero drive, dividing by it invented
        gains of 10^4, and the next settle peaked at 10^8.

        This instead does what synaptic scaling actually is -- one excitability for
        the whole network, plus a bounded boost for cells that have *some* drive but
        sit under threshold:

        * `sigma` is `clamp(mean |drive|, 0.2, 1.0)` so a quiet cell is at most 5×
          more excitable, never 10,000×.
        * cells with no drive at all stay at sigma=1 and stay silent -- nothing we
          can sense reaches them, and inventing a current would be making them up.

        One pass, no iteration. The terms stay learnable afterwards.
        """
        batch = obs.unsqueeze(0) if obs.dim() == 1 else obs
        with torch.no_grad():
            u = self.encode_u(batch)
            w = self.sign * torch.exp(self.log_gain)
            h = torch.zeros_like(u)
            drive = u
            for _ in range(self.steps):
                drive = sparse_apply(h, self.src, self.dst, w, self.n) + u
                pre = self.gamma * (drive - self.mu) / torch.exp(
                    self.log_sigma).clamp(min=1e-3) + self.beta
                h = 0.5 * h + 0.5 * F.relu(pre)
            mag = drive.abs().mean(dim=0)
            silent = drive.abs().amax(dim=0) < 1e-9
            sigma = torch.where(silent, torch.ones_like(mag), mag.clamp(0.2, 1.0))
            self.log_sigma.copy_(torch.log(sigma))
            h = self.forward_hidden(batch)
            dn = h.index_select(-1, self.dn_idx)
        return {
            "silent_neurons": int(silent.sum()),
            "sigma_median": float(sigma.median()),
            "motor_peak": float(dn.max()),
            "motor_active": int((dn > 1e-6).sum()),
            "rate_peak": float(h.max()),
        }

    def encode_u(self, obs: torch.Tensor) -> torch.Tensor:
        weight = self.encoder.weight * self.enc_mask
        currents = F.relu(F.linear(obs, weight, self.encoder.bias))
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

    def decode(self, dn: torch.Tensor) -> torch.Tensor:
        """Each action reads only its own descending / motor unit (plus a bias)."""
        return F.linear(dn, self.decoder.weight * self.dec_mask, self.decoder.bias)

    def forward(self, obs: torch.Tensor):
        h = self.forward_hidden(obs)
        dn = h.index_select(-1, self.dn_idx)
        logits = self.decode(dn)
        value = self.value(dn).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, greedy: bool = False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, logits

    # How many neurons of one group the raster tracks individually. A raster is pixels:
    # the pane is a few hundred rows tall, so sending all 159,810 central cells would be
    # a megabyte a tick to draw a line thinner than a pixel. Groups bigger than this are
    # sampled evenly and reported in full as an aggregate instead.
    TRACK_PER_GROUP = 48

    def tracked(self) -> torch.Tensor:
        """Neuron indices the dashboard follows one by one.

        Every descending and motor unit, because those are the output and each one is
        named and meaningful. Then an even sample of each other group, taken on a fixed
        stride so the same cells are followed for the whole run and the raster does not
        shimmer.
        """
        if getattr(self, "_tracked", None) is not None:
            return self._tracked
        keep = set(int(i) for i in self.dn_idx.tolist())
        by_group: dict[str, list[int]] = {}
        for i, group in enumerate(self.graph.groups):
            by_group.setdefault(group, []).append(i)
        for group, members in by_group.items():
            if group.startswith("motor:"):
                keep.update(members)
                continue
            stride = max(1, len(members) // self.TRACK_PER_GROUP)
            keep.update(members[::stride][: self.TRACK_PER_GROUP])
        self._tracked = torch.tensor(sorted(keep), dtype=torch.long)
        return self._tracked

    def group_index(self) -> dict[str, torch.Tensor]:
        if getattr(self, "_group_index", None) is None:
            by_group: dict[str, list[int]] = {}
            for i, group in enumerate(self.graph.groups):
                by_group.setdefault(group, []).append(i)
            self._group_index = {g: torch.tensor(v, dtype=torch.long)
                                 for g, v in by_group.items()}
        return self._group_index

    def topology(self) -> dict:
        """The parts of `inspect` that never change. Sent once, not every tick."""
        idx = self.tracked().tolist()
        return {
            "n": int(self.n),
            "n_edges": int(self.graph.n_edges),
            "dataset": self.graph.provenance.get("dataset", ""),
            "tracked": idx,
            "names": [self.graph.names[i] for i in idx],
            "roles": [self.graph.roles[i] for i in idx],
            "groups": [self.graph.groups[i] for i in idx],
            "bodies": [self.graph.bodies[i] for i in idx],
            "sign": [float(self.node_sign[i]) for i in idx],
            "group_sizes": {g: int(v.numel()) for g, v in self.group_index().items()},
            "action_units": {a: list(u) for a, u in self.graph.action_units.items()},
            "descending": {p: list(u) for p, u in self._descending().items()},
            "action_populations": dict(ACTION_POPULATIONS),
            "descending_what": {p: list(w) for p, w in DESCENDING_WHAT.items()},
            "unread": list(self.graph.unread),
        }

    def _report(self, u: torch.Tensor, h: torch.Tensor, logits: torch.Tensor,
                value: torch.Tensor, idx: int) -> dict:
        """Neuroscope payload from an already-computed settle.

        `h` covers the tracked neurons only, in `topology()["tracked"]` order, and
        `groups` carries mean / peak / active-fraction over *every* cell of each group
        so nothing is hidden by the sampling -- if a population the raster only samples
        lights up, the group summary still says so.
        """
        rates = h[0] if h.dim() == 2 else h
        currents = u[0] if u.dim() == 2 else u
        tracked = self.tracked()
        groups = {}
        for group, members in self.group_index().items():
            g = rates.index_select(0, members)
            groups[group] = {
                "mean": float(g.mean()),
                "peak": float(g.max()),
                "active": float((g > 1e-6).float().mean()),
            }
        return {
            "action": ACTIONS[idx],
            "index": idx,
            "value": float(value.reshape(-1)[0].item()),
            "logits": [float(x) for x in logits.reshape(-1)[:self.n_actions].tolist()],
            "h": [float(x) for x in rates.index_select(0, tracked).tolist()],
            "u": [float(x) for x in currents.index_select(0, tracked).tolist()],
            "group_stats": groups,
            "peak": float(rates.max()),
            "active": int((rates > 1e-6).sum()),
            "n": int(self.n),
            "n_edges": int(self.graph.n_edges),
        }

    def inspect(self, obs: torch.Tensor, greedy: bool = True) -> dict:
        """Rates and chosen action for the neuroscope."""
        batch = obs.unsqueeze(0) if obs.dim() == 1 else obs
        with torch.no_grad():
            u = self.encode_u(batch)
            h = self.forward_hidden(batch)
            dn = h.index_select(-1, self.dn_idx)
            logits = self.decode(dn)
            value = self.value(dn).squeeze(-1)
            idx = int(logits[0].argmax().item()) if greedy else int(
                torch.distributions.Categorical(logits=logits[0]).sample().item()
            )
        return self._report(u, h, logits[0], value, idx)

    def act_and_inspect(self, obs: torch.Tensor, greedy: bool = False):
        """One settle, used for both the action and the dashboard.

        Calling `act` and then `inspect` runs the network twice on the same input. On
        the full connectome a settle is most of the control period, so doing it twice
        halves the achievable tick rate to show a picture of a decision that was
        already made -- and the picture would be of a *second*, identical settle, not
        of the one that acted. Returns `(action, logp, value, report)`.
        """
        batch = obs.unsqueeze(0) if obs.dim() == 1 else obs
        with torch.no_grad():
            u = self.encode_u(batch)
            h = self.forward_hidden(batch)
            dn = h.index_select(-1, self.dn_idx)
            logits = self.decode(dn)
            value = self.value(dn).squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if greedy else dist.sample()
            logp = dist.log_prob(action)
        idx = int(action.reshape(-1)[0].item())
        report = self._report(u, h, logits[0], value, idx)
        if obs.dim() == 1:
            return action.reshape(())[()], logp.reshape(())[()], value.reshape(())[()], report
        return action, logp, value, report

    def _descending(self) -> dict[str, list[str]]:
        """Descending populations, grouped as the graph itself groups them.

        The graph labels every motor node `motor:<population>`, so the grouping is read
        back off the graph rather than kept in a second hand-written table that could
        disagree with it.
        """
        out: dict[str, list[str]] = {}
        for i in self.dn_idx.tolist():
            group = self.graph.groups[i]
            if group.startswith("motor:"):
                out.setdefault(group.split(":", 1)[1], []).append(self.graph.names[i])
        return out

    def edge_flow(self, h: torch.Tensor) -> list[float]:
        """Signed current carried by each edge, so the viz can show signals moving."""
        with torch.no_grad():
            w = self.sign * torch.exp(self.log_gain)
            rates = h[0] if h.dim() == 2 else h
            return [float(x) for x in (rates.index_select(0, self.src) * w).tolist()]

    def frames_to_obs(self, frames) -> torch.Tensor:
        if isinstance(frames, dict):
            frames = [frames]
        vecs = [frame_to_vector(f, self.n_retina) for f in frames]
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
            "n_retina": self.n_retina,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load(cls, graph: FlyGraph, path: str | Path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(graph, ckpt["n_obs"], ckpt["n_actions"], ckpt.get("steps", 5),
                    ckpt.get("n_retina"))
        model.load_state_dict(ckpt["state_dict"])
        return model

    @classmethod
    def load_compatible(cls, graph: FlyGraph, path: str | Path, map_location="cpu"):
        """Load what still fits after a connectome change, and report what did not.

        Changing the graph changes the shape of everything indexed by neuron or
        edge, so a checkpoint from an older connectome cannot be loaded whole.
        Rather than refuse to train, keep the tensors that still match and leave
        the rest at init. Returns `(model, kept, dropped)`.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(graph, ckpt["n_obs"], ckpt["n_actions"], ckpt.get("steps", 5),
                    ckpt.get("n_retina"))
        current = model.state_dict()
        kept, dropped = [], []
        for key, tensor in ckpt["state_dict"].items():
            if key in current and current[key].shape == tensor.shape:
                current[key] = tensor
                kept.append(key)
            else:
                dropped.append(key)
        model.load_state_dict(current)
        return model, kept, dropped


# Every graph here is derived from male-cns:v1.0. `full` is the published connectome
# untouched -- 176,422 neurons, 6,287,749 measured connections -- and is what we prefer.
# The smaller ones are subgraphs of the same data, for when a run has to be quick.
#
# There is deliberately no hand-built fallback. There used to be one, and roughly half
# of its 517 edges do not exist in male-cns: `DNa02_L ⊣ DNa02_R`, `DNp09 ⊣ MDN`,
# `DNg60 ⊣ DNp09`, `LC11 → DNa02` are all zero-synapse, and it violated Dale's law by
# putting sign on the edge instead of the neuron. A network trained on it was learning a
# circuit the fly does not have, so it is better for this to fail loudly and say how to
# build the real graph.
GRAPH_PREFERENCE = ("malecns_full.npz", "malecns_wide.npz", "malecns_path.npz",
                    "malecns_type.npz")

BUILD_HINT = (
    "Build it with `uv run python python/tools/build_graph_from_flyb.py "
    "--variant {variant}`. That needs malecns-v1.0.flyb.gz, which is not committed "
    "(23 MB, and it belongs to its publishers) -- see python/connectome/flyb.py for "
    "where it is looked for and how to rebuild it from neuPrint."
)


def default_graph_path(root: Path, variant: str | None = None) -> Path:
    """Best available graph, or a named variant.

    A named variant is returned whether or not it exists, so a config asking for one
    fails with a missing-file error naming what it wanted rather than quietly training
    on a different connectome than the one the config chose.
    """
    folder = root / "data" / "connectome"
    if variant:
        name = variant if variant.endswith(".npz") else f"malecns_{variant}.npz"
        return folder / name
    for candidate in GRAPH_PREFERENCE:
        if (folder / candidate).exists():
            return folder / candidate
    return folder / GRAPH_PREFERENCE[0]


def load_graph(root: Path, variant: str | None = None) -> FlyGraph:
    """Load a graph derived from the real connectome, or say how to build one."""
    path = default_graph_path(root, variant)
    if not path.exists():
        wanted = path.stem.replace("malecns_", "")
        raise FileNotFoundError(
            f"{path} not found. " + BUILD_HINT.format(variant=wanted)
        )
    graph = FlyGraph(path)
    if not graph.derived:
        raise ValueError(
            f"{path} is not a graph derived from male-cns: it carries no synapse "
            "counts or routing. " + BUILD_HINT.format(variant="full")
        )
    return graph


def build_policy(root: Path, variant: str | None = None) -> FlyPolicy:
    graph = load_graph(root, variant)
    return FlyPolicy(graph, vector_size(graph.n_retina or 64), len(ACTIONS))
