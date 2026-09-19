# AGENTS.md — fruit-fly Mineflayer connectome

Read this before changing sense, goals, or the policy. The plan file is history; this file is current repo truth.

## What this repo is

Two stacked agents:

1. **Legacy lumberjack** — [`bot.js`](bot.js) + [`brain.js`](brain.js). Chat `go`. Pathfinder checklist for nine woods. Expert teacher only.
2. **Fly policy** — Minecraft → **real male-cns cells** → frozen connectome + learnable encoder/gains/decoder → discrete Mineflayer actions.

The running graph is **male-cns:v1.0**, 176,422 traced neurons and 6,287,749 measured connections (`data/connectome/malecns_full.npz`). It is derived, not drawn: selectors in [`python/connectome/anchors.py`](python/connectome/anchors.py) name the real sensory and motor types, and [`python/tools/build_graph_from_flyb.py`](python/tools/build_graph_from_flyb.py) writes the NPZ from `malecns-v1.0.flyb.gz`. There is no hand-built mini graph.

Goals are **not** a wood enum into an MLP. A `GoalSpec` adds drive on the same `SensoryFrame` the world already fills.

**Navigate first.** [`configs/train/navigate_minecraft.yaml`](configs/train/navigate_minecraft.yaml) has no GoalSpec: the reward is covering new ground without colliding (`js/sense/nav_reward.js`). Rarest-log census is an outer loop for a later recipe only.

The fly **must** run in Minecraft (`fly.js`, `python/play.py`, or `python/train.py`). Paper is local (`./start-server.sh`, 127.0.0.1:25565).

Secrets live in **`.env`** (`WANDB_API_KEY=`). Load via `python/load_env.py` / `js/sense/loadEnv.js`. Never send the key to the dashboard HTML.

## Do not

- Add `requirements.txt` / `pip install`. Python is **`uv`** (`pyproject.toml`, `.venv`).
- Hardcode “wood odor” as the world model. Woods are rows in generic odor tables.
- Vendor `malecns-v1.0.flyb.gz` (23 MB, belongs to its publishers). The derived NPZ is local.
- Invent neuron types. If a cell is not in male-cns, it is not in the graph. Abdominal touch is a declared gap (`unknown_sensory`), not an `SNta_abdomen`.
- Overwrite `checkpoints/fly_mc.pt` with PPO. Navigate writes `fly_mc_navigate.pt`. Oak PPO is `fly_mc_ppo.pt`. Rarest PPO is `fly_mc_rarest.pt`.
- Commit `.env` or paste `WANDB_API_KEY` into the browser client.
- Edit the Cursor plan file.

## Commands

```bash
./start-server.sh            # Paper 1.21.11 on 127.0.0.1:25565 (not the public internet)
uv run python python/tools/build_graph_from_flyb.py --variant full
uv run python python/play.py --fresh          # open world, dashboard :8766
uv run python python/train.py configs/train/navigate_minecraft.yaml
# dashboard: http://127.0.0.1:8766/  (orbit world · fly retina · neuron→body)
uv sync --group dev
npm test
uv run pytest
```

## Layout

| Path | Role |
|------|------|
| `configs/sense/*` | Retina, odor, taste, mechano, neuPrint `population_index` (real type strings) |
| `configs/sense/retina_columns.json` | 128 measured ommatidial directions (generated) |
| `data/connectome/malecns_full.npz` | Whole male-cns:v1.0, one node per neuron |
| `python/connectome/` | FLYB reader, anchors, subgraph derive |
| `js/sense/` | SenseBridge, GoalToSense, voxel stream, nav reward, `mc_rollout.js` |
| `python/sense/` | Same GoalToSense + `frame_to_vector` + channel resolver |
| `python/fly_policy/` | `FlyGraph`, ChessFly settle, `FlyPolicy` |
| `python/train.py` | YAML trainer. Live Minecraft PPO when `minecraft.required` |
| `python/play.py` | No-goal livestream of a (possibly fresh) policy |
| `python/dashboard/` | FastAPI live view on `:8766` |

## Sensory contract

Copied from fly-brain-minecraft `SensoryFrame` / `WorldSenses` (clone: `.research/fly-brain-minecraft`):

- Vision: **measured ommatidial rays** (128, farthest-point sampled from `column_directions.csv`) + analytic LC4 / LPLC2 / LC11 / LC10a / HS. Lamina is driven by darkness (histaminergic R1–R6).
- Olfaction: class → glomerulus affinities, `exp(-d/6)`, L/R via `odorBearingDeg` (**+ = fly’s right**). VP glomeruli are thermo/hygro, not ORNs.
- Gustation: contact → GRNs (`taste_table.json`). Valence is downstream: bitter GRNs are cholinergic.
- JO / bristles / hair plates / campaniform / rain / damage / thermo-hygro

`js/sense/senseBridge.js` `sampleWorld(world, cfg)` is the testable core. `sampleBot(bot, cfg)` fills `world` from Mineflayer.

## Goal contract

```json
{ "id": "collect:oak_log", "injections": [{ "modality": "olfaction", "pattern": {...}, "strength": 0.15, "falloff": "tonic" }], "success": { "type": "inventory_contains", "item": "oak_log" } }
```

`falloff`: `tonic` | `when_seen` | `when_near` | `distance`. Compiler: `js/sense/goalToSense.js` and `python/sense/goal_to_sense.py` (keep them in sync).

## Policy contract

ChessFly dynamics (`huggingface.co/mlabonne/chessfly`):

`h ← (1-a)h + a relu(γ (W h + u − μ)/σ + β)`, `W = sign * exp(θ)`.

- Frozen: topology + Dale signs + synapse-count gains (`malecns_full.npz`)
- Learned: encoder (obs → sensory currents, **masked by modality**), `log_gain`, homeostatic γ/μ/σ/β, decoder + value on DN / motor units
- Actions: `forward back turn_left turn_right jump mine camera_up camera_down noop`
  - `noop` is DNg60 (GABAergic halt), not an absence
  - `turn_left` / `turn_right` both read DNa02; side is the signal
  - `camera_up` ← MNnm* pool; `camera_down` ← ADNM/FNM; the decoder learns the sign

Obs vector: 128 ommatidia + glomeruli + bearing sin/cos + GRNs + object channels + mechano (`python/sense/frame.py`). Routing keys live on the graph (`retina:37`, `glomerulus:DM1`) and are resolved by `python/sense/channels.py`.

TCP: one JSON object per line. `{ "type": "act", "frame": {...}, "greedy": true }` → `{ "action": "forward", ... }`.

The full graph settles in ~260 ms on CPU (~3.6 Hz). `control_dt_ms: 280`. PPO minibatches at 1 because the backward pass keeps every intermediate rate on 6.29M edges.

## Live dashboard

`train.py` / `play.py` start FastAPI on **127.0.0.1:8766**. Three collapsible panes:

1. Minecraft world — WebGL2 voxels, orbit / first-person / follow. Pose at 30 Hz.
2. What the fly sees — 128 measured ommatidia + glomeruli / GRNs / mechano.
3. Neurons → body — descending populations (real male-cns cells), per-group activity over **every** cell, raster of a per-population sample. Topology is sent once; ticks carry rates only.

The RGB/orbit view is **for us**. The policy still only sees luminance + the other mapped senses.

## Tests

- JS: official **`node:test`** (`npm test`)
- Python: **`uv run pytest`** (`pythonpath = python`)

A learning-loop regression is “PPO eval success > random on oak taxis”, not “IL always 100% episodes”.
