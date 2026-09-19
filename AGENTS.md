# AGENTS.md — fruit-fly Mineflayer connectome

Read this before changing sense, goals, or the policy. The plan file is history; this file is current repo truth.

## What this repo is

Two stacked agents:

1. **Legacy lumberjack** — [`bot.js`](bot.js) + [`brain.js`](brain.js). Chat `go`. Pathfinder checklist for nine woods. Expert teacher only.
2. **Fly policy** — Minecraft (or a synthetic stand-in) → **real fly sensory populations** → frozen connectome + learnable encoder/gains/decoder → discrete Mineflayer actions.

Goals are **not** a wood enum into an MLP. A `GoalSpec` adds drive on the same `SensoryFrame` the world already fills.

**Rarest log** is an outer loop only: census explored `*_log` blocks → `rarest_log()` → pick a spec from [`configs/goals/log_collect.yaml`](configs/goals/log_collect.yaml) → `applyGoal`. Do not teach GoalToSense about rarity.

The fly **must** run in Minecraft (`fly.js`). Training YAML: [`configs/train/rarest_minecraft.yaml`](configs/train/rarest_minecraft.yaml).

## Do not

- Add `requirements.txt` / `pip install`. Python is **`uv`** (`pyproject.toml`, `.venv`).
- Hardcode “wood odor” as the world model. Woods are rows in generic odor tables.
- Vendor the full male-cns / FlyWire graph. Mini graph only; see `python/tools/fetch_connectome.py`.
- Overwrite `checkpoints/fly_mc.pt` with PPO. PPO writes `checkpoints/fly_mc_ppo.pt`.
- Edit the Cursor plan file.

## Commands

```bash
./start-server.sh            # Paper must be up — the fly does not run synthetic
node fly.js                  # FruitFly: explore + interact. Chat: stop | explore | log
uv run python python/train.py configs/train/rarest_minecraft.yaml
uv sync --group dev
npm test
uv run pytest
```

## Layout

| Path | Role |
|------|------|
| `configs/sense/*` | Retina, odor (block/item/entity), taste, mechano, neuPrint `population_index` |
| `configs/goals/*.json` | GoalSpecs: injections + `success` (eval/RL only) |
| `js/sense/` | SenseBridge, GoalToSense, JSONL logger, action apply, policy TCP client, `flyLoop` |
| `python/sense/` | Same GoalToSense + `frame_to_vector` |
| `python/fly_policy/` | Mini graph, ChessFly settle, `FlyPolicy`, synthetic oak taxis env |
| `python/train_il.py` / `train_rl.py` / `eval_harness.py` / `infer_server.py` | Train / eval / deploy |
| `docs/SENSE_PROVENANCE.md` | Why each mapping exists (cloned refs in `.research/`, gitignored) |
| `logs/learning_loop_001/` | First logged loop (read `SUMMARY.md`) |

## Sensory contract

Copied from fly-brain-minecraft `SensoryFrame` / `WorldSenses` (clone: `.research/fly-brain-minecraft`):

- Vision: raycast luminance + analytic LC4 / LPLC2 / LC11 / LC10a / HS
- Olfaction: class → glomerulus affinities, `exp(-d/6)`, L/R via `odorBearingDeg` (**+ = fly’s right**)
- Gustation: contact → GRNs (`taste_table.json`)
- JO / bristles / rain / damage / thermo-hygro

`js/sense/senseBridge.js` `sampleWorld(world, cfg)` is the testable core. `sampleBot(bot, cfg)` fills `world` from Mineflayer.

## Goal contract

```json
{ "id": "collect:oak_log", "injections": [{ "modality": "olfaction", "pattern": {...}, "strength": 0.15, "falloff": "tonic" }], "success": { "type": "inventory_contains", "item": "oak_log" } }
```

`falloff`: `tonic` | `when_seen` | `when_near` | `distance`. Compiler: `js/sense/goalToSense.js` and `python/sense/goal_to_sense.py` (keep them in sync).

## Policy contract

ChessFly dynamics (`huggingface.co/mlabonne/chessfly`):

`h ← (1-a)h + a relu(γ (W h + u − μ)/σ + β)`, `W = sign * exp(θ)`.

- Frozen: topology + signs (`data/connectome/mini_male_cns.npz`, 63 neurons, 233 edges, real ORN/GRN/LC/DN **names**)
- Learned: encoder (obs → sensory currents), `log_gain`, homeostatic γ/μ/σ/β, decoder + value on DN units
- Actions: `forward back turn_left turn_right jump mine camera_up camera_down noop`

Obs vector: 64 retina + glomeruli + bearing sin/cos + GRNs + object channels + mechano (`python/sense/frame.py`).

TCP: one JSON object per line. `{ "type": "act", "frame": {...}, "greedy": true }` → `{ "action": "forward", ... }`.

## Learning loop 001 (already ran)

Logged in [`logs/learning_loop_001/SUMMARY.md`](logs/learning_loop_001/SUMMARY.md).

On synthetic oak taxis (20 episodes): **random 0% · expert 100% (16 steps) · IL 5% · PPO 45%**.

Env heading is `bearing_right = -heading_error` so `turn_right` decreases yaw. Do not “fix” this without re-running the expert 20/20 test (`python/tests/test_synth_env.py`).

IL uses **class-weighted** CE so `mine` is not dropped. PPO returns include +10 on collect.

Minecraft-in-the-loop IL from `brain.js` is wired (`fly.logExpert` when logging is on) but **not yet run** — next session if the Paper server is up: `log` then `go`, then `uv run python python/train_il.py --demos logs/sense_*.jsonl`.

## Next loops (suggested)

1. Live Paper + `infer_server` + chat `fly` with `collect_oak`.
2. Record lumberjack JSONL and IL on real SenseBridge frames.
3. Swap GoalSpecs (`feed`, `flee_creeper`, `collect_spruce`) without changing topology.
4. Only then consider a larger connectome (neuPrint pull). Keep sensory + DN identities.

## Research clones (gitignored)

`.research/fly-brain-minecraft` and `.research/flybody` were cloned for implementation. ChessFly only publishes `flynet.safetensors` + `connectome_meta.json` (graph not redistributed). mujoco-py is deprecated; flybody uses official MuJoCo — not on the Minecraft path.

## Tests

- JS: official **`node:test`** (`npm test`)
- Python: **`uv run pytest`** (`pythonpath = python`)

A learning-loop regression is “PPO eval success > random on oak taxis”, not “IL always 100% episodes”.
