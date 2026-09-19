# Learning loop 001 — 2026-09-19

Synthetic `collect:oak_log` GoalSpec (no Minecraft server this run).
Policy: ChessFly-style frozen mini male-cns + encoder/gains/decoder.

## Commands

```
uv sync --group dev
npm test
uv run pytest
uv run python python/train_il.py --epochs 40 --out checkpoints/fly_mc.pt
uv run python python/train_rl.py --ckpt checkpoints/fly_mc.pt --out checkpoints/fly_mc_ppo.pt
uv run python python/eval_harness.py --ckpt checkpoints/fly_mc.pt
uv run python python/eval_harness.py --ckpt checkpoints/fly_mc_ppo.pt
```

## Results (20 eval episodes)

| Policy | Success | Mean steps on success |
|--------|---------|------------------------|
| random | 0.00 | — |
| expert | 1.00 | 16 |
| IL `fly_mc.pt` | 0.05 | 42 |
| PPO `fly_mc_ppo.pt` | **0.45** | 40 |

IL train: 714 expert ticks, weighted CE, acc 0.842 after 40 epochs.
PPO: 6 epochs × 12 episodes, last return ≈ 16.5 (includes +10 collect bonus).

## Bugs found in this loop

1. First expert used inverted bearing vs `turn_left`/`turn_right` → 5% expert success. Fixed in `python/fly_policy/synth_env.py` (`bearing_right = -heading_error`).
2. PPO must not overwrite the IL checkpoint; use `fly_mc_ppo.pt`.
3. Unweighted IL never emitted `mine` (majority `forward`). Weighted CE fixed recall.

## Artifacts

- `logs/expert_synth.jsonl`
- `checkpoints/fly_mc.pt` + `fly_mc.safetensors`
- `checkpoints/fly_mc_ppo.pt` + `fly_mc_ppo.safetensors` (if exported)
- `logs/learning_loop_001/eval_il.json`
- `logs/learning_loop_001/eval_ppo.json`
- `logs/learning_loop_001/04_weighted_il.txt`
