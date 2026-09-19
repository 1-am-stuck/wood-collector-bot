#!/usr/bin/env python3
"""Train from a YAML recipe. Requires a live Minecraft server when minecraft.required is true."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from dashboard.hub import publish
from load_env import load_repo_env, wandb_api_key
from fly_policy.export import export_safetensors
from fly_policy.graph import FlyGraph
from fly_policy.mc_env import MinecraftEnv
from fly_policy.policy import ACTIONS, FlyPolicy, default_graph_path
from tools.build_mini_graph import build
from train_config import load_goal_catalog, load_train_yaml, spec_for_log
from train_rl import advantages, ppo_epoch_update
from sense.rarity import rarest_log


def start_dashboard(host: str = "127.0.0.1", port: int = 8766):
    import threading

    def _run():
        import uvicorn
        from dashboard.app import app
        uvicorn.run(app, host=host, port=port, log_level="warning")

    threading.Thread(target=_run, daemon=True, name="fly-dashboard").start()
    print(f"dashboard http://{host}:{port}/", flush=True)


def minecraft_up(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _emit(epoch: int, episode: int, step: int, action: str, value, pkt: dict):
    import time
    publish({
        "t": time.time(),
        "epoch": epoch,
        "episode": episode,
        "step": step,
        "action": action,
        "reward": float(pkt.get("reward") or 0.0),
        "done": bool(pkt.get("done")),
        "value": float(value) if value is not None else None,
        "frame": pkt.get("frame") or {},
        "facts": pkt.get("facts") or {},
        "eye": pkt.get("eye"),
    })


def rollout_mc(
    env: MinecraftEnv,
    model: FlyPolicy,
    max_steps: int,
    epoch: int = 0,
    episode: int = 0,
    dt_s: float = 1.0,
):
    import time
    pkt = env.reset()
    _emit(epoch, episode, 0, "reset", None, pkt)
    obs, acts, logps, vals, rewards = [], [], [], [], []
    last = pkt
    done = False
    while not done and len(obs) < max_steps:
        t0 = time.monotonic()
        o = model.frames_to_obs(pkt.get("frame") or {})
        with torch.no_grad():
            action, logp, value, _ = model.act(o, greedy=False)
        name = ACTIONS[int(action.item())]
        pkt = env.step(name)
        obs.append(o)
        acts.append(action)
        logps.append(logp)
        vals.append(value)
        rewards.append(float(pkt.get("reward") or 0.0))
        done = bool(pkt.get("done"))
        last = pkt
        _emit(epoch, episode, len(obs), name, float(value.item()), pkt)
        leftover = dt_s - (time.monotonic() - t0)
        if leftover > 0:
            time.sleep(leftover)
    return obs, acts, logps, vals, rewards, last


def train_minecraft(model: FlyPolicy, env: MinecraftEnv, cfg: dict, log_path: Path, wb=None):
    train = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=float(train.get("lr") or 3e-4))
    epochs = int(train.get("epochs") or 12)
    episodes = int(train.get("episodes_per_epoch") or 4)
    max_steps = int(train.get("max_steps") or 200)
    gamma = float(train.get("gamma") or 0.98)
    clip = float(train.get("clip") or 0.2)
    last_ret = 0.0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        for ep in range(epochs):
            batch_obs, batch_act, batch_oldlp, batch_adv, batch_ret = [], [], [], [], []
            rets, collects, rarests = [], [], []
            for i in range(episodes):
                dt_s = float(train.get("control_dt_ms") or 1000) / 1000.0
                obs, acts, logps, vals, rewards, last = rollout_mc(
                    env, model, max_steps, epoch=ep, episode=i, dt_s=dt_s,
                )
                vs = [float(v.item()) for v in vals]
                adv, ret = advantages(rewards, vs, gamma=gamma)
                batch_obs.extend(obs)
                batch_act.extend(acts)
                batch_oldlp.extend(logps)
                batch_adv.extend(adv)
                batch_ret.extend(ret)
                rets.append(sum(rewards))
                facts = (last or {}).get("facts") or {}
                rarests.append(facts.get("rarest"))
                collects.append(1 if any(r >= 15 for r in rewards) else 0)
            last_ret = sum(rets) / max(1, len(rets))
            collect_rate = sum(collects) / max(1, len(collects))
            if not batch_obs:
                raise RuntimeError("empty PPO batch — Minecraft rollout produced no steps")
            loss = ppo_epoch_update(
                model,
                torch.stack(batch_obs),
                torch.stack(batch_act),
                torch.stack(batch_oldlp).detach(),
                batch_adv,
                batch_ret,
                opt,
                clip=clip,
            )
            line = (
                f"ppo epoch {ep} return={last_ret:.3f} loss={loss:.4f} "
                f"collect={collect_rate:.2f} rarest={rarests}"
            )
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()
            if wb is not None:
                wb.log({
                    "epoch": ep,
                    "return": last_ret,
                    "loss": loss,
                    "collect_rate": collect_rate,
                })
    return model, last_ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", default=str(ROOT / "configs/train/rarest_minecraft.yaml"))
    args = p.parse_args()
    cfg = load_train_yaml(args.config)
    catalog = load_goal_catalog(cfg["goal"]["catalog"])
    mc = cfg["minecraft"]
    print(json.dumps({
        "run": cfg["run"]["name"],
        "ckpt": cfg["run"]["ckpt"],
        "selector": cfg["goal"]["selector"],
        "catalog_logs": list(catalog),
        "example_rarest": spec_for_log(catalog, rarest_log({"oak_log": 9, "cherry_log": 1}))["id"],
        "minecraft": f"{mc['host']}:{mc['port']}",
    }, indent=2), flush=True)

    if mc.get("required", True) and not minecraft_up(mc["host"], mc["port"]):
        print(
            f"Minecraft is not running at {mc['host']}:{mc['port']}.\n"
            "Start Paper (./start-server.sh) then:\n"
            f"  uv run python python/train.py {args.config}",
            file=sys.stderr,
        )
        sys.exit(2)

    load_repo_env(ROOT)
    wb_cfg = cfg.get("wandb") or {}
    if wb_cfg.get("base_url"):
        os.environ["WANDB_BASE_URL"] = wb_cfg["base_url"]
    else:
        os.environ.pop("WANDB_BASE_URL", None)
    os.environ.setdefault("WANDB_DIR", str(ROOT / "wandb"))
    (ROOT / "wandb").mkdir(parents=True, exist_ok=True)
    wb = None
    key = wandb_api_key(ROOT)
    if not key:
        print("WANDB_API_KEY missing in .env — wandb disabled", flush=True)
    else:
        try:
            import wandb
            wandb.login(key=key, relogin=True)
            wb = wandb.init(
                project=wb_cfg.get("project") or "fly-mc",
                name=cfg["run"]["name"],
                config=cfg,
                mode=wb_cfg.get("mode") or "online",
                tags=wb_cfg.get("tags") or [],
            )
            print("wandb:", wb.url if hasattr(wb, "url") else "online", flush=True)
        except Exception as exc:
            print("wandb init failed, continuing without:", exc, flush=True)

    gpath = default_graph_path(ROOT)
    if not gpath.exists():
        build()
    graph = FlyGraph(gpath)
    ckpt = Path(cfg["run"]["ckpt"])
    if not ckpt.exists():
        raise FileNotFoundError(f"best checkpoint missing: {ckpt}")
    model = FlyPolicy.load(graph, ckpt)
    print(f"loaded {ckpt}", flush=True)

    start_dashboard()
    import time as _time
    _time.sleep(0.6)
    env = MinecraftEnv(cfg)
    log_path = ROOT / "logs" / "learning_loop_002" / "train.txt"
    try:
        hello = env.start()
        print(f"joined minecraft as {hello.get('username')}", flush=True)
        model, ret = train_minecraft(model, env, cfg, log_path, wb=wb)
    finally:
        env.close()

    out = Path(cfg["run"]["out"])
    model.save_checkpoint(out, extra={"ppo_return": ret, "world": "minecraft"})
    export_safetensors(model, out.with_suffix(".safetensors"))
    print(f"wrote {out} return={ret:.3f}", flush=True)
    if wb is not None:
        wb.log({"final_return": ret})
        wb.finish()


if __name__ == "__main__":
    main()
