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

from dashboard.hub import publish, publish_stream, publish_topology
from load_env import load_repo_env, wandb_api_key
from fly_policy.export import export_safetensors
from fly_policy.mc_env import MinecraftEnv
from fly_policy.policy import ACTIONS, FlyPolicy, load_graph
from sense.frame import vector_size
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


def _emit(epoch: int, episode: int, step: int, action: str, value, pkt: dict, brain: dict | None = None):
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
        "brain": brain,
    })


def _log_step(fh, line: str):
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def rollout_mc(
    env: MinecraftEnv,
    model: FlyPolicy,
    max_steps: int,
    epoch: int = 0,
    episode: int = 0,
    dt_s: float = 1.0,
    log_fh=None,
):
    import time
    _emit(epoch, episode, 0, "resetting", None, {})
    pkt = env.reset()
    o0 = model.frames_to_obs(pkt.get("frame") or {})
    with torch.no_grad():
        boot = model.inspect(o0, greedy=True)
    facts0 = pkt.get("facts") or {}
    _log_step(
        log_fh,
        f"reset epoch={epoch} ep={episode} rarest={facts0.get('rarest')} dist={facts0.get('distance')}",
    )
    _emit(epoch, episode, 0, "reset", None, pkt, boot)
    obs, acts, logps, vals, rewards = [], [], [], [], []
    last = pkt
    done = False
    while not done and len(obs) < max_steps:
        t0 = time.monotonic()
        o = model.frames_to_obs(pkt.get("frame") or {})
        # One settle for both the decision and the picture of it, so the dashboard
        # shows the rates that actually chose this action.
        action, logp, value, brain = model.act_and_inspect(o, greedy=False)
        name = ACTIONS[int(action.item())]
        pkt = env.step(name)
        obs.append(o)
        acts.append(action)
        logps.append(logp)
        vals.append(value)
        rewards.append(float(pkt.get("reward") or 0.0))
        done = bool(pkt.get("done"))
        last = pkt
        _emit(epoch, episode, len(obs), name, float(value.item()), pkt, brain)
        if len(obs) == 1 or len(obs) % 10 == 0 or done:
            facts = pkt.get("facts") or {}
            _log_step(
                log_fh,
                f"step {epoch}.{episode}.{len(obs)} {name} "
                f"rew={rewards[-1]:.3f} dist={facts.get('distance')} "
                f"rarest={facts.get('rarest')} inv={facts.get('inventory')}",
            )
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
                    env, model, max_steps, epoch=ep, episode=i, dt_s=dt_s, log_fh=fh,
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
                minibatch=train.get("minibatch"),
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
    goal = cfg.get("goal") or {}
    catalog = load_goal_catalog(goal["catalog"]) if goal.get("catalog") else {}
    mc = cfg["minecraft"]
    banner = {
        "run": cfg["run"]["name"],
        "ckpt": cfg["run"]["ckpt"],
        "mode": cfg.get("mode") or "rarest",
        "minecraft": f"{mc['host']}:{mc['port']}",
    }
    if catalog:
        banner["selector"] = goal.get("selector")
        banner["catalog_logs"] = list(catalog)
        banner["example_rarest"] = spec_for_log(
            catalog, rarest_log({"oak_log": 9, "cherry_log": 1})
        )["id"]
    print(json.dumps(banner, indent=2), flush=True)

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

    graph = load_graph(ROOT, cfg.get("graph", {}).get("variant"))
    prov = graph.provenance
    whole = prov.get("whole_connectome")
    print(
        f"graph: {prov['dataset']} "
        f"{'whole connectome' if whole else 'derived subgraph'}, {graph.n:,} neurons / "
        f"{graph.n_edges:,} connections carrying {graph.weight.sum():,.0f} synapses, "
        f"{graph.n_retina} ommatidia",
        flush=True,
    )
    ckpt = Path(cfg["run"]["ckpt"])
    model = None
    if ckpt.exists():
        model, kept, dropped = FlyPolicy.load_compatible(graph, ckpt)
        if dropped:
            print(
                f"loaded {ckpt} partially: kept {len(kept)} tensors, reinitialised "
                f"{len(dropped)} that the current connectome changed shape on "
                f"({', '.join(dropped[:6])}{'…' if len(dropped) > 6 else ''})",
                flush=True,
            )
        else:
            print(f"loaded {ckpt}", flush=True)
    elif cfg["run"].get("allow_fresh"):
        # Changing the connectome changes the shape of every neuron- and edge-indexed
        # tensor, so there is nothing to carry over from a run on a different graph.
        # Starting from the wiring is the honest initial condition: measured synapse
        # counts as gains, and each receptor reading only its own modality.
        model = FlyPolicy(graph, vector_size(graph.n_retina), len(ACTIONS))
        print(f"no checkpoint at {ckpt}; starting from the connectome itself", flush=True)
    else:
        raise FileNotFoundError(
            f"checkpoint missing: {ckpt}. Set `run.allow_fresh: true` to start from "
            "the untrained connectome instead."
        )

    start_dashboard()
    publish_topology(model.topology())
    dummy = torch.zeros(4, model.n_obs)
    dummy[:, : model.n_retina] = 0.55
    cal = model.calibrate(dummy)
    print(
        f"calibrated: motor_peak={cal['motor_peak']:.3f} "
        f"motor_active={cal['motor_active']} rate_peak={cal['rate_peak']:.2f} "
        f"silent={cal['silent_neurons']:,}",
        flush=True,
    )
    import time as _time
    _time.sleep(0.6)
    env = MinecraftEnv(cfg, on_stream=publish_stream)
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
