#!/usr/bin/env python3
"""PPO fine-tune. Rewards come from GoalSpec.success, not a wood-hardcoded net."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from load_env import load_repo_env
from fly_policy.export import export_safetensors
from fly_policy.policy import ACTIONS, FlyPolicy, load_graph
from fly_policy.synth_env import OdorTaxisEnv
from sense.frame import frame_to_vector
from sense.goal_to_sense import load_goal


def rollout(env, model, max_steps=80):
    frame, facts, _ = env.reset()
    obs, acts, logps, vals, rewards, dones = [], [], [], [], [], []
    done = False
    while not done and len(obs) < max_steps:
        # Eye size comes from the model, not a default: the vector reserves one slot
        # per ommatidium and everything after the retina shifts with it, so a
        # hard-coded 64 would silently misalign every other modality.
        o = torch.tensor(frame_to_vector(frame, model.n_retina), dtype=torch.float32)
        with torch.no_grad():
            action, logp, value, _ = model.act(o, greedy=False)
        a = int(action.item())
        frame, r, done, facts = env.step(ACTIONS[a])
        obs.append(o)
        acts.append(action)
        logps.append(logp)
        vals.append(value)
        rewards.append(r)
        dones.append(done)
    return obs, acts, logps, vals, rewards, dones


def advantages(rewards, values, gamma=0.98, lam=0.95):
    adv, ret = [], []
    gae = 0.0
    next_v = 0.0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_v - v
        gae = delta + gamma * lam * gae
        adv.append(gae)
        ret.append(gae + v)
        next_v = v
    adv.reverse()
    ret.reverse()
    return adv, ret


def ppo_epoch_update(model, obs_t, act_t, oldlp, adv, ret, opt, clip=0.2,
                     minibatch: int | None = None):
    """One PPO pass, in minibatches.

    Minibatching is not a tuning nicety here, it is a memory requirement. The settle
    keeps every intermediate rate for the backward pass, so peak memory is
    `batch x edges x settle steps`. On the full male-cns graph one step of a single
    sample is already 6.29M edge activations; a whole epoch's batch of a few thousand
    steps would need tens of gigabytes and be killed. Advantages are still normalised
    over the full batch, so the update itself is unchanged -- only the arithmetic is
    split up.

    `minibatch=None` keeps the old single-pass behaviour, which is what the small
    synthetic-environment runs want.
    """
    adv_t = torch.as_tensor(adv, dtype=torch.float32)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.as_tensor(ret, dtype=torch.float32)
    n = obs_t.shape[0]
    size = n if not minibatch else max(1, int(minibatch))

    total = 0.0
    batches = 0
    for start in range(0, n, size):
        stop = min(n, start + size)
        logits, values = model(obs_t[start:stop])
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act_t[start:stop])
        ratio = torch.exp(logp - oldlp[start:stop])
        chunk_adv = adv_t[start:stop]
        surr1 = ratio * chunk_adv
        surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * chunk_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, ret_t[start:stop])
        entropy = dist.entropy().mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item())
        batches += 1
    return total / max(1, batches)


def ppo(model, goal, epochs=8, episodes=16, lr=3e-4, clip=0.2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last_ret = 0.0
    for ep in range(epochs):
        batch_obs, batch_act, batch_oldlp, batch_adv, batch_ret = [], [], [], [], []
        rets = []
        for i in range(episodes):
            env = OdorTaxisEnv(goal, seed=100 + ep * 17 + i)
            obs, acts, logps, vals, rewards, _ = rollout(env, model)
            vs = [float(v.item()) for v in vals]
            adv, ret = advantages(rewards, vs)
            batch_obs.extend(obs)
            batch_act.extend(acts)
            batch_oldlp.extend(logps)
            batch_adv.extend(adv)
            batch_ret.extend(ret)
            rets.append(sum(rewards))
        last_ret = sum(rets) / max(1, len(rets))
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
        print(f"ppo epoch {ep} return={last_ret:.3f} loss={loss:.4f}")
    return model, last_ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "fly_mc.pt"))
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "fly_mc_ppo.pt"))
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--episodes", type=int, default=12)
    args = p.parse_args()
    graph = load_graph(ROOT)
    ckpt = Path(args.ckpt)
    if ckpt.exists():
        model = FlyPolicy.load(graph, ckpt)
    else:
        from sense.frame import vector_size
        model = FlyPolicy(graph, vector_size(64))
    goal = load_goal(ROOT / "configs" / "goals" / "collect_oak.json")
    model, ret = ppo(model, goal, epochs=args.epochs, episodes=args.episodes)
    out = Path(args.out)
    model.save_checkpoint(out, extra={"ppo_return": ret})
    export_safetensors(model, out.with_suffix(".safetensors"))
    print(f"wrote {out} return={ret:.3f}")


if __name__ == "__main__":
    load_repo_env()
    main()
