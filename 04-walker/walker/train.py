"""PPO for a MuJoCo robot, on a laptop CPU.

    python -m walker.train --env Walker2d-v5 --steps 1500000

Same algorithm as the arcade — clipped surrogate, GAE, value loss, entropy
bonus — with the three changes continuous control actually needs:

* a Gaussian actor instead of a softmax, because an action is a torque per joint;
* running normalisation of observations, because joint angles and velocities do
  not share a scale and the network should not spend its first thousand updates
  discovering that;
* checkpoints from step zero, which is the house rule here: the demonstration is
  the untrained robot and the trained one walking the same episode, so the
  untrained one has to be kept.

Gymnasium's own `SyncVectorEnv` does the stepping. MuJoCo is fast enough on CPU
that process-based workers would spend more time on pipes than on physics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .policy import GaussianActorCritic, RunningNorm

# Fractions of training at which a checkpoint is kept, zero included: without
# the untrained policy there is nothing to put beside the trained one.
LADDER = (0.0, 0.05, 0.15, 0.35, 0.65, 1.0)


def make_envs(env_id: str, n: int, seed: int) -> gym.vector.VectorEnv:
    def one(i: int):
        def build():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + i)
            return env
        return build
    return gym.vector.SyncVectorEnv([one(i) for i in range(n)])


def save(path: Path, model, norm, env_id: str, step: int, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "norm": norm.state_dict(),
                "env_id": env_id, "step": step,
                "obs_dim": model.obs_dim, "act_dim": model.act_dim,
                "history": history}, path)


def train(env_id: str, steps: int, envs: int, rollout: int, lr: float,
          gamma: float, lam: float, clip: float, entropy_coef: float,
          epochs: int, minibatches: int, out: Path, seed: int,
          log=print) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    vec = make_envs(env_id, envs, seed)
    obs_dim = int(np.prod(vec.single_observation_space.shape))
    act_dim = int(np.prod(vec.single_action_space.shape))
    low = vec.single_action_space.low
    high = vec.single_action_space.high

    model = GaussianActorCritic(obs_dim, act_dim)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)
    norm = RunningNorm((obs_dim,))

    batch = envs * rollout
    n_updates = max(1, steps // batch)
    wanted = sorted({0} | {max(1, int(round(f * n_updates))) for f in LADDER if f > 0})
    history: list[dict] = []
    returns_seen: list[float] = []

    save(out / "ckpt_000.pt", model, norm, env_id, 0, history)
    log(f"{env_id}: {obs_dim} observations, {act_dim} joints, "
        f"{n_updates} updates of {batch} steps")

    raw_obs, _ = vec.reset(seed=seed)
    norm.update(raw_obs)
    obs = torch.from_numpy(norm(raw_obs))
    started = time.time()
    step_count = 0

    for update in range(1, n_updates + 1):
        buf_obs = torch.zeros(rollout, envs, obs_dim)
        buf_act = torch.zeros(rollout, envs, act_dim)
        buf_logp = torch.zeros(rollout, envs)
        buf_val = torch.zeros(rollout, envs)
        buf_rew = torch.zeros(rollout, envs)
        buf_done = torch.zeros(rollout, envs)

        for t in range(rollout):
            action, logp, value = model.act(obs)
            buf_obs[t], buf_act[t] = obs, torch.from_numpy(action)
            buf_logp[t], buf_val[t] = torch.from_numpy(logp), torch.from_numpy(value)
            # The policy is unbounded; the robot is not. Clipping for the
            # simulator while storing the unclipped sample keeps the log
            # probability consistent with what was actually drawn.
            raw_obs, reward, terminated, truncated, info = vec.step(
                np.clip(action, low, high))
            done = np.logical_or(terminated, truncated)
            buf_rew[t] = torch.from_numpy(np.asarray(reward, dtype=np.float32))
            buf_done[t] = torch.from_numpy(done.astype(np.float32))
            norm.update(raw_obs)
            obs = torch.from_numpy(norm(raw_obs))
            step_count += envs
            if "episode" in info:
                finished = info["episode"]
                mask = np.asarray(info.get("_episode", np.ones(envs, bool)))
                returns_seen.extend(np.asarray(finished["r"])[mask].tolist())

        with torch.no_grad():
            _, last_value = model(obs)
            advantages = torch.zeros_like(buf_rew)
            carry = torch.zeros(envs)
            for t in reversed(range(rollout)):
                alive = 1.0 - buf_done[t]
                nxt = last_value if t == rollout - 1 else buf_val[t + 1]
                delta = buf_rew[t] + gamma * nxt * alive - buf_val[t]
                carry = delta + gamma * lam * alive * carry
                advantages[t] = carry
            target = advantages + buf_val

        flat_obs = buf_obs.reshape(-1, obs_dim)
        flat_act = buf_act.reshape(-1, act_dim)
        flat_logp = buf_logp.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = target.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        chunk = max(1, batch // minibatches)
        stats = []
        for _ in range(epochs):
            order = torch.randperm(batch)
            for start in range(0, batch, chunk):
                idx = order[start:start + chunk]
                logp, ent, value = model.evaluate(flat_obs[idx], flat_act[idx])
                ratio = (logp - flat_logp[idx]).exp()
                adv = flat_adv[idx]
                policy_loss = -torch.min(
                    ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                value_loss = (value - flat_ret[idx]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coef * ent.mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimiser.step()
                stats.append((policy_loss.item(), value_loss.item(), ent.mean().item()))

        recent = returns_seen[-20:]
        row = {
            "update": update,
            "steps": step_count,
            "episodes": len(returns_seen),
            "return_mean": round(float(np.mean(recent)), 2) if recent else None,
            "value_loss": round(float(np.mean([s[1] for s in stats])), 3),
            "entropy": round(float(np.mean([s[2] for s in stats])), 3),
            "log_std": round(float(model.log_std.mean().item()), 3),
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        if update in wanted:
            save(out / f"ckpt_{update:03d}.pt", model, norm, env_id, step_count, history)
        if update == 1 or update % 10 == 0 or update == n_updates:
            log(f"u{update}/{n_updates} steps={step_count} "
                f"return={row['return_mean']} entropy={row['entropy']} "
                f"log_std={row['log_std']} sps={step_count / max(1e-9, row['seconds']):.0f}")

    save(out / "policy.pt", model, norm, env_id, step_count, history)
    summary = {
        "env_id": env_id, "steps": step_count, "updates": n_updates,
        "episodes": len(returns_seen),
        "final_return_mean": round(float(np.mean(returns_seen[-20:])), 2)
        if returns_seen else None,
        "wall_seconds": round(time.time() - started, 1),
        "params": sum(p.numel() for p in model.parameters()),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    vec.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="Walker2d-v5")
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatches", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or Path(__file__).resolve().parents[1] / "runs"
               / args.env.split("-")[0].lower())
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    handle = log_path.open("a", encoding="utf-8")

    def log(line: str) -> None:
        print(line, flush=True)
        handle.write(line + "\n")
        handle.flush()

    summary = train(args.env, args.steps, args.envs, args.rollout, args.lr,
                    args.gamma, args.lam, args.clip, args.entropy, args.epochs,
                    args.minibatches, out, args.seed, log=log)
    log(json.dumps(summary, indent=1))
    handle.close()


if __name__ == "__main__":
    main()
