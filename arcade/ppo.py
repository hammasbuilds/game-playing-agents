"""PPO, with checkpoints taken from the very first step.

The checkpoints are the point of this file as much as the learning is. The thing
worth showing about a trained agent is not its final score but the difference
between how it played at the start and how it plays at the end, and that
comparison only exists if the weights were saved before training touched them.
So a checkpoint is written at step 0 -- the randomly initialised network, which
is the "before" in every side-by-side clip -- and at a set of fractions after.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .envs import CATALOGUE, Game, n_actions, obs_shape
from .model import ActorCritic
from .vec import VecArcade, fine_timer


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 12
    rollout: int = 128
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True            # kept for older callers; see lr_schedule
    lr_schedule: str = "linear"       # linear | constant | cosine | warmup_cosine
    warmup: float = 0.05              # share of training spent warming up
    torch_threads: int = 4
    width: int = 1
    arch: str = "nature"              # nature | impala | small
    device: str = "auto"
    # Fractions of training at which to save the weights. 0.0 is the untrained
    # network and is not optional: it is the "step 1" of every demo.
    checkpoints: tuple[float, ...] = (0.0, 0.05, 0.15, 0.35, 0.65, 1.0)


class Rollout:
    def __init__(self, steps: int, n_envs: int, shape):
        self.obs = np.zeros((steps, n_envs, *shape), dtype=np.uint8)
        self.actions = np.zeros((steps, n_envs), dtype=np.int64)
        self.logp = np.zeros((steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((steps, n_envs), dtype=np.float32)
        self.values = np.zeros((steps, n_envs), dtype=np.float32)


LR_SCHEDULES = ("linear", "constant", "cosine", "warmup_cosine")


def learning_rate_at(cfg: "PPOConfig", update: int, n_updates: int) -> float:
    """The learning rate for this update, as a shape over the whole run.

    A single number is a schedule too -- the flat one -- and saying which shape
    is in use beats a boolean called `anneal_lr` that silently means "linear".
    The shapes:

    * **linear** — full rate at the start, zero at the end. The PPO default, and
      what every result in this repository was trained with.
    * **constant** — the same rate throughout. Useful when a run will be
      continued, since a decayed rate leaves the next run starting from nearly
      nothing.
    * **cosine** — a smooth fall that spends longer near the top than linear and
      lands softly.
    * **warmup_cosine** — rises over the first few percent, then the cosine fall.
      Worth having for the wider trunks, where the first updates of a randomly
      initialised residual stack can be violent enough to matter.
    """
    if n_updates <= 1:
        return cfg.lr
    progress = (update - 1) / n_updates          # 0 at the first update, ->1 at the last
    schedule = cfg.lr_schedule if cfg.lr_schedule in LR_SCHEDULES else "linear"
    if schedule == "constant":
        return cfg.lr
    if schedule == "linear":
        return cfg.lr * (1.0 - progress)
    if schedule == "cosine":
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    warm = max(1e-6, cfg.warmup)
    if progress < warm:
        return cfg.lr * (progress / warm)
    after = (progress - warm) / max(1e-6, 1.0 - warm)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * after))


def compute_gae(buf: Rollout, last_value: np.ndarray, gamma: float, lam: float):
    steps, n_envs = buf.rewards.shape
    adv = np.zeros_like(buf.rewards)
    running = np.zeros(n_envs, dtype=np.float32)
    next_value = last_value
    for t in reversed(range(steps)):
        mask = 1.0 - buf.dones[t]
        delta = buf.rewards[t] + gamma * next_value * mask - buf.values[t]
        running = delta + gamma * lam * mask * running
        adv[t] = running
        next_value = buf.values[t]
    return adv, adv + buf.values


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was asked for and torch.cuda.is_available() is False")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(choice)


def save_checkpoint(model, game: Game, cfg: PPOConfig, out_dir: Path, step: int,
                    tag: str, stats: dict) -> Path:
    path = out_dir / f"ckpt_{tag}.pt"
    # The architecture is saved with the weights: a checkpoint that does not say
    # what shape it is can only be loaded by guessing, and the guess is wrong the
    # moment a second architecture exists.
    torch.save({"model": model.state_dict(), "game": game.key, "width": cfg.width,
                "arch": cfg.arch, "n_actions": n_actions(game),
                "obs_shape": obs_shape(game), "step": step, "tag": tag,
                "stats": stats}, path)
    return path


def train(game: Game, cfg: PPOConfig, out_dir: Path, seed: int = 0,
          log_every: int = 10) -> dict:
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)

    shape = obs_shape(game)
    actions = n_actions(game)
    model = ActorCritic(shape, actions, width=cfg.width, arch=cfg.arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    batch = cfg.n_envs * cfg.rollout
    n_updates = max(1, cfg.total_steps // batch)
    wanted = sorted({max(1, int(round(f * n_updates))) if f > 0 else 0
                     for f in cfg.checkpoints})
    buf = Rollout(cfg.rollout, cfg.n_envs, shape)

    history: list[dict] = []
    saved: list[dict] = []
    recent: list[float] = []
    lengths: list[int] = []

    timer = fine_timer()
    timer.__enter__()
    envs = VecArcade(game, cfg.n_envs, seed=seed)
    started = time.time()
    step_count = 0
    try:
        if 0 in wanted:   # the untrained network, before a single gradient step
            path = save_checkpoint(model, game, cfg, out_dir, 0, "000", {})
            saved.append({"tag": "000", "update": 0, "steps": 0, "path": path.name,
                          "return_mean": None})
            print(f"checkpoint 000 at step 0 (untrained) -> {path.name}", flush=True)

        obs = envs.reset()
        for update in range(1, n_updates + 1):
            lr_now = learning_rate_at(cfg, update, n_updates)
            for group in opt.param_groups:
                group["lr"] = lr_now

            for t in range(cfg.rollout):
                action, logp, value = model.act(torch.from_numpy(obs))
                buf.obs[t], buf.actions[t], buf.logp[t], buf.values[t] = obs, action, logp, value
                obs, reward, done, finished = envs.step(action)
                buf.rewards[t], buf.dones[t] = reward, done.astype(np.float32)
                for episode_return, episode_length in finished:
                    recent.append(episode_return)
                    lengths.append(episode_length)
                step_count += cfg.n_envs

            with torch.no_grad():
                _, last_value = model(torch.from_numpy(obs).to(device))
            adv, ret = compute_gae(buf, last_value.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

            b_obs = torch.from_numpy(buf.obs.reshape(batch, *shape)).to(device)
            b_act = torch.from_numpy(buf.actions.reshape(batch)).to(device)
            b_logp = torch.from_numpy(buf.logp.reshape(batch)).to(device)
            b_adv = torch.from_numpy(adv.reshape(batch)).to(device)
            b_ret = torch.from_numpy(ret.reshape(batch)).to(device)

            mb_size = batch // cfg.minibatches
            index = np.arange(batch)
            losses = []
            for _ in range(cfg.epochs):
                np.random.shuffle(index)
                for start in range(0, batch, mb_size):
                    mb = index[start:start + mb_size]
                    new_logp, entropy, value = model.evaluate(b_obs[mb], b_act[mb])
                    ratio = (new_logp - b_logp[mb]).exp()
                    a = b_adv[mb]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    v_loss = 0.5 * (value - b_ret[mb]).pow(2).mean()
                    loss = pg + cfg.value_coef * v_loss - cfg.entropy_coef * entropy.mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    opt.step()
                    losses.append((pg.item(), v_loss.item(), entropy.mean().item()))

            recent = recent[-100:]
            lengths = lengths[-100:]
            pg, v_loss, ent = (float(np.mean([l[i] for l in losses])) for i in range(3))
            row = {"update": update, "steps": step_count,
                   "return_mean": float(np.mean(recent)) if recent else None,
                   "length_mean": float(np.mean(lengths)) if lengths else None,
                   "sps": round(step_count / (time.time() - started), 1),
                   "pg_loss": round(pg, 4), "value_loss": round(v_loss, 4),
                   "entropy": round(ent, 4)}
            history.append(row)

            if update in wanted:
                tag = f"{round(100 * update / n_updates):03d}"
                stats = {"return_mean": row["return_mean"], "sps": row["sps"]}
                path = save_checkpoint(model, game, cfg, out_dir, step_count, tag, stats)
                saved.append({"tag": tag, "update": update, "steps": step_count,
                              "path": path.name, "return_mean": row["return_mean"]})
                # Flush the registry with the checkpoint rather than on the next
                # logging tick. A run stopped in between leaves weights on disk
                # that nothing downstream can see: racing kept a 419k-step
                # checkpoint that its own ladder never scored, because the stop
                # landed in that gap.
                (out_dir / "checkpoints.json").write_text(json.dumps(saved, indent=1))
                print(f"checkpoint {tag} at {step_count} steps, "
                      f"training return {row['return_mean']} -> {path.name}", flush=True)

            if update % log_every == 0 or update == n_updates:
                print(f"u{update}/{n_updates} steps={step_count} "
                      f"return={row['return_mean']} sps={row['sps']} "
                      f"ent={row['entropy']}", flush=True)
                (out_dir / "history.json").write_text(json.dumps(history, indent=1))
                (out_dir / "checkpoints.json").write_text(json.dumps(saved, indent=1))
    finally:
        envs.close()
        timer.__exit__(None, None, None)

    summary = {"game": game.key, "env_id": game.env_id, "updates": len(history),
               "steps": step_count, "wall_seconds": round(time.time() - started, 1),
               "final_return_mean": history[-1]["return_mean"] if history else None,
               "params": model.n_params(), "width": cfg.width, "arch": cfg.arch,
               "device": str(device), "envs": cfg.n_envs,
               "learning_rate": cfg.lr, "lr_schedule": cfg.lr_schedule,
               "entropy_coef": cfg.entropy_coef,
               "rollout": cfg.rollout, "epochs": cfg.epochs,
               "checkpoints": saved}
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    (out_dir / "checkpoints.json").write_text(json.dumps(saved, indent=1))
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
