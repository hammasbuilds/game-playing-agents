"""PPO, written out rather than imported, because the interesting knobs on a CPU
budget are the ones a library hides: how many envs, how long a rollout, and how
many epochs of reuse a batch can take before the clip ratio stops it anyway.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import ActorCritic
from .vecenv import VecDoom, fine_timer
from .env import EnvConfig


@dataclass
class PPOConfig:
    total_steps: int = 1_500_000      # counted in agent steps, i.e. frames // frame_skip
    n_envs: int = 8
    rollout: int = 128                # steps per env per update
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    torch_threads: int = 2            # the rollout, not the backward pass, is the bottleneck
    width: int = 1                    # channel multiplier; >1 wants a GPU
    device: str = "auto"              # "auto" takes CUDA when it is really there


class Rollout:
    """Preallocated buffer for one update's worth of transitions."""

    def __init__(self, steps: int, n_envs: int, obs_shape):
        self.obs = np.zeros((steps, n_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((steps, n_envs), dtype=np.int64)
        self.logp = np.zeros((steps, n_envs), dtype=np.float32)
        self.rewards = np.zeros((steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((steps, n_envs), dtype=np.float32)
        self.values = np.zeros((steps, n_envs), dtype=np.float32)


def compute_gae(buf: Rollout, last_value: np.ndarray, gamma: float, lam: float):
    steps, n_envs = buf.rewards.shape
    adv = np.zeros_like(buf.rewards)
    running = np.zeros(n_envs, dtype=np.float32)
    next_value = last_value
    for t in reversed(range(steps)):
        # dones[t] marks "the episode ended on this step", so the bootstrap from
        # t+1 must be masked here, not one step later.
        mask = 1.0 - buf.dones[t]
        delta = buf.rewards[t] + gamma * next_value * mask - buf.values[t]
        running = delta + gamma * lam * mask * running
        adv[t] = running
        next_value = buf.values[t]
    return adv, adv + buf.values


def resolve_device(choice: str) -> torch.device:
    """Pick a device, and say so out loud.

    "auto" falls back to CPU when CUDA is absent, which is the right default --
    but a run that *meant* to use a GPU and silently did not is a wasted run, so
    asking for "cuda" when it is unavailable is an error rather than a fallback.
    """
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda was asked for and torch.cuda.is_available() is False")
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(choice)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"device: {device} ({name})", flush=True)
    return device


def train(env_cfg: EnvConfig, cfg: PPOConfig, out_dir: Path, log_every: int = 10,
          seed: int = 0) -> dict:
    torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The trainer blocks in recv() too, so it wants the 1 ms tick as much as the
    # workers do -- see fine_timer's note on why this is per-process.
    timer = fine_timer()
    timer.__enter__()
    device = resolve_device(cfg.device)
    envs = VecDoom(env_cfg, cfg.n_envs)
    model = ActorCritic(envs.obs_shape, envs.n_actions, width=cfg.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    batch = cfg.n_envs * cfg.rollout
    n_updates = max(1, cfg.total_steps // batch)
    buf = Rollout(cfg.rollout, cfg.n_envs, envs.obs_shape)
    obs = envs.reset()

    history: list[dict] = []
    recent: list[float] = []
    started = time.time()
    step_count = 0
    try:
        for update in range(1, n_updates + 1):
            if cfg.anneal_lr:
                for g in opt.param_groups:
                    g["lr"] = cfg.lr * (1.0 - (update - 1) / n_updates)

            for t in range(cfg.rollout):
                obs_t = torch.from_numpy(obs)
                action, logp, value = model.act(obs_t)
                buf.obs[t], buf.actions[t], buf.logp[t], buf.values[t] = obs, action, logp, value
                obs, reward, done, finished = envs.step(action)
                buf.rewards[t], buf.dones[t] = reward, done.astype(np.float32)
                recent.extend(finished)
                step_count += cfg.n_envs

            with torch.no_grad():
                _, last_value = model(torch.from_numpy(obs).to(device))
            adv, ret = compute_gae(buf, last_value.cpu().numpy(), cfg.gamma, cfg.gae_lambda)

            b_obs = torch.from_numpy(buf.obs.reshape(batch, *envs.obs_shape)).to(device)
            b_act = torch.from_numpy(buf.actions.reshape(batch)).to(device)
            b_logp = torch.from_numpy(buf.logp.reshape(batch)).to(device)
            b_adv = torch.from_numpy(adv.reshape(batch)).to(device)
            b_ret = torch.from_numpy(ret.reshape(batch)).to(device)

            mb_size = batch // cfg.minibatches
            idx = np.arange(batch)
            losses = []
            for _ in range(cfg.epochs):
                np.random.shuffle(idx)
                for start in range(0, batch, mb_size):
                    mb = idx[start:start + mb_size]
                    new_logp, entropy, value = model.evaluate(b_obs[mb], b_act[mb])
                    ratio = (new_logp - b_logp[mb]).exp()
                    # Advantages are normalised per minibatch: the reward scale
                    # differs by an order of magnitude between these scenarios
                    # and this is what lets one lr serve all of them.
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

            if recent:
                recent = recent[-100:]
            pg, v_loss, ent = (float(np.mean([l[i] for l in losses])) for i in range(3))
            row = {
                "update": update,
                "steps": step_count,
                "episodes_seen": len(recent),
                "return_mean": float(np.mean(recent)) if recent else None,
                "sps": round(step_count / (time.time() - started), 1),
                "pg_loss": round(pg, 4), "value_loss": round(v_loss, 4), "entropy": round(ent, 4),
            }
            history.append(row)
            if update % log_every == 0 or update == n_updates:
                print(f"u{update}/{n_updates} steps={step_count} "
                      f"return={row['return_mean']} sps={row['sps']} ent={row['entropy']}",
                      flush=True)
                (out_dir / "history.json").write_text(json.dumps(history, indent=1))
                torch.save({"model": model.state_dict(), "obs_shape": envs.obs_shape,
                            "n_actions": envs.n_actions, "env": asdict(env_cfg),
                            "width": cfg.width}, out_dir / "policy.pt")
    finally:
        envs.close()
        timer.__exit__(None, None, None)

    torch.save({"model": model.state_dict(), "obs_shape": envs.obs_shape,
                "n_actions": envs.n_actions, "env": asdict(env_cfg),
                "width": cfg.width}, out_dir / "policy.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"updates": len(history), "steps": step_count,
            "wall_seconds": round(time.time() - started, 1),
            "final_return_mean": history[-1]["return_mean"] if history else None,
            "params": model.n_params(), "width": cfg.width, "device": str(device),
            "envs": cfg.n_envs}
