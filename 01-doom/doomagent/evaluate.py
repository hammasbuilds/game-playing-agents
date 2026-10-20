"""Score an agent over N fixed episodes and report the spread, not just the mean.

Every agent in a comparison is handed the same seed list, so a lucky map does not
decide the table.
"""
from __future__ import annotations

import numpy as np

from .env import DoomEnv, EnvConfig
from .vecenv import fine_timer


def run_episodes(agent, env_cfg: EnvConfig, seeds: list[int]) -> dict:
    returns, lengths = [], []
    with fine_timer():  # the same 1 ms scheduler tick the trainer needs
        for seed in seeds:
            env = DoomEnv(EnvConfig(**{**env_cfg.__dict__, "seed": seed}))
            try:
                obs = env.reset()
                agent.reset()
                steps = 0
                while True:
                    obs, _, done = env.step(agent.act(obs))
                    steps += 1
                    if done:
                        break
                returns.append(env.episode_return())
                lengths.append(steps)
            finally:
                env.close()
    arr = np.array(returns, dtype=np.float64)
    return {
        "agent": getattr(agent, "name", type(agent).__name__),
        "episodes": len(seeds),
        "return_mean": round(float(arr.mean()), 2),
        "return_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 2),
        "return_min": round(float(arr.min()), 2),
        "return_max": round(float(arr.max()), 2),
        "steps_mean": round(float(np.mean(lengths)), 1),
        "returns": [round(float(r), 2) for r in returns],
    }
