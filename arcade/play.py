"""Load a checkpoint and play, either to score it or to film it.

Everything that scores a checkpoint and everything that records one goes through
`rollout` here, so the clip a viewer watches was produced by the same code path
as the number underneath it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .envs import CATALOGUE, Game, make, obs_shape, to_uint8
from .model import ActorCritic


@dataclass
class Episode:
    total_reward: float
    length: int
    frames: list[np.ndarray]


def load(checkpoint: Path) -> tuple[ActorCritic, Game, dict]:
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    game = CATALOGUE[blob["game"]]
    # Checkpoints written before architectures existed carry no 'arch' and are
    # all the nature trunk, so that is the default rather than an error.
    model = ActorCritic(tuple(blob["obs_shape"]), int(blob["n_actions"]),
                        width=int(blob.get("width", 1)),
                        arch=str(blob.get("arch", "nature")))
    model.load_state_dict(blob["model"])
    model.eval()
    return model, game, blob


def rollout(model: ActorCritic | None, game: Game, seed: int, max_steps: int = 3000,
            record: bool = False, greedy: bool = False,
            rng: np.random.Generator | None = None) -> Episode:
    """One episode. `model=None` plays uniformly at random, which is the floor.

    Sampling rather than taking the argmax is the default on purpose: a policy
    that has learned little is nearly uniform, and its argmax is one arbitrary
    action repeated forever, which looks like a bug rather than like an untrained
    agent. The recordings would misrepresent the "before" state.
    """
    env = make(game, render=record)
    rng = rng or np.random.default_rng(seed)
    frames: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=seed)
        total, steps = 0.0, 0
        for _ in range(max_steps):
            if record:
                shot = env.render()
                if shot is not None:
                    frames.append(np.asarray(shot))
            if model is None:
                action = int(rng.integers(env.action_space.n))
            else:
                stacked = torch.from_numpy(to_uint8(obs)[None])
                picked, _, _ = model.act(stacked, greedy=greedy)
                action = int(picked[0])
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()
    return Episode(total_reward=total, length=steps, frames=frames)


def score(checkpoint: Path | None, game: Game, seeds: list[int],
          max_steps: int = 3000) -> dict:
    """Mean and spread over a fixed set of levels.

    Every checkpoint of a game is scored on the same seeds, so the ladder from
    untrained to final is a like-for-like comparison and not a tour of different
    levels.
    """
    model = None
    if checkpoint is not None:
        model, loaded, _ = load(checkpoint)
        game = loaded
    returns, lengths = [], []
    for seed in seeds:
        episode = rollout(model, game, seed, max_steps=max_steps)
        returns.append(episode.total_reward)
        lengths.append(episode.length)
    arr = np.array(returns, dtype=np.float64)
    return {"episodes": len(seeds),
            "return_mean": round(float(arr.mean()), 2),
            "return_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 2),
            "return_min": round(float(arr.min()), 2),
            "return_max": round(float(arr.max()), 2),
            "length_mean": round(float(np.mean(lengths)), 1),
            "returns": [round(float(r), 2) for r in returns]}
