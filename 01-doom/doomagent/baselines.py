"""Two opponents for the learned policy to beat, so the score has a floor.

A learned score only means something next to what the scenario gives away for
free. `random` is the usual floor; `spray` is the cheap human reflex -- hold the
trigger and keep turning -- and on defend_the_center it is a much harder floor
than random, which is the point of including it.
"""
from __future__ import annotations

import numpy as np


class RandomAgent:
    name = "random"

    def __init__(self, n_actions: int, rng: np.random.Generator):
        self.n_actions, self.rng = n_actions, rng

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> int:
        return int(self.rng.integers(self.n_actions))


class SprayAgent:
    """Turn a few steps, shoot a few steps, repeat. Index 2 is ATTACK in every
    action set this project uses; the turn direction flips so it sweeps the room
    instead of spinning one way forever."""

    name = "spray"

    def __init__(self, n_actions: int, rng: np.random.Generator, turn: int = 3, fire: int = 2):
        self.n_actions, self.rng, self.turn, self.fire = n_actions, rng, turn, fire
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.direction = 0

    def act(self, obs: np.ndarray) -> int:
        phase = self.t % (self.turn + self.fire)
        self.t += 1
        if phase < self.turn:
            return self.direction
        if phase == self.turn + self.fire - 1:
            self.direction = 1 - self.direction
        return 2 if self.n_actions > 2 else 0


class PolicyAgent:
    """The trained network. `greedy` takes the argmax; sampling is kept as an
    option because on health_gathering the argmax can walk into a wall and stay
    there, and the sampled policy scores higher for the same weights."""

    def __init__(self, model, greedy: bool = True, name: str = "ppo"):
        import torch
        self.torch = torch
        self.model = model
        self.greedy = greedy
        self.name = name

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> int:
        # Keep uint8: ActorCritic.forward divides by 255 only for that dtype, so
        # casting to float here would feed the policy 0-255 inputs it never saw.
        obs_t = self.torch.from_numpy(np.ascontiguousarray(obs[None]))
        action, _, _ = self.model.act(obs_t, greedy=self.greedy)
        return int(action[0])
