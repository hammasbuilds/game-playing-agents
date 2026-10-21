"""Crafter, wrapped so the arcade's trainer can play it.

Crafter predates the Gymnasium API: `reset()` takes no seed and returns the
frame alone, `step()` returns four values rather than five, and the frame is
64x64 RGB rather than something Atari-shaped. The arcade trainer wants a
Gymnasium environment handing back 84x84 greyscale in stacks of four.

This is the adapter, and it exists for one reason: the Crafter result in this
repository is a **planner** that trains nothing at all, scoring 12.73% against
1.13% for random. That number has no learned arm beside it. Until one exists,
"a planner beats learning here" is a claim nobody has tested.

What the learned agent is allowed is exactly what the planner is allowed: the
64x64 frame, nothing else. No tile decoding, no map, no inventory read out of
the game's state.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .envs import FRAME_H, FRAME_W, STACK


class CrafterGym(gym.Env):
    """Gymnasium's five-value API over Crafter's four."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, length: int = 10_000, render_mode: str | None = None):
        import crafter

        self._crafter = crafter
        self.length = length
        self.render_mode = render_mode
        self._env = crafter.Env(area=(64, 64), view=(9, 9), size=(64, 64),
                                length=length)
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self._last = None

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            # Crafter takes its seed at construction, not at reset, so a new
            # seed means a new world. Rebuilding is cheap and it is the only
            # way to give two agents the same world.
            self._env = self._crafter.Env(area=(64, 64), view=(9, 9),
                                          size=(64, 64), length=self.length,
                                          seed=int(seed))
            self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self._last = self._env.reset()
        return self._last, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        self._last = obs
        # Crafter ends an episode both by death and by running out of steps and
        # does not say which. Reporting every ending as a termination teaches
        # the value head that the world stops at the time limit; reporting the
        # length limit as a truncation is what bootstrapping expects.
        truncated = bool(done and info.get("discount", 1.0) == 1.0)
        terminated = bool(done and not truncated)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return self._env.render((256, 256)) if self.render_mode else None

    def close(self):
        pass


def make_crafter(seed: int | None = None, render: bool = False) -> gym.Env:
    """The same wrapper stack every other arcade game gets, minus the resize.

    Crafter draws 64x64. The Atari games are cut down to 84x84 because they draw
    160x210, but resizing *up* to 84 invents pixels, costs a resample on every
    step, and makes the first convolution 2.7x larger for nothing: 427,874
    parameters against 157,538 at the native size. The frame is left alone.
    """
    env = CrafterGym(render_mode="rgb_array" if render else None)
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=STACK)
    if seed is not None:
        env.reset(seed=seed)
    return env
