"""Several games in worker processes, stepped together.

This is the same arrangement the DOOM project arrived at, for the same reasons,
and it carries the same hard-won line: a worker that sleeps between steps pays a
wake-up rounded up to the Windows scheduler tick, and since Windows 10 2004 that
resolution is **per process**, so `timeBeginPeriod(1)` has to be called inside
the worker. It was worth 5x there and it is worth having here for free.

Only the newest frame crosses the pipe, as uint8. The stack is kept on the
learner's side.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys

import numpy as np

from .envs import FRAME_H, FRAME_W, STACK, Game, make, to_uint8


@contextlib.contextmanager
def fine_timer():
    """Ask Windows for a 1 ms scheduler tick for the duration of the block."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)
    try:
        yield
    finally:
        winmm.timeEndPeriod(1)


def _worker(remote, game: Game, seed: int) -> None:
    env = make(game, seed=seed)
    obs, _ = env.reset(seed=seed)
    episode_return, episode_length = 0.0, 0
    try:
        with fine_timer():
            while True:
                cmd, data = remote.recv()
                if cmd == "step":
                    obs, reward, terminated, truncated, _ = env.step(int(data))
                    episode_return += float(reward)
                    episode_length += 1
                    done = bool(terminated or truncated)
                    finished = (episode_return, episode_length) if done else None
                    if done:
                        obs, _ = env.reset()
                        episode_return, episode_length = 0.0, 0
                    # Only the newest frame of the stack needs to travel.
                    remote.send((to_uint8(obs)[-1], float(reward), done, finished))
                elif cmd == "reset":
                    obs, _ = env.reset()
                    episode_return, episode_length = 0.0, 0
                    remote.send(to_uint8(obs)[-1])
                elif cmd == "close":
                    remote.send(None)
                    break
    finally:
        env.close()
        remote.close()


class VecArcade:
    """n copies of one game behind a single step() call."""

    def __init__(self, game: Game, n_envs: int, seed: int = 0):
        ctx = mp.get_context("spawn")
        self.n_envs = n_envs
        self.remotes, self.procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            # Distinct seeds: identical ones would make n copies of a single
            # episode and the batch would hold one trajectory's information at n
            # times the cost.
            proc = ctx.Process(target=_worker, args=(child, game, seed + 1000 * i),
                               daemon=True)
            proc.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(proc)
        self.frames = np.zeros((n_envs, STACK, FRAME_H, FRAME_W), dtype=np.uint8)

    def _push(self, i: int, frame: np.ndarray) -> None:
        self.frames[i, :-1] = self.frames[i, 1:]
        self.frames[i, -1] = frame

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))
        self.frames[:] = 0
        for i, remote in enumerate(self.remotes):
            self._push(i, remote.recv())
        return self.frames.copy()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        finished = []
        for i, remote in enumerate(self.remotes):
            frame, reward, done, episode = remote.recv()
            rewards[i], dones[i] = reward, done
            if done:
                # A reset env starts from a blank stack: carrying the last
                # episode's frames in would teach the policy that a game-over
                # screen predicts the next episode's opening.
                self.frames[i] = 0
            self._push(i, frame)
            if episode is not None:
                finished.append(episode)
        return self.frames.copy(), rewards, dones, finished

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
                remote.recv()
            except (OSError, EOFError):
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
