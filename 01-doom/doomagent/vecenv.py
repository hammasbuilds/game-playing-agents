"""Run several ViZDoom instances in worker processes.

ViZDoom keeps per-instance global state, so one instance per process is the only
arrangement that is both fast and safe. Two details carry the throughput:

* a worker sends back a single 60x80 **uint8** frame, not the stack, and not
  float32 -- the stacked-float version of this file ran the rollout at 153
  steps/s against the 383 steps/s a bare env manages, i.e. the pipe, not DOOM,
  was the game;
* workers auto-reset, reporting the finished episode's return alongside the
  first frame of the next one, which keeps the learner's loop branchless.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys
from dataclasses import replace

import numpy as np

from .env import H, W, DoomEnv, EnvConfig


@contextlib.contextmanager
def fine_timer():
    """Ask Windows for a 1 ms scheduler tick for the duration of the block.

    This is worth 5x here and it took a profiler to believe it. A worker sits in
    `remote.recv()` between steps, so every step pays a wake-up that the OS
    rounds up to the timer granularity -- 15.6 ms by default, against the 0.7 ms
    the game actually needs. Setting it raised an 8-worker rollout from 292 to
    1530 steps/s.

    Since Windows 10 2004 the resolution is per-process, which is the trap: the
    call has to happen *in the worker*, not in the trainer that spawned it.
    """
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


def _worker(remote, cfg: EnvConfig) -> None:
    env = DoomEnv(cfg)
    env.reset()
    try:
        with fine_timer():
            while True:
                cmd, data = remote.recv()
                if cmd == "step":
                    _, reward, done = env.step(int(data))
                    finished = env.episode_return() if done else None
                    frame = env.frame()
                    if done:
                        # Report the finished episode's return, then hand back
                        # the first frame of the next one in the same message.
                        env.reset()
                        frame = env.frame()
                    remote.send((frame, reward, done, finished))
                elif cmd == "reset":
                    env.reset()
                    remote.send(env.frame())
                elif cmd == "spec":
                    remote.send((env.obs_shape, env.n_actions))
                elif cmd == "close":
                    remote.send(None)
                    break
    finally:
        env.close()
        remote.close()


class VecDoom:
    """n_envs games behind one step() call. Observations come out as uint8
    (n_envs, stack, H, W); the policy normalises them."""

    def __init__(self, cfg: EnvConfig, n_envs: int):
        ctx = mp.get_context("spawn")  # Windows has no fork
        self.n_envs = n_envs
        self.stack = cfg.stack
        self.remotes, self.procs = [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            # Distinct seeds: identical seeds would make n_envs copies of one
            # episode, so the batch would carry a single trajectory's worth of
            # information at n_envs times the compute.
            seed = None if cfg.seed is None else cfg.seed + 1000 * i
            p = ctx.Process(target=_worker, args=(child, replace(cfg, seed=seed)), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.remotes[0].send(("spec", None))
        self.obs_shape, self.n_actions = self.remotes[0].recv()
        self.frames = np.zeros((n_envs, cfg.stack, H, W), dtype=np.uint8)

    def _push(self, i: int, frame: np.ndarray) -> None:
        self.frames[i, :-1] = self.frames[i, 1:]
        self.frames[i, -1] = frame

    def reset(self) -> np.ndarray:
        for r in self.remotes:
            r.send(("reset", None))
        self.frames[:] = 0
        for i, r in enumerate(self.remotes):
            self._push(i, r.recv())
        return self.frames.copy()

    def step(self, actions: np.ndarray):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        finished: list[float] = []
        for i, r in enumerate(self.remotes):
            frame, reward, done, episode_return = r.recv()
            rewards[i], dones[i] = reward, done
            if done:
                # A reset env starts from a blank stack: carrying the dead
                # episode's frames into the new one would teach the policy that
                # a death screen predicts the next map's opening.
                self.frames[i] = 0
            self._push(i, frame)
            if episode_return is not None:
                finished.append(episode_return)
        return self.frames.copy(), rewards, dones, finished

    def close(self) -> None:
        for r in self.remotes:
            try:
                r.send(("close", None))
                r.recv()
            except (OSError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
