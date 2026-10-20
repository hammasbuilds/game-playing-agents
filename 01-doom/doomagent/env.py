"""ViZDoom wrapped down to what a small CPU policy can chew on.

The raw game hands back a 3x240x320 RGB frame 35 times a second. That is far more
pixels than this agent needs and far more decisions than it benefits from, so each
step here holds one action for `frame_skip` tics and returns a stack of the last
`stack` greyscale 60x80 frames.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import vizdoom as vzd

# Scenario -> the buttons the agent is allowed to press. Keeping the action set
# minimal matters more on CPU than any network change: every extra button is a
# wider softmax and a longer search for the same behaviour.
ACTION_SETS: dict[str, list[list[int]]] = {
    # basic.cfg declares MOVE_LEFT, MOVE_RIGHT, ATTACK
    "basic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # defend_the_center.cfg declares TURN_LEFT, TURN_RIGHT, ATTACK
    "defend_the_center": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # health_gathering.cfg declares MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
    "health_gathering": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "health_gathering_supreme": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    # deadly_corridor.cfg declares MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD,
    # MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT -- seven buttons, and a wider softmax
    # is the one change that reliably costs a CPU run its budget. Four of them
    # are enough to walk the corridor and shoot what is in it.
    "deadly_corridor": [
        [0, 0, 0, 1, 0, 0, 0],   # forward
        [0, 0, 0, 0, 0, 1, 0],   # turn left
        [0, 0, 0, 0, 0, 0, 1],   # turn right
        [0, 0, 1, 0, 0, 0, 0],   # attack
    ],
}

H, W = 60, 80

# What the game renders, separately from what the agent is fed. The agent always
# gets 60x80; the screen it is cut from does not have to be 160x120, and when the
# point is a film rather than a training step it should not be -- a recording of
# the 160x120 buffer is the reason these demos looked like 1996.
#
# Every option here is an exact multiple of 80x60, so the agent's frame is a
# clean box filter of the screen in all of them rather than a resampled
# approximation of one.
RESOLUTIONS = {
    "160x120": vzd.ScreenResolution.RES_160X120,     # 2x -- what training uses
    "320x240": vzd.ScreenResolution.RES_320X240,     # 4x
    "640x480": vzd.ScreenResolution.RES_640X480,     # 8x -- what films use
    "800x600": vzd.ScreenResolution.RES_800X600,     # 10x
    "1280x960": vzd.ScreenResolution.RES_1280X960,   # 16x
}


@dataclass
class EnvConfig:
    scenario: str = "defend_the_center"
    frame_skip: int = 4
    stack: int = 4
    seed: int | None = None
    window: bool = False
    resolution: str = "160x120"


def _downsample(screen: np.ndarray) -> np.ndarray:
    """3xHRxWR uint8 RGB -> 60x80 uint8 greyscale.

    A box filter over whatever the screen resolution is, using the ITU-R 601 luma
    weights; cv2 is not a dependency of this project and for frames this size
    numpy is quicker than the import. Staying in uint8 is not a detail: these
    frames cross a process boundary every step, and sending them as float32 made
    the rollout 2.5x slower than the game.

    The factor is derived from the frame rather than fixed at 2, so the same
    agent can be fed from a 160x120 screen while training and a 640x480 one while
    being filmed, and sees the same thing either way: the mean over a block is
    the mean over its sub-blocks.
    """
    rgb = screen.astype(np.float32)
    grey = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    fy, fx = grey.shape[0] // H, grey.shape[1] // W
    if fy * H != grey.shape[0] or fx * W != grey.shape[1]:
        raise ValueError(f"screen {grey.shape} is not a whole multiple of {H}x{W}")
    return grey.reshape(H, fy, W, fx).mean(axis=(1, 3)).astype(np.uint8)


class DoomEnv:
    """Single ViZDoom instance. No gym dependency: step() returns plain tuples."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        if cfg.scenario not in ACTION_SETS:
            raise ValueError(f"unknown scenario {cfg.scenario!r}; have {sorted(ACTION_SETS)}")
        game = vzd.DoomGame()
        game.load_config(os.path.join(vzd.scenarios_path, f"{cfg.scenario}.cfg"))
        if cfg.resolution not in RESOLUTIONS:
            raise ValueError(f"unknown resolution {cfg.resolution!r}; "
                             f"have {sorted(RESOLUTIONS)}")
        game.set_screen_resolution(RESOLUTIONS[cfg.resolution])
        game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        game.set_window_visible(cfg.window)
        if cfg.seed is not None:
            game.set_seed(cfg.seed)
        game.init()
        self.game = game
        self.actions = ACTION_SETS[cfg.scenario]
        self.frames = np.zeros((cfg.stack, H, W), dtype=np.uint8)

    # -- plumbing ---------------------------------------------------------
    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        return (self.cfg.stack, H, W)

    def frame(self) -> np.ndarray:
        """The newest 60x80 uint8 frame. The vectorised trainer sends this across
        the process boundary and keeps the stack on its own side."""
        state = self.game.get_state()
        if state is None:  # episode just finished; the last frame still stands
            return self.frames[-1]
        return _downsample(state.screen_buffer)

    def screen_rgb(self) -> np.ndarray | None:
        """The full-resolution colour frame, for recording. None between episodes."""
        state = self.game.get_state()
        if state is None:
            return None
        return np.transpose(state.screen_buffer, (1, 2, 0)).copy()

    def _observe(self) -> np.ndarray:
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = self.frame()
        return self.frames.copy()

    def reset(self) -> np.ndarray:
        self.game.new_episode()
        self.frames[:] = 0
        return self._observe()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        reward = self.game.make_action(self.actions[action], self.cfg.frame_skip)
        done = self.game.is_episode_finished()
        return self._observe(), float(reward), done

    def episode_return(self) -> float:
        return float(self.game.get_total_reward())

    def close(self) -> None:
        self.game.close()
