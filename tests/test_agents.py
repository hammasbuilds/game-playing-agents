"""Tests for the parts that can be checked without launching a game.

The three agents are judged by playing, which is what the evaluation scripts are
for. What is worth pinning here is the machinery underneath them, where a quiet
mistake would move a score without crashing anything: the advantage estimator,
the pathfinder, the shelter geometry, the achievement inference, and the
character grid the Brogue decoder cuts out of a frame.

    python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
# This repository lays the projects out as numbered folders at the top level,
# where game-agents keeps them under projects/ with longer names. The code under
# test is the same code.
#
# Appended, not inserted at the front: each project folder holds its own
# film.py, and ahead of the repository root it shadows the shared film package,
# so "from film import side_by_side" became a circular import into
# 03-brogue/film.py. The agent packages imported below have no name clash.
sys.path.insert(0, str(ROOT))
for _project in ("01-doom", "02-crafter", "03-brogue"):
    sys.path.append(str(ROOT / _project))


# -- 01 doom ---------------------------------------------------------------
def test_gae_matches_hand_computation():
    """One env, three steps, no episode end: the advantage telescopes."""
    from doomagent.ppo import Rollout, compute_gae

    buf = Rollout(steps=3, n_envs=1, obs_shape=(1, 2, 2))
    buf.rewards[:, 0] = [1.0, 1.0, 1.0]
    buf.values[:, 0] = [0.5, 0.5, 0.5]
    buf.dones[:] = 0.0
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(buf, np.array([0.5], dtype=np.float32), gamma, lam)

    delta = 1.0 + gamma * 0.5 - 0.5
    expected_t2 = delta
    expected_t1 = delta + gamma * lam * expected_t2
    expected_t0 = delta + gamma * lam * expected_t1
    assert adv[2, 0] == pytest.approx(expected_t2, rel=1e-6)
    assert adv[1, 0] == pytest.approx(expected_t1, rel=1e-6)
    assert adv[0, 0] == pytest.approx(expected_t0, rel=1e-6)
    assert ret[0, 0] == pytest.approx(expected_t0 + 0.5, rel=1e-6)


def test_gae_stops_at_an_episode_boundary():
    """A done flag on step t must cut the bootstrap from t+1, not from t+2.

    Off by one here leaks the next episode's value backwards and is invisible
    except as a slightly worse policy.
    """
    from doomagent.ppo import Rollout, compute_gae

    buf = Rollout(steps=2, n_envs=1, obs_shape=(1, 2, 2))
    buf.rewards[:, 0] = [1.0, 1.0]
    buf.values[:, 0] = [0.0, 0.0]
    buf.dones[:, 0] = [1.0, 0.0]          # the first step ended the episode
    adv, _ = compute_gae(buf, np.array([10.0], dtype=np.float32), 0.99, 0.95)
    assert adv[0, 0] == pytest.approx(1.0)   # no bootstrap across the boundary


def test_policy_normalises_uint8_observations():
    """The frames cross a process boundary as uint8; the policy divides by 255.

    Feeding it float32 in 0-255 would run without error and score like noise.
    """
    import torch
    from doomagent.model import ActorCritic

    model = ActorCritic((4, 60, 80), 3)
    obs = np.full((1, 4, 60, 80), 255, dtype=np.uint8)
    logits_uint8, _ = model(torch.from_numpy(obs))
    logits_float, _ = model(torch.from_numpy(obs.astype(np.float32) / 255.0))
    assert torch.allclose(logits_uint8, logits_float, atol=1e-5)


def test_width_scales_the_trunk():
    from doomagent.model import ActorCritic

    small = ActorCritic((4, 60, 80), 3, width=1).n_params()
    wide = ActorCritic((4, 60, 80), 3, width=4).n_params()
    assert small == 219_476
    assert wide > 10 * small / 2      # four times the channels is far more weights


# -- 02 crafter ------------------------------------------------------------
def _memory_with(grid: list[str]):
    """Build a WorldMemory from a picture. '.' walkable, '#' stone, '@' player."""
    from crafteragent.memory import WorldMemory

    mem = WorldMemory()
    for y, row in enumerate(grid):
        for x, ch in enumerate(row):
            if ch == "@":
                mem.pos = (x, y)
                mem.tiles[(x, y)] = "grass"
            elif ch == ".":
                mem.tiles[(x, y)] = "grass"
            elif ch == "#":
                mem.tiles[(x, y)] = "stone"
            elif ch == "~":
                mem.tiles[(x, y)] = "water"
    return mem


def test_first_step_walks_around_a_wall():
    from crafteragent.navigate import first_step

    mem = _memory_with([
        "#####",
        "#@#.#",
        "#.#.#",
        "#...#",
        "#####",
    ])
    action, dist = first_step(mem, {(3, 1)}, allow_unknown=False)
    assert action == "move_down"      # the direct route is blocked by stone
    # down the left corridor, along the bottom, and back up the right one
    assert dist == 6


def test_first_step_gives_up_when_nothing_is_reachable():
    from crafteragent.navigate import first_step

    mem = _memory_with([
        "###",
        "#@#",
        "###",
    ])
    action, dist = first_step(mem, {(0, 0)}, allow_unknown=False)
    assert action is None and dist is None


def test_open_sides_counts_unknown_tiles_as_open():
    """An unseen neighbour is not a wall just because nobody has looked at it."""
    from crafteragent.shelter import is_sheltered, open_sides

    mem = _memory_with([
        "###",
        "#@#",
        "#.#",
    ])
    assert len(open_sides(mem)) == 1          # only the floor below is open
    assert not is_sheltered(mem)
    mem.tiles[(1, 2)] = "stone"
    assert is_sheltered(mem)


def test_tracker_infers_crafting_from_the_inventory():
    from crafteragent.achievements import AchievementTracker

    class FakePercept:
        def __init__(self, inv, sleeping=False):
            self.inventory = inv
            self.sleeping = sleeping

    mem = _memory_with(["...", ".@.", "..."])
    tracker = AchievementTracker()
    base = {"wood": 1, "wood_pickaxe": 0, "energy": 9}
    tracker.after_step(mem, FakePercept(base))
    new = tracker.after_step(mem, FakePercept({**base, "wood_pickaxe": 1}))
    assert "make_wood_pickaxe" in new
    assert "make_wood_pickaxe" in tracker.unlocked


def test_tracker_does_not_claim_a_kill_when_the_monster_merely_moved():
    """Undercounting is the chosen failure here: a zombie that steps aside and
    one that dies look alike for a frame, so the tracker demands that no zombie
    be adjacent either."""
    from crafteragent.achievements import AchievementTracker

    class FakePercept:
        def __init__(self, inv, sleeping=False):
            self.inventory = inv
            self.sleeping = sleeping

    mem = _memory_with(["...", ".@.", "..."])
    mem.objects = {(2, 1): "zombie"}
    tracker = AchievementTracker()
    tracker.after_step(mem, FakePercept({"health": 9}))
    mem.facing = (1, 0)
    tracker.remember_intent(mem, "do")
    mem.objects = {(2, 0): "zombie"}          # it moved one tile, still adjacent
    new = tracker.after_step(mem, FakePercept({"health": 9}))
    assert "defeat_zombie" not in new


def test_daylight_follows_the_game_clock():
    from crafteragent.perceive import daylight_at

    assert daylight_at(60) == pytest.approx(1.0, abs=1e-6)   # noon
    assert daylight_at(210) < 0.05                           # midnight
    assert daylight_at(0) > 0.5


# -- 03 brogue -------------------------------------------------------------
def test_cell_grid_covers_the_whole_frame():
    """Brogue stretches 100x34 cells over the window, so the grid must be cut on
    fractional boundaries -- a fixed 9x16 lattice leaves the right-hand columns
    sliced in the wrong place."""
    from brogueagent.screen import GRID_COLS, GRID_ROWS, cell_vectors

    frame = np.random.default_rng(0).integers(0, 255, (567, 908, 3), dtype=np.uint8)
    vectors, norms, colour = cell_vectors(frame)
    assert vectors.shape[:2] == (GRID_ROWS, GRID_COLS)
    assert norms.shape == (GRID_ROWS, GRID_COLS)
    assert colour is None            # colour is opt-in: it costs more than the decode
    unit = np.linalg.norm(vectors.reshape(-1, vectors.shape[-1]), axis=1)
    assert np.allclose(unit[norms.reshape(-1) > 0], 1.0, atol=1e-5)


def test_blank_cells_decode_as_spaces():
    from brogueagent.screen import Decoder

    decoder = Decoder.load()
    frame = np.zeros((567, 908, 3), dtype=np.uint8)
    screen = decoder(frame)
    assert (screen.chars == " ").all()
    assert screen.unknown == 0


def test_agent_will_not_climb_out_of_the_dungeon_on_depth_one():
    """`<` on depth 1 leaves the dungeon and ends the run. Hurt and cornered on
    the first level, the agent must do anything but that."""
    from brogueagent.agent import BrogueAgent, Observation
    from brogueagent.screen import Decoder

    decoder = Decoder.load()
    screen = decoder(np.zeros((567, 908, 3), dtype=np.uint8))
    obs = Observation(screen=screen, health=0.1, player=(5, 5),
                      stairs=[], threats=[(9, 9)], coverage=1.0)
    agent = BrogueAgent()
    agent.reset()
    assert agent.depth == 0
    key, why = agent.act(obs)
    assert key != "<", why

    agent.depth = 2                      # two levels down, retreating is allowed
    key, _ = agent.act(obs)
    assert key == "<"


def test_dive_is_committed_for_several_presses():
    """Brogue's `>` travels first and descends on arrival, so one press is not a
    descent -- interleaving it with `x` walks the agent back."""
    from brogueagent.agent import BrogueAgent, Observation
    from brogueagent.screen import Decoder

    decoder = Decoder.load()
    screen = decoder(np.zeros((567, 908, 3), dtype=np.uint8))
    obs = Observation(screen=screen, health=1.0, player=(5, 5),
                      stairs=[(1, 1)], threats=[], coverage=1.0)
    agent = BrogueAgent()
    agent.reset()
    keys = [agent.act(obs)[0] for _ in range(5)]
    assert keys[0] == ">"
    assert keys.count(">") == 5
