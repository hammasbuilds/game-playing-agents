"""Breadth-first search over the remembered map, and the turn-then-act rule.

Crafter's `do` acts on the tile the player faces, and a move into a blocked tile
only changes the facing. Both facts are load-bearing here: to mine a tree the
agent walks to a tile beside it and then issues the *move* action pointing at it,
which turns it without moving it, and only then `do`.
"""
from __future__ import annotations

from collections import deque

from .memory import CANDIDATES, WorldMemory

MOVES = {(-1, 0): "move_left", (1, 0): "move_right",
         (0, -1): "move_up", (0, 1): "move_down"}


def _neighbours(cell):
    x, y = cell
    for dx, dy in CANDIDATES[1:]:
        yield (x + dx, y + dy), (dx, dy)


def first_step(memory: WorldMemory, goals: set[tuple[int, int]],
               allow_unknown: bool = False, limit: int = 4000):
    """The first move on a shortest path from the player to any goal cell.

    Returns (action, distance) or (None, None) when no goal is reachable. With
    `allow_unknown` the search may step onto tiles never seen, which is how
    exploration gets anywhere: the map starts empty and a strictly known-walkable
    search would refuse to leave the first screen.
    """
    start = memory.pos
    if start in goals:
        return "noop", 0
    seen = {start}
    queue = deque([(start, None, 0)])
    expanded = 0
    while queue:
        cell, first, dist = queue.popleft()
        expanded += 1
        if expanded > limit:
            break
        for nxt, delta in _neighbours(cell):
            if nxt in seen:
                continue
            seen.add(nxt)
            step = first if first is not None else MOVES[delta]
            if nxt in goals:
                return step, dist + 1
            passable = memory.is_walkable(nxt) or (allow_unknown and memory.is_unknown(nxt))
            if passable:
                queue.append((nxt, step, dist + 1))
    return None, None


def adjacent_cells(cells) -> set[tuple[int, int]]:
    """The tiles from which `cells` can be faced."""
    out = set()
    for x, y in cells:
        for dx, dy in CANDIDATES[1:]:
            out.add((x + dx, y + dy))
    return out


def approach(memory: WorldMemory, targets, allow_unknown: bool = True):
    """Walk to a tile beside one of `targets`, turn to face it, and report that
    the agent is in position.

    Returns (action, ready): `ready` is True when the agent already stands beside
    a target and is facing it, which is the moment `do` / `place_*` will land.
    """
    targets = {t for t in targets}
    if not targets:
        return None, False
    px, py = memory.pos
    for (dx, dy), action in MOVES.items():
        cell = (px + dx, py + dy)
        if cell in targets:
            if memory.facing == (dx, dy):
                return action, True        # already facing it; caller sends `do`
            return action, False           # one move turns us to face it
    stand = adjacent_cells(targets) - {memory.pos}
    stand = {c for c in stand if memory.is_walkable(c) or memory.is_unknown(c)}
    action, _ = first_step(memory, stand, allow_unknown=allow_unknown)
    return action, False


def _unknown_around(memory: WorldMemory, cell, radius: int = 3) -> int:
    x, y = cell
    return sum(memory.is_unknown((x + dx, y + dy))
               for dx in range(-radius, radius + 1)
               for dy in range(-radius, radius + 1))


class Explorer:
    """Frontier exploration that commits to a target.

    Walking to the *nearest* frontier tile is the obvious thing and it thrashes:
    the nearest unseen edge is usually the tile just behind you, so the agent
    steps back and forth and spends 78% of an episode exploring 20 tiles -- which
    is exactly what the first version of this planner did. Picking the frontier
    with the most unseen neighbours per step of walking, and keeping it until it
    is reached or stale, turns that into a direction.
    """

    def __init__(self, patience: int = 40):
        self.target: tuple[int, int] | None = None
        self.age = 0
        self.patience = patience

    def reset(self) -> None:
        self.target, self.age = None, 0

    def _pick(self, memory: WorldMemory):
        best, best_value = None, 0.0
        for cell in memory.frontier():
            gain = _unknown_around(memory, cell)
            if gain <= 0:
                continue
            dx = abs(cell[0] - memory.pos[0]) + abs(cell[1] - memory.pos[1])
            value = gain / (1 + dx)
            if value > best_value:
                best, best_value = cell, value
        return best

    def next_action(self, memory: WorldMemory):
        self.age += 1
        stale = (self.target is None or self.age > self.patience
                 or memory.pos == self.target
                 or _unknown_around(memory, self.target) == 0)
        if stale:
            self.target = self._pick(memory)
            self.age = 0
        if self.target is not None:
            action, _ = first_step(memory, {self.target}, allow_unknown=True)
            if action:
                return action
            self.target = None  # unreachable; fall through to the blind push
        # Nothing known borders the unknown (the opening frame, or a dead end):
        # push outward through unseen tiles instead of standing still.
        px, py = memory.pos
        for radius in (4, 8, 16):
            unknown = {(px + dx, py + dy)
                       for dx in range(-radius, radius + 1)
                       for dy in range(-radius, radius + 1)
                       if memory.is_unknown((px + dx, py + dy))}
            if unknown:
                action, _ = first_step(memory, unknown, allow_unknown=True)
                if action:
                    return action
        return None


def retreat(memory: WorldMemory, threats) -> str | None:
    """Step to whichever neighbouring tile is furthest from the nearest threat.

    Used when something dangerous is close and the agent has nothing to fight it
    with. Standing and trading hits with a zombie while unarmed loses: it deals 2
    a hit and has 5 health against the agent's 1 damage.
    """
    if not threats:
        return None
    px, py = memory.pos
    options = []
    for (dx, dy), action in MOVES.items():
        cell = (px + dx, py + dy)
        if not memory.is_walkable(cell):
            continue
        dist = min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in threats)
        options.append((dist, action))
    if not options:
        return None
    return max(options)[1]
