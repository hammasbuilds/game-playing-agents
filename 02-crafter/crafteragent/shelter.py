"""Surviving the night by building a box and getting inside it.

This is the single change that mattered most to the score. Crafter spawns zombies
on grass in the dark, several at a time, and they do 2 damage a hit (7 if you are
asleep). Running is not an escape -- they move as fast as the player and spawn
ahead of it -- so the trace of an early version reads: forty steps of `flee`, ten
zombies on screen, dead at step 223 with nine unused stone in the inventory.

A sealed tile is safe: zombies cannot path through stone, and closing the last
open side costs one action. The three ways to get there, cheapest first:

1. **Seal where you stand** -- only if there is a block for every open side.
   Four blocks and eight actions on open grass, one block in a dead end.
2. **Walk into a pocket** -- a tile that already has three walls.
3. **Mine one in** -- a pickaxe turns a rock face into a three-walled pocket.

The second version of this file chose between them afresh every step, and the
agent spent the night oscillating: `shelter` moved it one step toward a pocket,
`flee` moved it back, and it died in the open with the plan half executed. So a
plan is now *committed* -- picked once, followed until it is reached or proven
impossible -- and `prepare_for_night` goes and fetches the stone it will need
while it is still light enough to walk around safely.
"""
from __future__ import annotations

from collections import deque

from .memory import CANDIDATES, WorldMemory
from .navigate import MOVES, first_step

SEALED = frozenset({"stone", "table", "furnace", "coal", "iron", "diamond", "tree"})
STONE_FOR_SHELTER = 4  # one per side, which is what an open-grass box costs


def open_sides(memory: WorldMemory, cell: tuple[int, int] | None = None):
    """Directions a zombie could step in from.

    Unknown tiles count as open: an unseen neighbour is not a wall just because
    nobody has looked at it.
    """
    cx, cy = cell if cell is not None else memory.pos
    out = []
    for dx, dy in CANDIDATES[1:]:
        nxt = (cx + dx, cy + dy)
        if memory.is_walkable(nxt) or memory.is_unknown(nxt):
            out.append((dx, dy))
    return out


def is_sheltered(memory: WorldMemory) -> bool:
    return not open_sides(memory)


def find_pocket(memory: WorldMemory, radius: int = 10, max_open: int = 1):
    """The closest standable tile with at most `max_open` ways in.

    Breadth-first so that "closest" is counted in steps, not in straight lines:
    a tile on the far side of a lake is not close.
    """
    start = memory.pos
    seen = {start}
    queue = deque([(start, 0)])
    best, best_key = None, None
    while queue:
        cell, dist = queue.popleft()
        n_open = len(open_sides(memory, cell))
        key = (n_open, dist)
        if n_open <= max_open and (best_key is None or key < best_key):
            best, best_key = cell, key
        if dist >= radius:
            continue
        for dx, dy in CANDIDATES[1:]:
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen:
                continue
            seen.add(nxt)
            if memory.is_walkable(nxt):
                queue.append((nxt, dist + 1))
    return best


def diggable_face(memory: WorldMemory):
    """An adjacent stone tile whose far side is mostly walled, so mining it
    opens a pocket rather than a hole in the edge of a boulder."""
    px, py = memory.pos
    for (dx, dy), move in MOVES.items():
        cell = (px + dx, py + dy)
        if memory.material(cell) != "stone":
            continue
        walls = sum(1 for ddx, ddy in CANDIDATES[1:]
                    if memory.material((cell[0] + ddx, cell[1] + ddy)) in SEALED)
        if walls >= 3:
            return cell, move
    return None, None


def sealable_sides(memory: WorldMemory):
    """Open sides that a block can actually go on.

    `place_stone` is refused on a tile holding an object, so a side with a zombie
    standing in it is not a side the agent can close this step.
    """
    px, py = memory.pos
    return [s for s in open_sides(memory)
            if (px + s[0], py + s[1]) not in memory.objects]


def pick_side(memory: WorldMemory, threats):
    """Which side to close next: the one most exposed to the nearest threat."""
    sides = sealable_sides(memory)
    if not sides:
        return None
    px, py = memory.pos

    def exposure(side):
        cell = (px + side[0], py + side[1])
        if not threats:
            return 0
        return -min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in threats)

    return max(sides, key=exposure)


class ShelterPlan:
    """A committed plan to get safe, so the agent stops changing its mind."""

    def __init__(self):
        self.target: tuple[int, int] | None = None
        self.mode: str = "idle"
        self.age = 0
        self.side: tuple[int, int] | None = None

    def reset(self) -> None:
        self.target, self.mode, self.age = None, "idle", 0
        self.side = None

    def _seal_step(self, memory: WorldMemory, threats) -> str | None:
        """Turn to one open side and block it, keeping the same side until it is
        actually blocked.

        Re-choosing the most exposed side every step is what the previous version
        did, and because the zombies move, the "most exposed" side changed from
        step to step: the agent turned, turned again, and never once placed a
        block. Eight episodes ended with four to nine stone in the inventory and
        `steps_sealed=0`.
        """
        px, py = memory.pos
        if self.side is not None:
            cell = (px + self.side[0], py + self.side[1])
            blocked = not (memory.is_walkable(cell) or memory.is_unknown(cell))
            if blocked or cell in memory.objects:
                self.side = None
        if self.side is None:
            self.side = pick_side(memory, threats)
        if self.side is None:
            return None
        return "place_stone" if memory.facing == self.side else MOVES[self.side]

    @property
    def active(self) -> bool:
        return self.mode != "idle"

    def _choose(self, memory: WorldMemory, inventory: dict):
        stone = inventory.get("stone", 0)
        sides = sealable_sides(memory)
        if sides and stone >= len(sides):
            return "seal", memory.pos
        pocket = find_pocket(memory, max_open=1)
        if pocket is not None and stone >= 1:
            return ("seal", memory.pos) if pocket == memory.pos else ("goto", pocket)
        # A nook with two ways in needs two blocks rather than four, and is far
        # more common than a dead end. Taking it before walking off to quarry is
        # worth a lot after dark: the episodes that ended "killed outright" were
        # mostly spent mining in the open with two or three stone already in hand.
        loose = find_pocket(memory, max_open=2)
        if loose is not None and stone >= 2:
            return ("seal", memory.pos) if loose == memory.pos else ("goto", loose)
        if inventory.get("wood_pickaxe", 0) > 0:
            cell, _ = diggable_face(memory)
            if cell is not None:
                return "dig", cell
            if memory.cells_of({"stone"}):
                return "mine", None
        return "idle", None

    def next_action(self, memory: WorldMemory, inventory: dict, threats) -> str | None:
        if is_sheltered(memory):
            self.reset()
            return None
        self.age += 1
        # Sealing where we stand, when there are blocks for every way in, beats
        # every other plan -- so check it every step rather than only when the
        # committed plan goes stale. Without this override the agent kept mining
        # for thirty steps with seven stone already in the bag, taking hits the
        # whole time, because `mine` had been the right answer once.
        sides = sealable_sides(memory)
        if sides and inventory.get("stone", 0) >= len(sides) and self.mode != "seal":
            self.mode, self.target, self.age = "seal", memory.pos, 0
        stale = (not self.active or self.age > 30
                 or (self.mode == "goto" and memory.pos == self.target))
        if stale:
            self.mode, self.target = self._choose(memory, inventory)
            self.age = 0

        if self.mode == "seal":
            action = self._seal_step(memory, threats)
            if action:
                return action
            self.reset()
        elif self.mode == "goto" and self.target is not None:
            if memory.pos == self.target:
                self.mode = "seal"
                return self._seal_step(memory, threats)
            action, _ = first_step(memory, {self.target}, allow_unknown=False)
            if action:
                return action
            self.reset()
        elif self.mode == "dig" and self.target is not None:
            px, py = memory.pos
            delta = (self.target[0] - px, self.target[1] - py)
            if delta in MOVES:
                if memory.facing != delta:
                    return MOVES[delta]
                if memory.material(self.target) == "stone":
                    return "do"
                # The tile is mined out. Step into it: a hole in a rock face has
                # three stone walls already, so the agent is one block from safe
                # instead of the four an open-grass box costs. Mining the tile
                # and then walking away -- which is what this did before -- left
                # it standing outside its own shelter with the blocks in hand.
                return MOVES[delta]
            self.reset()
        elif self.mode == "mine":
            stones = memory.cells_of({"stone"})
            if not stones:
                self.reset()
            else:
                px, py = memory.pos
                for (dx, dy), move in MOVES.items():
                    if memory.material((px + dx, py + dy)) == "stone":
                        return "do" if memory.facing == (dx, dy) else move
                stand = {(x + dx, y + dy) for x, y in stones for dx, dy in CANDIDATES[1:]}
                stand = {c for c in stand if memory.is_walkable(c)}
                action, _ = first_step(memory, stand, allow_unknown=False)
                if action:
                    return action
                self.reset()
        return None
