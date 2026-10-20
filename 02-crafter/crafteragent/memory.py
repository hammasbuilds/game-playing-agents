"""A map stitched out of 9x7 glimpses.

The agent is never told where it is. Crafter puts the player's position in
`info`, and this project does not read it: the planner has to work out its own
displacement from one frame to the next, which it does by sliding the new view
over the remembered map and keeping the offset that agrees best. Moving into a
wall in Crafter turns the player without moving them, so "I pressed left" is not
evidence that anything moved -- the pixels are.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .perceive import Percept

WALKABLE = frozenset({"grass", "path", "sand"})
DEADLY = frozenset({"lava"})
# The candidate displacements for one step: standing still, or one tile in any
# direction. Diagonals are impossible in Crafter, which keeps the search to five.
CANDIDATES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class WorldMemory:
    """Everything the agent remembers, in a frame whose origin is wherever it
    happened to start."""

    tiles: dict[tuple[int, int], str] = field(default_factory=dict)
    seen_at: dict[tuple[int, int], int] = field(default_factory=dict)
    pos: tuple[int, int] = (0, 0)
    facing: tuple[int, int] = (0, 1)
    step: int = 0
    objects: dict[tuple[int, int], str] = field(default_factory=dict)
    localised: bool = True   # False when the last stitch was ambiguous

    # -- stitching --------------------------------------------------------
    def _agreement(self, percept: Percept, origin: tuple[int, int]) -> tuple[int, int]:
        """How well the view placed at `origin` agrees with what is remembered.

        Returns (matches, overlap). Only tiles already in the map count, and
        object tiles are skipped: a cow walks off and would otherwise look like
        evidence against the correct offset.
        """
        cols, rows = percept.material.shape
        cx, cy = percept.center
        matches = overlap = 0
        for x in range(cols):
            for y in range(rows):
                if percept.overlay_at(x, y) is not None:
                    continue
                mat = percept.material_at(x, y)
                if mat is None:
                    continue
                cell = (origin[0] + x - cx, origin[1] + y - cy)
                known = self.tiles.get(cell)
                if known is None:
                    continue
                overlap += 1
                matches += int(known == mat)
        return matches, overlap

    def update(self, percept: Percept) -> None:
        """Localise against the remembered map, then write the new view in."""
        self.step += 1
        if self.tiles:
            scored = []
            for dx, dy in CANDIDATES:
                origin = (self.pos[0] + dx, self.pos[1] + dy)
                matches, overlap = self._agreement(percept, origin)
                rate = matches / overlap if overlap else 0.0
                scored.append((rate, overlap, origin))
            scored.sort(key=lambda s: (-s[0], -s[1]))
            best, second = scored[0], scored[1]
            # Mining changes tiles under the agent's feet, so a perfect match is
            # not on offer every step; what matters is that the winner is clearly
            # better than the runner-up. When it is not, the map stops being
            # trustworthy and the planner is told so rather than quietly
            # accumulating a shifted copy of the world.
            self.localised = best[0] - second[0] > 0.02 or best[0] > 0.98
            self.pos = best[2]
        self.facing = percept.facing
        cols, rows = percept.material.shape
        cx, cy = percept.center
        self.objects = {}
        for x in range(cols):
            for y in range(rows):
                mat = percept.material_at(x, y)
                if mat is None:
                    continue
                cell = (self.pos[0] + x - cx, self.pos[1] + y - cy)
                self.tiles[cell] = mat
                self.seen_at[cell] = self.step
                obj = percept.overlay_at(x, y)
                if obj is not None and not obj.startswith("player"):
                    self.objects[cell] = obj

    # -- queries ----------------------------------------------------------
    def material(self, cell: tuple[int, int]) -> str | None:
        return self.tiles.get(cell)

    def is_walkable(self, cell: tuple[int, int]) -> bool:
        mat = self.tiles.get(cell)
        return mat in WALKABLE and cell not in self.objects

    def is_unknown(self, cell: tuple[int, int]) -> bool:
        return cell not in self.tiles

    def cells_of(self, materials: frozenset[str] | set[str]) -> list[tuple[int, int]]:
        return [c for c, m in self.tiles.items() if m in materials]

    def objects_of(self, names: frozenset[str] | set[str]) -> list[tuple[int, int]]:
        return [c for c, n in self.objects.items() if n in names]

    def nearby_materials(self, radius: int = 1) -> set[str]:
        """What `make_*` means by nearby: the Chebyshev-radius-1 block around the
        player, which is what World.nearby(pos, 1) returns."""
        out = set()
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                mat = self.tiles.get((self.pos[0] + dx, self.pos[1] + dy))
                if mat:
                    out.add(mat)
        return out

    def frontier(self) -> list[tuple[int, int]]:
        """Known walkable tiles that touch something unseen -- where exploring
        actually buys new information."""
        out = []
        for cell in self.tiles:
            if not self.is_walkable(cell):
                continue
            x, y = cell
            if any(self.is_unknown((x + dx, y + dy)) for dx, dy in CANDIDATES[1:]):
                out.append(cell)
        return out

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for mat in self.tiles.values():
            counts[mat] = counts.get(mat, 0) + 1
        return {"tiles_known": len(self.tiles), "pos": self.pos,
                "materials": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
