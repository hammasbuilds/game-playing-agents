"""Work out what the agent has achieved, from what the agent can see.

Crafter reports achievements in `info`, and the planner used to read them. It
does not any more, because every one of the 22 is inferable from the HUD and the
tiles: a pickaxe appearing in the inventory *is* make_wood_pickaxe, a table
appearing on the tile you just placed on *is* place_table. The ladder in
`agent.py` skips goals it believes are already unlocked, so this tracker's
mistakes cost real score -- which is why `evaluate.py` scores the tracker against
the env's own record and the README prints the number.

The only inference that is not exact is defeating a creature: the agent sees the
tile it attacked, and a zombie that dies and a zombie that steps aside look the
same for one frame. The tracker demands that the creature be gone *and* that no
creature of that kind is adjacent, which is why it undercounts rather than
overcounts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import CANDIDATES

COLLECTABLE = ("wood", "stone", "coal", "iron", "diamond", "sapling", "drink")
TOOLS = ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
         "wood_sword", "stone_sword", "iron_sword")


@dataclass
class AchievementTracker:
    unlocked: set[str] = field(default_factory=set)
    _inventory: dict[str, int] = field(default_factory=dict)
    _facing_cell: tuple[int, int] | None = None
    _facing_was: tuple[str | None, str | None] = (None, None)
    _action: str | None = None
    _sleeping: bool = False

    def reset(self) -> None:
        self.unlocked = set()
        self._inventory = {}
        self._facing_cell = None
        self._facing_was = (None, None)
        self._action = None
        self._sleeping = False

    def after_step(self, memory, percept) -> set[str]:
        """Compare this frame with the last one and bank what changed."""
        new: set[str] = set()
        inv, prev = percept.inventory, self._inventory

        if prev:
            for item in COLLECTABLE:
                if inv.get(item, 0) > prev.get(item, 0):
                    new.add(f"collect_{item}")
            for tool in TOOLS:
                if inv.get(tool, 0) > prev.get(tool, 0):
                    new.add(f"make_{tool}")

            gained_food = inv.get("food", 0) > prev.get("food", 0)
            was_mat, was_obj = self._facing_was
            if gained_food and self._action == "do":
                if was_obj == "cow":
                    new.add("eat_cow")
                elif was_obj == "plant-ripe":
                    new.add("eat_plant")

            if self._action and self._action.startswith("place_") and self._facing_cell:
                what = self._action[len("place_"):]
                now_mat = memory.material(self._facing_cell)
                now_obj = memory.objects.get(self._facing_cell)
                placed = now_mat == what or (what == "plant" and
                                             (now_obj or "").startswith("plant"))
                if placed and was_mat != what:
                    new.add(f"place_{what}")

            if self._action == "do" and was_obj in ("zombie", "skeleton") and self._facing_cell:
                still_there = memory.objects.get(self._facing_cell) == was_obj
                x, y = self._facing_cell
                nearby = any(memory.objects.get((x + dx, y + dy)) == was_obj
                             for dx, dy in CANDIDATES)
                if not still_there and not nearby:
                    new.add(f"defeat_{was_obj}")

            # Energy back at full after a night's sleep is exactly the condition
            # the game itself uses to grant wake_up.
            if self._sleeping and not percept.sleeping and inv.get("energy", 0) >= 9:
                new.add("wake_up")

        self.unlocked |= new
        self._inventory = dict(inv)
        self._sleeping = percept.sleeping
        return new

    def remember_intent(self, memory, action: str) -> None:
        """Called with the action about to be sent, so the next frame knows what
        tile to look at."""
        self._action = action
        fx, fy = memory.facing
        cell = (memory.pos[0] + fx, memory.pos[1] + fy)
        self._facing_cell = cell
        self._facing_was = (memory.material(cell), memory.objects.get(cell))
