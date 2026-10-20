"""The planner: no gradients, no rollouts, no learned weights.

Every step it decodes the frame, stitches the map, then picks the highest-value
goal whose preconditions hold and takes one step toward it. The ordering is the
whole agent, so it is written as data (`GOALS`) rather than as a chain of ifs.

Three rules earn their place, each because the obvious version failed:

* **Needs come before progress, but only when they bite.** Drinking at 8/9 water
  wastes the walk; drinking at 2/9 avoids losing health to thirst.
* **An achievement is worth pursuing once.** Crafter's score counts the fraction
  of *episodes* in which each achievement was unlocked at least once, so a
  second cow is worth nothing to the score and the ladder stops asking for one.
* **Unknown tiles are passable while planning.** The map starts empty; a search
  restricted to known-walkable tiles never leaves the opening screen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .achievements import AchievementTracker
from .memory import WorldMemory
from .navigate import MOVES, Explorer, approach, retreat
from .perceive import Perceiver
from .shelter import STONE_FOR_SHELTER, ShelterPlan, is_sheltered

TREE = frozenset({"tree"})
STONE = frozenset({"stone"})
COAL = frozenset({"coal"})
IRON = frozenset({"iron"})
DIAMOND = frozenset({"diamond"})
WATER = frozenset({"water"})
GRASS = frozenset({"grass"})
SAND = frozenset({"sand"})
PLACEABLE = frozenset({"grass", "sand", "path"})
COW = frozenset({"cow"})
ZOMBIE = frozenset({"zombie"})
SKELETON = frozenset({"skeleton"})
RIPE_PLANT = frozenset({"plant-ripe"})


@dataclass
class Goal:
    """One thing the agent can want.

    `ready` decides whether it is worth wanting right now, given the inventory,
    what has already been achieved and the map. `act` returns the action to take,
    or None if the goal turns out to be unreachable this step -- in which case the
    planner moves down the list.
    """

    name: str
    ready: callable
    act: callable
    unlocks: str | None = None   # the achievement it is chasing, if any


@dataclass
class PlannerAgent:
    """Pixels in, one of Crafter's 17 actions out."""

    name: str = "planner"
    view: tuple[int, int] = (9, 9)
    size: tuple[int, int] = (64, 64)
    perceiver: Perceiver = field(default_factory=Perceiver)
    memory: WorldMemory = field(default_factory=WorldMemory)
    tracker: AchievementTracker = field(default_factory=AchievementTracker)
    explorer: Explorer = field(default_factory=Explorer)
    shelter_plan: ShelterPlan = field(default_factory=ShelterPlan)
    inventory: dict[str, int] = field(default_factory=dict)
    sleeping: bool = False
    chosen: str = "none"
    goal_counts: dict[str, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.perceiver = Perceiver(self.view, self.size)
        self.memory = WorldMemory()
        self.tracker.reset()
        self.explorer.reset()
        self.shelter_plan.reset()
        self.inventory = {}
        self.sleeping = False
        self.goal_counts = {}

    # -- helpers used by the goal table ------------------------------------
    def have(self, item: str, n: int = 1) -> bool:
        return self.inventory.get(item, 0) >= n

    @property
    def unlocked(self) -> set[str]:
        return self.tracker.unlocked

    def done(self, achievement: str) -> bool:
        return achievement in self.tracker.unlocked

    def _collect(self, materials, allow_unknown: bool = True):
        """Walk to a tile of this material, face it, mine it."""
        cells = self.memory.cells_of(materials)
        action, ready = approach(self.memory, cells, allow_unknown)
        return "do" if ready else action

    def _attack(self, names):
        cells = self.memory.objects_of(names)
        action, ready = approach(self.memory, cells, allow_unknown=False)
        return "do" if ready else action

    def _place(self, what: str, on=PLACEABLE):
        """Place a block on a free neighbouring tile of an allowed material."""
        px, py = self.memory.pos
        for (dx, dy), move in MOVES.items():
            cell = (px + dx, py + dy)
            if self.memory.material(cell) in on and cell not in self.memory.objects:
                if self.memory.facing == (dx, dy):
                    return f"place_{what}"
                return move
        # Nothing suitable adjacent: step onto open ground and try again.
        return self.explorer.next_action(self.memory)

    def _adjacent(self, materials) -> list[tuple[int, int]]:
        """Neighbouring tiles of these materials -- one `do` away."""
        px, py = self.memory.pos
        return [(px + dx, py + dy) for dx, dy in MOVES
                if self.memory.material((px + dx, py + dy)) in materials]

    def _adjacent_objects(self, names) -> list[tuple[int, int]]:
        px, py = self.memory.pos
        return [(px + dx, py + dy) for dx, dy in MOVES
                if self.memory.objects.get((px + dx, py + dy)) in names]

    def _face_and_do(self, cells):
        """Turn toward an adjacent cell if needed, then act on it.

        Crafter's `do` hits the tile the player faces, and a move into a blocked
        tile only turns the player, so "turn, then act" is two steps and never
        moves the agent off the spot.
        """
        if not cells:
            return None
        px, py = self.memory.pos
        for cell in cells:
            delta = (cell[0] - px, cell[1] - py)
            if delta == self.memory.facing:
                return "do"
        delta = (cells[0][0] - px, cells[0][1] - py)
        return MOVES.get(delta)

    # -- shelter -----------------------------------------------------------
    def _sheltered(self) -> bool:
        return is_sheltered(self.memory)

    def _minable(self) -> set[str]:
        """Blocking materials the agent currently has the tool to dig through."""
        out = {"tree"}
        if self.have("wood_pickaxe"):
            out |= {"stone", "coal"}
        if self.have("stone_pickaxe"):
            out |= {"iron"}
        if self.have("iron_pickaxe"):
            out |= {"diamond"}
        return out

    def _break_out(self):
        """Dig out of the box.

        Sealing in is a night tactic and it has to be reversible: the agent
        cannot eat or drink through a wall, and the first version of this that
        could seal but not open starved behind its own stone -- four of eight
        episodes ended at food 0 or drink 0 with the agent safe and motionless.
        Stone is only obtainable with a pickaxe, so anything the agent walled
        itself in with, it can also mine back out.
        """
        px, py = self.memory.pos
        minable = self._minable()
        for (dx, dy), move in MOVES.items():
            cell = (px + dx, py + dy)
            if self.memory.material(cell) in minable:
                return "do" if self.memory.facing == (dx, dy) else move
        return None

    def _shelter(self, threats):
        return self.shelter_plan.next_action(self.memory, self.inventory, threats)

    def _goals(self) -> list[Goal]:
        """The ladder, most urgent first.

        The ordering was rewritten after watching a trace: the agent walked past
        three trees on the way to a cow eleven tiles off because food sat at 3 of
        9 and "hungry" outranked "collect wood", and it finished the episode with
        one log and no table. Hence the shape -- a thin band of genuinely
        critical needs, then everything that is one keypress away, then the tool
        ladder, and only then moderate hunger.
        """
        inv, have, done = self.inventory, self.have, self.done
        mem = self.memory
        near = mem.nearby_materials()
        g: list[Goal] = []

        threats = mem.objects_of(ZOMBIE) + mem.objects_of(SKELETON)
        close = [t for t in threats
                 if abs(t[0] - mem.pos[0]) + abs(t[1] - mem.pos[1]) <= 3]

        # -- 1. survive the next few steps ---------------------------------
        g.append(Goal("fight_adjacent",
                      lambda: have("wood_sword") and inv.get("health", 9) > 4
                      and bool(self._adjacent_objects(ZOMBIE)),
                      lambda: self._face_and_do(self._adjacent_objects(ZOMBIE))))
        # Night, or anything chasing us: get boxed in. This sits above `flee`
        # because fleeing from zombies in Crafter does not work -- see shelter.py.
        g.append(Goal("flee",
                      lambda: bool(close) and (not have("wood_sword")
                                               or inv.get("health", 9) <= 4),
                      lambda: retreat(mem, close)))
        # Morning, or an empty stomach: open the door again.
        g.append(Goal("break_out",
                      lambda: self._sheltered()
                      and (self.perceiver.daylight_now > 0.4
                           or inv.get("food", 9) <= 1 or inv.get("drink", 9) <= 1),
                      lambda: self._break_out()))
        g.append(Goal("drink_critical", lambda: inv.get("drink", 9) <= 2,
                      lambda: self._collect(WATER)))
        g.append(Goal("eat_critical", lambda: inv.get("food", 9) <= 2,
                      lambda: self._attack(COW) or self._attack(RIPE_PLANT)))

        g.append(Goal("shelter",
                      lambda: (self.perceiver.daylight_now < 0.5 or bool(close))
                      and not self._sheltered(),
                      lambda: self._shelter(close)))


        # Nothing to eat or drink in the remembered map is not a reason to carry
        # on mining: two episodes ended at food 0 while the agent quarried stone
        # it had no use for. Go and look instead.
        g.append(Goal("forage_for_water", lambda: inv.get("drink", 9) <= 2
                      and not mem.cells_of(WATER),
                      lambda: self.explorer.next_action(mem)))
        g.append(Goal("forage_for_food", lambda: inv.get("food", 9) <= 2
                      and not (mem.objects_of(COW) or mem.objects_of(RIPE_PLANT)),
                      lambda: self.explorer.next_action(mem)))

        # -- 2. anything one keypress away ---------------------------------
        g.append(Goal("mine_adjacent_tree",
                      lambda: inv.get("wood", 0) < 9 and bool(self._adjacent(TREE)),
                      lambda: self._face_and_do(self._adjacent(TREE))))
        g.append(Goal("mine_adjacent_stone",
                      lambda: have("wood_pickaxe") and inv.get("stone", 0) < 9
                      and bool(self._adjacent(STONE)),
                      lambda: self._face_and_do(self._adjacent(STONE))))
        g.append(Goal("mine_adjacent_coal",
                      lambda: have("wood_pickaxe") and bool(self._adjacent(COAL)),
                      lambda: self._face_and_do(self._adjacent(COAL))))
        g.append(Goal("mine_adjacent_iron",
                      lambda: have("stone_pickaxe") and bool(self._adjacent(IRON)),
                      lambda: self._face_and_do(self._adjacent(IRON))))
        g.append(Goal("mine_adjacent_diamond",
                      lambda: have("iron_pickaxe") and bool(self._adjacent(DIAMOND)),
                      lambda: self._face_and_do(self._adjacent(DIAMOND))))
        g.append(Goal("drink_adjacent",
                      lambda: inv.get("drink", 9) < 9 and bool(self._adjacent(WATER)),
                      lambda: self._face_and_do(self._adjacent(WATER))))
        g.append(Goal("eat_adjacent_cow",
                      lambda: inv.get("food", 9) < 7 and bool(self._adjacent_objects(COW)),
                      lambda: self._face_and_do(self._adjacent_objects(COW))))
        g.append(Goal("eat_adjacent_plant",
                      lambda: inv.get("food", 9) < 9
                      and bool(self._adjacent_objects(RIPE_PLANT)),
                      lambda: self._face_and_do(self._adjacent_objects(RIPE_PLANT))))
        g.append(Goal("sapling_adjacent",
                      lambda: not have("sapling") and not done("collect_sapling")
                      and bool(self._adjacent(GRASS)),
                      lambda: self._face_and_do(self._adjacent(GRASS))))

        # -- 2b. before dark, fetch the blocks the night will need ---------
        # Dusk is around daylight 0.5 and the first zombies arrive before 0.35,
        # so the stone has to be in the inventory by then: an agent that goes
        # looking for it at midnight is doing so in a crowd.
        g.append(Goal("stock_stone_for_night",
                      lambda: self.perceiver.daylight_now < 0.75
                      and have("wood_pickaxe")
                      and inv.get("stone", 0) < STONE_FOR_SHELTER
                      and not self._sheltered(),
                      lambda: self._collect(STONE)))

        # -- 3. the tool ladder --------------------------------------------
        g.append(Goal("collect_wood", lambda: inv.get("wood", 0) < 6,
                      lambda: self._collect(TREE)))
        g.append(Goal("place_table",
                      lambda: have("wood", 2) and "table" not in near
                      and not (have("wood_pickaxe") and have("wood_sword")),
                      lambda: self._place("table")))
        g.append(Goal("make_wood_pickaxe",
                      lambda: have("wood") and "table" in near and not have("wood_pickaxe"),
                      lambda: "make_wood_pickaxe", unlocks="make_wood_pickaxe"))
        g.append(Goal("make_wood_sword",
                      lambda: have("wood") and "table" in near and not have("wood_sword"),
                      lambda: "make_wood_sword", unlocks="make_wood_sword"))
        g.append(Goal("collect_stone",
                      lambda: have("wood_pickaxe") and inv.get("stone", 0) < 6,
                      lambda: self._collect(STONE)))
        g.append(Goal("make_stone_pickaxe",
                      lambda: have("wood") and have("stone") and "table" in near
                      and not have("stone_pickaxe"),
                      lambda: "make_stone_pickaxe", unlocks="make_stone_pickaxe"))
        g.append(Goal("make_stone_sword",
                      lambda: have("wood") and have("stone") and "table" in near
                      and not have("stone_sword"),
                      lambda: "make_stone_sword", unlocks="make_stone_sword"))
        g.append(Goal("place_stone", lambda: have("stone") and not done("place_stone"),
                      lambda: self._place("stone"), unlocks="place_stone"))
        g.append(Goal("collect_coal", lambda: have("wood_pickaxe") and not have("coal"),
                      lambda: self._collect(COAL)))
        g.append(Goal("place_furnace",
                      lambda: have("stone", 4) and "furnace" not in near
                      and not have("iron_pickaxe"),
                      lambda: self._place("furnace")))
        g.append(Goal("collect_iron", lambda: have("stone_pickaxe") and not have("iron"),
                      lambda: self._collect(IRON)))
        g.append(Goal("make_iron_pickaxe",
                      lambda: have("wood") and have("coal") and have("iron")
                      and {"table", "furnace"} <= near and not have("iron_pickaxe"),
                      lambda: "make_iron_pickaxe", unlocks="make_iron_pickaxe"))
        g.append(Goal("make_iron_sword",
                      lambda: have("wood") and have("coal") and have("iron")
                      and {"table", "furnace"} <= near and not have("iron_sword"),
                      lambda: "make_iron_sword", unlocks="make_iron_sword"))
        g.append(Goal("collect_diamond",
                      lambda: have("iron_pickaxe") and not have("diamond"),
                      lambda: self._collect(DIAMOND)))

        # -- 4. moderate needs ---------------------------------------------
        g.append(Goal("drink", lambda: inv.get("drink", 9) <= 5,
                      lambda: self._collect(WATER)))
        g.append(Goal("eat_cow", lambda: inv.get("food", 9) <= 5,
                      lambda: self._attack(COW)))

        # -- 5. the achievements nothing else covers -----------------------
        g.append(Goal("collect_drink", lambda: not done("collect_drink"),
                      lambda: self._collect(WATER), unlocks="collect_drink"))
        g.append(Goal("collect_sapling",
                      lambda: not have("sapling") and not done("collect_sapling"),
                      lambda: self._collect(GRASS), unlocks="collect_sapling"))
        g.append(Goal("place_plant", lambda: have("sapling") and not done("place_plant"),
                      lambda: self._place("plant", on=GRASS), unlocks="place_plant"))
        g.append(Goal("eat_plant",
                      lambda: not done("eat_plant") and bool(mem.objects_of(RIPE_PLANT)),
                      lambda: self._attack(RIPE_PLANT), unlocks="eat_plant"))
        g.append(Goal("eat_cow_once", lambda: not done("eat_cow") and have("wood_sword"),
                      lambda: self._attack(COW), unlocks="eat_cow"))
        g.append(Goal("fight_skeleton",
                      lambda: have("stone_sword") and inv.get("health", 9) > 5
                      and bool(mem.objects_of(SKELETON)),
                      lambda: self._attack(SKELETON), unlocks="defeat_skeleton"))
        g.append(Goal("hunt_zombie",
                      lambda: have("stone_sword") and inv.get("health", 9) > 6
                      and not done("defeat_zombie") and bool(mem.objects_of(ZOMBIE)),
                      lambda: self._attack(ZOMBIE), unlocks="defeat_zombie"))

        # -- 6. sleep, but not in the open ---------------------------------
        # Sleeping in the open cost the agent the game three nights running: it
        # lay down on grass with five zombies on screen and woke at 1 health, or
        # not at all. Zombies spawn on grass, so the rule is to wall in first.
        g.append(Goal("sleep",
                      lambda: inv.get("energy", 9) <= 4 and self._sheltered(),
                      lambda: "sleep", unlocks="wake_up"))
        # Waiting out the night behind a wall beats wandering in it even when
        # the agent is not tired enough to sleep.
        g.append(Goal("hold_shelter",
                      lambda: self.perceiver.daylight_now < 0.25
                      and self._sheltered() and bool(close),
                      lambda: "noop"))
        return g

    # -- the policy --------------------------------------------------------
    def act(self, frame) -> int:
        from crafter import constants

        percept = self.perceiver(frame)
        self.perceiver.tick()
        self.inventory = percept.inventory
        self.sleeping = percept.sleeping
        if self.sleeping:
            # The screen is unreadable and the only useful thing to do is keep
            # sleeping until the game wakes us, which is what scores wake_up.
            self.chosen = "sleeping"
            return constants.actions.index("sleep")
        self.memory.update(percept)
        self.tracker.after_step(self.memory, percept)

        action = None
        for goal in self._goals():
            if goal.unlocks and goal.unlocks in self.unlocked:
                continue
            try:
                if not goal.ready():
                    continue
                candidate = goal.act()
            except (KeyError, ValueError):
                continue
            if candidate:
                action, self.chosen = candidate, goal.name
                break
        if action is None:
            action = self.explorer.next_action(self.memory)
            action, self.chosen = (action or "noop"), "explore"
        self.goal_counts[self.chosen] = self.goal_counts.get(self.chosen, 0) + 1
        self.tracker.remember_intent(self.memory, action)
        return constants.actions.index(action)
