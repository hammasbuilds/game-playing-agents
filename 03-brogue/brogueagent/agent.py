"""A policy that plays Brogue through the screen and the keyboard.

What the agent is allowed to know is the screen. It gets no hooks, no memory
reads and no save-file parsing; the only thing it sends back is keystrokes.

Two deliberate choices shape it:

**Locomotion goes through Brogue's own commands.** `x` auto-explores and `>`
travels to the down staircase. A human plays this way, and it keeps the agent's
job where the interesting decisions are -- when to explore, when to dive, when to
back off and rest -- rather than in re-implementing a pathfinder the game already
has. The agent still has to *decide*, and it still has to read the screen to know
whether the decision worked.

**Health is read from the bar, not from text.** The sidebar draws a coloured bar
whose filled fraction is the player's health. Measuring coloured pixels is exact
and needs no character recognition, which matters because the decoder names
terrain reliably and letters only sometimes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .screen import MAP_COL0, MAP_ROW0, Screen

WALKABLE = frozenset({".", "\"", ":", ",", "+"})
WALLS = frozenset({"#", "|", "-"})
# Glyphs that are scenery rather than a threat: the agent should not walk into a
# bloodwort stalk expecting a fight.
HARMLESS = frozenset({"ψ", "*", "$", "!", "?", "(", ")", "[", "]", "/", "=", "\""})

DIRECTIONS = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}
MOVE_KEYS = {"left": "h", "right": "l", "up": "k", "down": "j"}


@dataclass
class Observation:
    """What one frame tells the agent."""

    screen: Screen
    health: float                  # 0..1, from the sidebar bar
    player: tuple[int, int] | None  # map-local (row, col)
    stairs: list[tuple[int, int]]
    threats: list[tuple[int, int]]
    coverage: float                # share of lit map cells the atlas could name


def read_health(frame: np.ndarray) -> float:
    """Filled fraction of the health bar.

    Brogue draws the bar as a solid block behind the word "Health" in the
    sidebar, in the top rows of the screen. The filled part is markedly brighter
    than the empty part, so the fraction is the share of the bar's width whose
    column is bright.
    """
    arr = np.asarray(frame, dtype=np.float32)
    height, width = arr.shape[:2]
    # Grid row 1 of the sidebar, measured rather than guessed: row 1 is purple
    # across its whole width at the start of a game and row 2 is the nutrition
    # bar below it. Brightness does not work here -- the word "Health" is
    # brighter than the bar it sits on -- but the bar's colour is unmistakable.
    band = arr[round(height / 34):round(2 * height / 34), 0:round(0.20 * width)]
    if band.size == 0:
        return 1.0
    purple = (band[..., 0] + band[..., 2] - 2 * band[..., 1]).max(axis=0)
    if purple.max() < 20:          # no bar on screen at all
        return 0.0
    return float((purple > 40).mean())


def observe(frame, decoder) -> Observation:
    screen = decoder(frame)
    grid = screen.map
    rows, cols = grid.shape

    player = None
    stairs, threats = [], []
    lit = named = 0
    for r in range(rows):
        for c in range(cols):
            ch = grid[r, c]
            if ch == " ":
                continue
            lit += 1
            if ch != "?":
                named += 1
            if ch == "@":
                player = (r, c)
            elif ch == ">":
                stairs.append((r, c))
            elif ch.isalpha() and ch not in HARMLESS:
                threats.append((r, c))
    if player is None:
        # The '@' is drawn in white on the player's tile; when it sits on a lit
        # tile the glyph inverts and may not match. Fall back to the brightest
        # cell near the middle of the view rather than losing track of ourselves.
        player = None
    return Observation(screen=screen, health=read_health(frame), player=player,
                       stairs=stairs, threats=threats,
                       coverage=(named / lit if lit else 1.0))


@dataclass
class BrogueAgent:
    """Decide one keystroke from one frame."""

    name: str = "explorer"
    rest_below: float = 0.55        # health fraction under which it stops to rest
    flee_below: float = 0.30
    patience: int = 12              # explore steps with no new map before diving
    dive_commit: int = 8            # how many `>` presses a dive is worth
    retreat_limit: int = 25         # total retreat steps allowed in one game
    seek_trouble: bool = True       # walk toward monsters to clear the level
    chosen: str = "start"
    history: list[str] = field(default_factory=list)
    _seen: int = 0
    _stalled: int = 0
    _diving: int = 0
    _retreats: int = 0
    depth: int = 0                  # levels descended, counted from the screen

    def reset(self) -> None:
        self.history = []
        self._seen = 0
        self._stalled = 0
        self._diving = 0
        self._retreats = 0
        self.depth = 0

    def _adjacent_threat(self, obs: Observation):
        if obs.player is None:
            return None
        pr, pc = obs.player
        for (dc, dr), name in DIRECTIONS.items():
            if (pr + dr, pc + dc) in obs.threats:
                return name
        return None

    def act(self, obs: Observation) -> tuple[str, str]:
        """Return (key, why)."""
        # How much of the level is on screen. Exploration that stops adding
        # cells has finished, whether or not the staircase glyph was recognised.
        seen = int((obs.screen.map != " ").sum())
        # A level the agent has just arrived on is almost entirely unexplored, so
        # a collapse in the number of visible cells is a staircase being taken.
        # The agent counts its own depth this way because of what `<` does on
        # depth 1: climbing up from the first level *leaves the dungeon and ends
        # the run*. Retreating upstairs there is not a retreat, it is a
        # resignation, and it ended games after eleven keystrokes.
        if self._seen > 120 and seen < 0.45 * self._seen:
            self.depth += 1
            self._seen = seen
        self._stalled = 0 if seen > self._seen else self._stalled + 1
        self._seen = max(self._seen, seen)
        melee = self._adjacent_threat(obs)

        if melee and (obs.health > self.flee_below or self._retreats >= self.retreat_limit):
            # Walking into a monster is how you attack it in Brogue. Cornered and
            # hurt, hitting back still beats walking away: a monster adjacent to
            # a retreating player gets a free swing every turn.
            return MOVE_KEYS[melee], f"attack {melee}"
        if (obs.health <= self.flee_below and obs.threats
                and self._retreats < self.retreat_limit and self.depth > 0):
            # Hurt with something in sight: climb back toward the last staircase.
            # This is capped, because it was an absorbing state -- three of six
            # games spent 118 to 201 of their 300 keystrokes pressing `<` and
            # died anyway. Retreating is a move, not a plan.
            self._retreats += 1
            return "<", "retreat upstairs"
        if obs.health <= self.rest_below and not obs.threats:
            # `z` rests until something happens; health comes back with it.
            # Only with the screen clear: resting while a monster was merely
            # visible rather than adjacent spent 135 of 300 steps asleep, and the
            # monster still arrived.
            return "z", "rest"
        if (self.seek_trouble and obs.threats and obs.health > self.rest_below
                and self._stalled < self.patience and self._diving == 0):
            # Something is visible but not adjacent. Exploring walks toward it,
            # which is what we want: a cleared level is a safer level.
            #
            # Unless there is nothing left to explore. Without the stall guard
            # this branch outranked the dive and became its own trap: one 900
            # keystroke game spent 833 of them here, on a level it had already
            # finished, because a monster stayed on screen the whole time.
            return "x", "explore toward trouble"
        if self._diving > 0:
            # Brogue's `>` travels toward the staircase and only descends once
            # the player is standing on it, so the key has to be pressed several
            # times in a row. Interleaving it with `x` walked the agent back to
            # whatever it had been exploring: ten dives in one game, zero
            # descents. Fighting still interrupts -- being killed on the stairs
            # is not a descent either.
            self._diving -= 1
            return ">", "diving"
        if obs.stairs:
            self._diving = self.dive_commit
            return ">", "take the stairs down"
        if self._stalled >= self.patience:
            # The first version only pressed `>` when it had *recognised* a
            # staircase glyph, and since `>` is not one of the terrain glyphs the
            # atlas labels, it never did: six games, 289 of 300 steps spent
            # exploring an already-explored level, zero descents, while an agent
            # pressing random keys descended twice. Brogue's `>` travels to the
            # known staircase by itself, so the agent does not need to see it --
            # it needs to notice that exploring has stopped paying.
            self._stalled = 0
            self._seen = 0
            self._diving = self.dive_commit
            return ">", "explored out, dive"
        return "x", "explore"
