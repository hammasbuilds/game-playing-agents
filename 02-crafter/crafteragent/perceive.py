"""Read a Crafter frame back into tiles, facing and inventory -- from pixels only.

Crafter hands a planner everything on a plate in `info` (a full-world semantic
map, the player's position, the inventory dict). This project does not touch any
of it while playing: the agent gets the same 64x64x3 image a human sees and has
to decode it. `info` is used in exactly one place, `perception_check.py`, to
score this decoder.

Two things make the decode non-trivial, and both are solved here rather than
approximated:

**Compositing.** A tile is a material texture with an optional object drawn over
it with alpha, so matching whole tiles against single textures fails on every
tile that has a cow in it. The atlas is therefore built over
(material, overlay) pairs, composited exactly the way the game composites them.

**Night.** Crafter dims and tints the view on a 300-step day cycle, and below
half daylight it blends in a vignetted field of uniform noise. Rather than
matching through that, the transform is inverted: PIL's colour enhance and the
tint are both affine per pixel, so the whole lighting pipeline is
`observed = M . true + k` with a known 3x3 M and offset k per pixel, and the
decoder solves for `true` before matching. The noise term is random, so it is
removed in expectation (E[U(32,127)] = 79.5), which is why accuracy at midnight
is lower than at noon rather than equal to it -- see the table in the README.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
from crafter import constants, engine

# Objects that can be drawn over a material tile, by the texture name the game
# uses for them (objects.py: Player.texture, Zombie.texture, ...).
OVERLAYS = (
    None,
    "player-left", "player-right", "player-up", "player-down", "player-sleep",
    "cow", "zombie", "skeleton",
    "arrow-left", "arrow-right", "arrow-up", "arrow-down",
    "plant", "plant-young", "plant-ripe",
)
MATERIALS = (None, *constants.materials)  # None renders as 'unknown' (out of bounds)

PLAYER_FACING = {
    "player-left": (-1, 0), "player-right": (+1, 0),
    "player-up": (0, -1), "player-down": (0, +1), "player-sleep": (0, +1),
}

# ImageEnhance.Color(img).enhance(f) is f*rgb + (1-f)*luma, and luma is the
# ITU-R 601 mix PIL uses for 'L'. Crafter calls it with f=0.4, then tints half
# way to (0, 16, 64).
_LUMA = np.array([0.299, 0.587, 0.114])
_ENHANCE = 0.4 * np.eye(3) + 0.6 * np.outer(np.ones(3), _LUMA)
_NIGHT_A = 0.5 * _ENHANCE
_NIGHT_B = 0.5 * np.array([0.0, 16.0, 64.0])
_NOISE_MEAN = 0.5 * (32.0 + 127.0)  # the game draws the noise field from U(32, 127)


@dataclass
class Percept:
    """What one frame says. Coordinates are view-local: (0, 0) is the top-left
    visible tile and `self.center` is the player."""

    material: np.ndarray      # (cols, rows) int8 index into MATERIALS
    overlay: np.ndarray       # (cols, rows) int8 index into OVERLAYS
    inventory: dict[str, int]
    facing: tuple[int, int]
    daylight: float
    sleeping: bool
    confidence: float         # mean cosine similarity of the winning templates

    @property
    def center(self) -> tuple[int, int]:
        return self.material.shape[0] // 2, self.material.shape[1] // 2

    def material_at(self, x: int, y: int) -> str | None:
        if not (0 <= x < self.material.shape[0] and 0 <= y < self.material.shape[1]):
            return None
        return MATERIALS[self.material[x, y]]

    def overlay_at(self, x: int, y: int) -> str | None:
        if not (0 <= x < self.overlay.shape[0] and 0 <= y < self.overlay.shape[1]):
            return None
        return OVERLAYS[self.overlay[x, y]]

    def grid_names(self) -> np.ndarray:
        """(cols, rows) array of 'material' or 'material+overlay' strings, for
        eyeballing a frame in a test."""
        out = np.empty(self.material.shape, dtype=object)
        for x in range(out.shape[0]):
            for y in range(out.shape[1]):
                mat, obj = self.material_at(x, y), self.overlay_at(x, y)
                out[x, y] = f"{mat}+{obj}" if obj else str(mat)
        return out


def daylight_at(step: int) -> float:
    """The game's own day cycle, from the agent's own step counter.

    Copied from Env._update_time. This is not privileged information -- it is the
    clock the agent is holding -- and knowing it exactly is what makes the
    lighting inversion below exact instead of a search over brightnesses.
    """
    progress = (step / 300) % 1 + 0.3
    return float(1 - np.abs(np.cos(np.pi * progress)) ** 3)


class TileAtlas:
    """Unlit (material, overlay) composites at one tile size, as a matrix of
    L2-normalised rows ready for a nearest-neighbour match.

    `unit_rows` matches colour tiles; `luma_rows` is the same atlas in one
    channel, for the sleep screen, which throws colour away (see `_unsleep`).
    """

    def __init__(self, unit: int):
        self.unit = unit
        textures = engine.Textures(constants.root / "assets")
        rows, keys = [], []
        for m_idx, material in enumerate(MATERIALS):
            base = np.zeros((unit, unit, 3), dtype=np.uint8)
            engine._draw(base, (0, 0), textures.get(material, (unit, unit)))
            for o_idx, overlay in enumerate(OVERLAYS):
                tile = base.copy()
                if overlay is not None:
                    engine._draw_alpha(tile, (0, 0), textures.get(overlay, (unit, unit)))
                # The game keeps textures in (x, y) layout and transposes only
                # the finished canvas, so a template composited the game's way
                # is transposed relative to the frame the agent is handed.
                # Matching without this costs ~25% of tiles, mostly as
                # grass-read-as-stone, because the atlas is nearly symmetric.
                rows.append(tile.transpose(1, 0, 2).astype(np.float32))
                keys.append((m_idx, o_idx))
        self.keys = np.array(keys, dtype=np.int16)
        stack = np.stack(rows)                     # (K, unit, unit, 3)
        mat = stack.reshape(len(stack), -1)
        self.unit_rows = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6)
        luma = (stack * _LUMA).sum(axis=-1).reshape(len(stack), -1)
        self.luma_rows = luma / (np.linalg.norm(luma, axis=1, keepdims=True) + 1e-6)


@functools.lru_cache(maxsize=4)
def _atlas(unit: int) -> TileAtlas:
    return TileAtlas(unit)


@functools.lru_cache(maxsize=64)
def _vignette(h: int, w: int, stddev: float = 0.5) -> np.ndarray:
    """Crafter's vignette, which decides how much noise each pixel gets."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    return (1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / (stddev ** 2))).T


@functools.lru_cache(maxsize=256)
def _inverse_lighting(h: int, w: int, daylight: float):
    """Per-pixel (M^-1, k) for `observed = M . true + k`.

    The game computes, for daylight d and per-pixel noise weight m:

        night    = A . ((1-m) true + m noise) + b
        observed = d . true + (1-d) . night

    so M = d I + (1-d)(1-m) A and k = (1-d)[(1-m) b + m (A . noise + b)],
    with the noise replaced by its mean. Both are returned stacked per pixel.
    """
    amount = 2 * (0.5 - daylight) if daylight < 0.5 else 0.0
    m = (amount * _vignette(h, w))[..., None] if amount > 0 else np.zeros((h, w, 1))
    eye = np.eye(3)[None, None]
    mats = daylight * eye + (1 - daylight) * (1 - m)[..., None] * _NIGHT_A[None, None]
    noise_term = _NIGHT_A @ np.full(3, _NOISE_MEAN) + _NIGHT_B
    k = (1 - daylight) * ((1 - m) * _NIGHT_B[None, None] + m * noise_term[None, None])
    return np.linalg.inv(mats), k


def _unlight(view: np.ndarray, daylight: float) -> np.ndarray:
    """Undo the day/night transform on the world part of the frame."""
    if daylight >= 0.999:
        return view.astype(np.float32)
    h, w = view.shape[:2]
    inv, k = _inverse_lighting(h, w, round(float(daylight), 4))
    centred = view.astype(np.float32) - k
    return np.einsum("hwij,hwj->hwi", inv, centred)


def _is_sleeping(view: np.ndarray) -> bool:
    """True if this frame is the sleep screen.

    The sleep overlay is `0.5 * grey + 0.5 * (0, 0, 16)`, so it leaves a
    signature no lit frame has: red equals green at every pixel, and blue sits a
    fixed 8 above them. Testing that is exact where a saturation threshold is a
    guess -- deep night is also desaturated, and a threshold called it sleep.
    """
    v = view.astype(np.float32)
    r, g, b = v[..., 0], v[..., 1], v[..., 2]
    return bool(np.abs(r - g).mean() < 1.5 and np.abs((b - r) - 8.0).mean() < 3.0)


def _unsleep(view: np.ndarray, daylight: float) -> np.ndarray:
    """Recover tile *luma* from the sleep screen.

    While the player sleeps the game desaturates the view completely and tints
    it, which is a rank-1 map: the colour is gone and no inverse exists. In one
    channel it is still exact, because luma commutes with both steps --
    `luma(enhance(x, f)) = luma(x)` for any f, and the tint is affine -- so

        luma(observed) = 0.5 [ d.luma(C) + (1-d)(0.5 luma(C) + luma(b)) ]
                         + 0.5 luma((0, 0, 16))

    which inverts to a scalar. Night noise is handled as its mean, as above.
    Costing colour here is why the sleep bucket in the perception table scores
    below the daylight one -- grass and stone differ in pattern, not just hue.
    """
    v = (view.astype(np.float32) * _LUMA).sum(axis=-1)
    lb = float(_NIGHT_B @ _LUMA)
    gain = 0.5 * (daylight + (1 - daylight) * 0.5)
    offset = 0.5 * ((1 - daylight) * lb) + 0.5 * float(np.array([0.0, 0.0, 16.0]) @ _LUMA)
    if daylight < 0.5:  # the noise field's mean passes through both maps
        h, w = view.shape[:2]
        m = 2 * (0.5 - daylight) * _vignette(h, w)
        # Only the noise's own contribution belongs here; `offset` already
        # carries the tint term lb for the whole pixel.
        noise_luma = 0.25 * (1 - daylight) * m * _NOISE_MEAN
        gain_pix = 0.5 * (daylight + (1 - daylight) * (1 - m) * 0.5)
        return (v - offset - noise_luma) / np.maximum(gain_pix, 1e-3)
    return (v - offset) / max(gain, 1e-3)


class Perceiver:
    """Frame -> Percept. One instance per episode: it keeps the step counter the
    day/night inversion needs."""

    def __init__(self, view: tuple[int, int] = (9, 9), size: tuple[int, int] = (64, 64)):
        self.view = np.array(view)
        self.size = np.array(size)
        self.unit = int((self.size // self.view)[0])
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self.world_grid = (view[0], view[1] - item_rows)   # (cols, rows) of world tiles
        self.item_grid = (view[0], item_rows)
        self.border = int(((self.size - (self.size // self.view) * self.view) // 2)[0])
        self.atlas = _atlas(self.unit)
        self.item_atlas = _atlas(int(0.8 * self.unit))
        self.digit_atlas = _atlas(int(0.6 * self.unit))
        self._digits = self._digit_templates()
        self.step = 0
        self.sleeping = False

    # -- inventory --------------------------------------------------------
    def _digit_templates(self) -> dict[str, np.ndarray]:
        """One template per (item, amount) HUD cell, composited the way the game
        composites it.

        The first version of this matched the count sprite alone against
        digit-on-black templates and read 5, 6 and 7 as 4, because the game draws
        the count *over* the item icon: the patch the decoder was matching had a
        pickaxe behind the digit. Templates are whole cells for that reason, and
        matching the whole cell confirms which item is in the slot as well.
        """
        unit = self.unit
        icon_off, icon_size = int(0.1 * unit), int(0.8 * unit)
        digit_off, digit_size = int(0.4 * unit), int(0.6 * unit)
        textures = engine.Textures(constants.root / "assets")
        out: dict[str, np.ndarray] = {}
        for item in constants.items:
            for amount in range(0, 11):
                cell = np.zeros((unit, unit, 3), dtype=np.uint8)
                if amount >= 1:
                    engine._draw_alpha(cell, (icon_off, icon_off),
                                       textures.get(item, (icon_size, icon_size)))
                    text = str(amount) if amount < 10 else "unknown"
                    engine._draw_alpha(cell, (digit_off, digit_off),
                                       textures.get(text, (digit_size, digit_size)))
                v = cell.transpose(1, 0, 2).reshape(-1).astype(np.float32)
                out[f"{item}:{amount}"] = v / (np.linalg.norm(v) + 1e-6)
        return out

    def _read_inventory(self, canvas: np.ndarray) -> dict[str, int]:
        """The HUD panel is drawn *without* lighting, so this part is exact.

        Counts above 9 render as the 'unknown' sprite -- the game does not show a
        human the real number either, so the decoder reports 10 and the planner
        treats it as 'at least 10'.
        """
        unit = self.unit
        inv: dict[str, int] = {}
        for index, item in enumerate(constants.items):
            cx = index % self.item_grid[0]
            cy = index // self.item_grid[0]
            x0 = self.border + cx * unit
            y0 = self.border + (self.world_grid[1] + cy) * unit
            cell = canvas[y0:y0 + unit, x0:x0 + unit]
            if cell.shape[:2] != (unit, unit):
                inv[item] = 0
                continue
            v = cell.reshape(-1).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm < 1e-3:  # an all-black cell is an empty slot
                inv[item] = 0
                continue
            v = v / norm
            amount, _ = max(
                ((a, float(self._digits[f"{item}:{a}"] @ v)) for a in range(1, 11)),
                key=lambda kv: kv[1])
            inv[item] = amount
        return inv

    # -- world ------------------------------------------------------------
    @staticmethod
    def _match(view: np.ndarray, rows: int, cols: int, atlas_rows: np.ndarray):
        """Nearest-neighbour match every tile of `view` against an atlas.

        `view` is (rows*unit, cols*unit, channels); the atlas rows must carry the
        same channel count. Returns the winning template index per tile, in
        row-major tile order, and the mean cosine similarity of the winners.
        """
        unit = view.shape[0] // rows
        chans = view.shape[2]
        tiles = view.reshape(rows, unit, cols, unit, chans).transpose(0, 2, 1, 3, 4)
        flat = tiles.reshape(rows * cols, -1)
        flat = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6)
        scores = flat @ atlas_rows.T
        best = scores.argmax(axis=1)
        return best, float(scores[np.arange(len(best)), best].mean())

    def __call__(self, frame: np.ndarray, step: int | None = None) -> Percept:
        if step is not None:
            self.step = step
        daylight = daylight_at(self.step)
        cols, rows = self.world_grid
        unit, b = self.unit, self.border
        world = frame[b:b + rows * unit, b:b + cols * unit]

        # Which decode to run is settled by the sleep screen's exact signature,
        # not by comparing the two decodes' scores: they live in a 3-channel and
        # a 1-channel space, and cosine similarity is systematically higher in
        # the smaller one, so "whichever matches better" picked sleep for every
        # frame and dropped tile accuracy to 0.6%.
        self.sleeping = _is_sleeping(world)
        if self.sleeping:
            best, confidence = self._match(_unsleep(world, daylight)[..., None],
                                           rows, cols, self.atlas.luma_rows)
        else:
            best, confidence = self._match(_unlight(world, daylight), rows, cols,
                                           self.atlas.unit_rows)

        keys = self.atlas.keys[best]
        material = keys[:, 0].reshape(rows, cols).T.astype(np.int8)
        overlay = keys[:, 1].reshape(rows, cols).T.astype(np.int8)

        inventory = self._read_inventory(frame)
        cx, cy = cols // 2, rows // 2
        facing = PLAYER_FACING.get(OVERLAYS[overlay[cx, cy]] or "", (0, +1))
        return Percept(material=material, overlay=overlay, inventory=inventory,
                       facing=facing, daylight=daylight, sleeping=self.sleeping,
                       confidence=confidence)

    def tick(self) -> None:
        self.step += 1

    @property
    def daylight_now(self) -> float:
        return daylight_at(self.step)
