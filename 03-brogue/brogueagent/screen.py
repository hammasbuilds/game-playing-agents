"""Turn a screenshot of Brogue into a grid of characters.

Brogue draws a 100x34 character grid, and *stretches it over the whole client
area*: at `--size 4` the window is 908x567, so a cell is 9.08 x 16.68 pixels and
the cell boundaries drift by eight pixels between the left edge of the screen and
the right. Three earlier versions of this file assumed an integer lattice and
each one failed in a way worth recording, because the symptom never pointed at
the cause:

* a fixed 9x16 grid finds **1,924 distinct glyph shapes** in a font that has
  about ninety -- every cell is sliced in a slightly different place;
* aligning that grid by looking for quiet seams wanders by up to 18 pixels from
  frame to frame, because most of a Brogue screen is black and nearly every
  offset looks quiet;
* aligning it by *repetition* -- the right grid draws the same floor glyph
  hundreds of times -- is stable in x and still ambiguous in y, because the font
  does not fill its cell.

Cutting on the rounded fractional boundaries and resampling each cell into one
canonical box fixes all three at once: 174 prototypes, 90 of which cover 96% of
the cells on screen.

Matching is colour-blind and polarity-blind. Brogue paints the same letter in a
dozen colours, and on a lit tile it inverts to dark-on-light; only the shape is
stable. Colour is kept separately, because it carries its own information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

GRID_COLS, GRID_ROWS = 100, 34
BOX_W, BOX_H = 12, 20          # the canonical cell, larger than any on-screen cell
MIN_INK = 30.0                 # L2 norm below this and the cell is blank

# Brogue's layout: sidebar down the left, messages along the top, a button bar at
# the bottom, the dungeon in between.
SIDEBAR_COLS = 20
MESSAGE_ROWS = 3
FOOTER_ROWS = 2
MAP_COL0, MAP_ROW0 = SIDEBAR_COLS, MESSAGE_ROWS
MAP_COLS = GRID_COLS - SIDEBAR_COLS
MAP_ROWS = GRID_ROWS - MESSAGE_ROWS - FOOTER_ROWS

ATLAS_PATH = Path(__file__).with_name("atlas.npz")
LABELS_PATH = Path(__file__).with_name("labels.json")


@dataclass
class Screen:
    """One decoded frame."""

    chars: np.ndarray          # (rows, cols) unicode
    colour: np.ndarray         # (rows, cols, 3) mean ink colour
    confidence: np.ndarray     # (rows, cols) cosine similarity of the match
    unknown: int               # cells whose shape is not in the atlas

    def line(self, row: int, col0: int = 0, col1: int | None = None) -> str:
        return "".join(self.chars[row, col0:col1]).rstrip()

    def text(self, col0: int = 0, col1: int | None = None) -> list[str]:
        return [self.line(r, col0, col1) for r in range(self.chars.shape[0])]

    @property
    def map(self) -> np.ndarray:
        return self.chars[MAP_ROW0:MAP_ROW0 + MAP_ROWS, MAP_COL0:MAP_COL0 + MAP_COLS]

    def find(self, char: str) -> list[tuple[int, int]]:
        """Map-local (row, col) of every cell showing this character."""
        rows, cols = np.nonzero(self.map == char)
        return list(zip(rows.tolist(), cols.tolist()))


def cell_vectors(frame: np.ndarray, want_colour: bool = False):
    """Every cell as a contrast- and polarity-normalised vector.

    Returns (vectors, ink norms, mean ink colour or None). Colour is off by
    default because pooling it per cell costs more than the rest of the decode
    put together -- 200 ms of a 230 ms frame -- and the policy reads hue from the
    health bar directly rather than from the grid.
    """
    img = Image.fromarray(np.asarray(frame)) if not isinstance(frame, Image.Image) else frame
    width, height = img.size
    grey = img.convert("L")
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)

    # Resampling the whole frame once to an exact (cols*BOX_W, rows*BOX_H) image
    # puts every cell on an integer boundary, so the cells are plain array slices
    # afterwards. Cropping and resizing cell by cell gives the same answer and
    # costs about half a second a frame in Python, which is most of the time a
    # game takes to play.
    scaled = np.asarray(grey.resize((GRID_COLS * BOX_W, GRID_ROWS * BOX_H), Image.BILINEAR),
                        dtype=np.float32)
    raw = (scaled.reshape(GRID_ROWS, BOX_H, GRID_COLS, BOX_W)
                 .transpose(0, 2, 1, 3)
                 .reshape(GRID_ROWS, GRID_COLS, BOX_H * BOX_W))

    # Colour is taken from the unscaled frame, pooled per cell on the true
    # fractional boundaries, because it is read for hue and not for shape.
    if not want_colour:
        flat = raw.reshape(-1, BOX_H * BOX_W)
        vectors = flat - np.median(flat, axis=1, keepdims=True)
        vectors[vectors.sum(axis=1) < 0] *= -1
        norms = np.linalg.norm(vectors, axis=1)
        safe = np.where(norms > 1e-6, norms, 1.0)[:, None]
        return (vectors / safe).reshape(GRID_ROWS, GRID_COLS, -1),             norms.reshape(GRID_ROWS, GRID_COLS), None

    ys = np.minimum((np.arange(GRID_ROWS) * height / GRID_ROWS).round().astype(int), height - 1)
    xs = np.minimum((np.arange(GRID_COLS) * width / GRID_COLS).round().astype(int), width - 1)
    yh = max(1, height // GRID_ROWS)
    xw = max(1, width // GRID_COLS)
    colour = np.zeros((GRID_ROWS, GRID_COLS, 3), dtype=np.uint8)
    for r, y in enumerate(ys):
        band = rgb[y:y + yh]
        for c, x in enumerate(xs):
            tile = band[:, x:x + xw].reshape(-1, 3)
            if len(tile):
                lum = tile.mean(axis=1)
                ink = tile[np.abs(lum - np.median(lum)) > 24]
                colour[r, c] = ink.mean(axis=0) if len(ink) else tile.mean(axis=0)

    flat = raw.reshape(-1, BOX_H * BOX_W)
    vectors = flat - np.median(flat, axis=1, keepdims=True)
    # Ink is the minority of a cell's pixels, so forcing it positive merges a
    # glyph with the same glyph drawn dark-on-light on a lit tile.
    vectors[vectors.sum(axis=1) < 0] *= -1
    norms = np.linalg.norm(vectors, axis=1)
    safe = np.where(norms > 1e-6, norms, 1.0)[:, None]
    return (vectors / safe).reshape(GRID_ROWS, GRID_COLS, -1), \
        norms.reshape(GRID_ROWS, GRID_COLS), colour


class Decoder:
    """Prototype glyphs plus their labels."""

    def __init__(self, prototypes: np.ndarray, labels: dict[int, str]):
        self.prototypes = prototypes.astype(np.float32)
        self.labels = labels

    @classmethod
    def load(cls, atlas: Path = ATLAS_PATH, labels: Path = LABELS_PATH) -> "Decoder":
        blob = np.load(atlas)
        table = json.loads(Path(labels).read_text())
        return cls(blob["prototypes"],
                   {int(k): v for k, v in table.items() if k.isdigit()})

    def __call__(self, frame, want_colour: bool = False) -> Screen:
        vectors, norms, colour = cell_vectors(frame, want_colour=want_colour)
        flat = vectors.reshape(-1, vectors.shape[-1])
        sims = flat @ self.prototypes.T
        best = sims.argmax(axis=1)
        score = sims[np.arange(len(best)), best]

        chars = np.full(GRID_ROWS * GRID_COLS, " ", dtype="<U1")
        lit = norms.reshape(-1) > MIN_INK
        unknown = 0
        for i in np.nonzero(lit)[0]:
            label = self.labels.get(int(best[i]))
            if label is None:
                chars[i] = "?"
                unknown += 1
            else:
                chars[i] = label
        return Screen(chars=chars.reshape(GRID_ROWS, GRID_COLS),
                      colour=colour,
                      confidence=score.reshape(GRID_ROWS, GRID_COLS),
                      unknown=unknown)
