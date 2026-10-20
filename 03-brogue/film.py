"""Replay stored frames through the decoder and film what the agent reads.

    python film.py --seed 1

Brogue is the one agent here that cannot be filmed by re-running it: it plays
the real game through its window, so a recording means launching Brogue, taking
over the keyboard and hoping nobody touches the machine for a minute. The frames
from a scored run are kept, so the film is built from those instead — the same
frames the agent saw, replayed through the same decoder.

That also makes a better demonstration than a screen recording would. The left
panel is the game. The right is what the agent *believes* is on screen: every
cell named from a glyph atlas, which is the entire method. Where the decoder is
unsure it prints `?`, and those are visible in the film rather than summarised
as an accuracy figure.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Order matters and is the opposite of what it looks like: the last insert ends
# up first on the path. This file is called film.py and the shared recorder is a
# package called film, so with HERE first `import film` finds this script again
# and fails as a circular import. ROOT has to win.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from brogueagent.screen import Decoder                           # noqa: E402
from film import side_by_side, write_clip                        # noqa: E402
from film.record import _font                                    # noqa: E402

# The dungeon view, without the sidebar and message lines around it.
INK = {
    "#": (150, 150, 150), ".": (90, 90, 90), "@": (255, 230, 120),
    ">": (120, 200, 255), "<": (120, 200, 255), "?": (230, 90, 90),
}


def render_decode(screen, cell: int = 11) -> np.ndarray:
    """Draw what the agent reads: the dungeon view, not the whole window.

    `screen.map` is the slice the agent acts on — the sidebar and the message
    lines are cut away, because the agent never reads them. Drawing the whole
    grid made the demonstration a wall of red question marks over the sidebar's
    text and implied the decoder was failing, when those cells are not ones it
    is asked to name.
    """
    grid = screen.map
    rows, cols = grid.shape
    img = Image.new("RGB", (cols * cell, rows * (cell + 3)), (10, 12, 16))
    draw = ImageDraw.Draw(img)
    font = _font(cell + 2)
    for r in range(rows):
        for c in range(cols):
            char = grid[r, c]
            if char == " ":
                continue
            draw.text((c * cell, r * (cell + 3)), char,
                      fill=INK.get(char, (120, 200, 150)), font=font)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames-dir", default=None, dest="frames_dir")
    ap.add_argument("--width", type=int, default=470)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gif-width", type=int, default=360, dest="gif_width")
    ap.add_argument("--gif-frames", type=int, default=60, dest="gif_frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir or HERE / "runs" / "frames")
    paths = sorted(frames_dir.glob(f"seed{args.seed}_*.png"),
                   key=lambda p: int(re.findall(r"(\d+)$", p.stem)[0]))
    if not paths:
        raise SystemExit(f"no stored frames for seed {args.seed} in {frames_dir}")

    decoder = Decoder.load()
    screens, decodes, shares = [], [], []
    for path in paths:
        frame = np.asarray(Image.open(path).convert("RGB"))
        screen = decoder(frame)
        grid = screen.map
        # The number worth printing is the share of *lit* dungeon cells that got
        # a name. Empty cells are not failures, and the sidebar is not the
        # agent's to read.
        lit = grid != " "
        named = lit & (grid != "?")
        shares.append(float(named.sum()) / max(1, int(lit.sum())))
        screens.append(frame)
        decodes.append(render_decode(screen))

    share = float(np.mean(shares))
    print(f"  {len(paths)} frames, {share * 100:.1f}% of lit dungeon cells named "
          f"(worst frame {min(shares) * 100:.1f}%)")

    strips = [
        ("the game", f"Brogue CE, seed {args.seed} — what the window shows", screens),
        ("what the agent reads", f"the dungeon view from a glyph atlas — "
                                 f"{share * 100:.0f}% of lit cells named, ? if unsure", decodes),
    ]
    out = Path(args.out or ROOT / "docs" / "media" / "brogue" / f"seed{args.seed}.mp4")
    film = side_by_side(strips, width=args.width)
    print(f"\n{len(film)} frames at {film[0].shape[1]}x{film[0].shape[0]}")
    write_clip(film, out, fps=args.fps, gif=True,
               gif_width=args.gif_width, gif_max_frames=args.gif_frames)


if __name__ == "__main__":
    main()
