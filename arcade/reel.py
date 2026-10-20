"""Stack every game's before-and-after film into one reel.

One clip, every game, each row showing the same level played by the untrained
network and by the trained one. This is the artefact for a README header or a
feed: it has to make sense with no sound, no captions a viewer has to read, and
no clicking.

    python -m arcade.reel

Writes docs/media/reel.mp4 and docs/media/reel.gif.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .envs import CATALOGUE
from .progress import SUB_FONT, TITLE_FONT, write_clip

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
HEADER_H = 30


def read_frames(path: Path, limit: int) -> list[np.ndarray]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    frames = []
    try:
        for i, frame in enumerate(reader):
            if i >= limit:
                break
            frames.append(np.asarray(frame)[:, :, :3])
    finally:
        reader.close()
    return frames


def banner(width: int, title: str, detail: str) -> Image.Image:
    strip = Image.new("RGB", (width, HEADER_H), (16, 19, 24))
    draw = ImageDraw.Draw(strip)
    draw.text((10, 5), title, fill=(240, 243, 248), font=TITLE_FONT)
    tw = draw.textlength(detail, font=SUB_FONT)
    draw.text((width - tw - 10, 8), detail, fill=(140, 170, 215), font=SUB_FONT)
    return strip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=16)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--columns", type=int, default=0, help="0 picks 2 for more than two games")
    args = ap.parse_args()

    limit = args.seconds * args.fps
    rows = []
    for key in (args.games or sorted(CATALOGUE)):
        clip = MEDIA / key / "progress.mp4"
        ladder_path = MEDIA / key / "ladder.json"
        if not clip.exists():
            print(f"{key}: no film yet, skipping")
            continue
        frames = read_frames(clip, limit)
        if not frames:
            print(f"{key}: the film is empty, skipping")
            continue
        detail = ""
        if ladder_path.exists():
            ladder = json.loads(ladder_path.read_text())["ladder"]
            first = next((r for r in ladder if r.get("steps") == 0), None)
            last = ladder[-1]
            if first:
                detail = (f"untrained {first['return_mean']:.0f}"
                          f"   →   trained {last['return_mean']:.0f}")
        rows.append((CATALOGUE[key].title, detail, frames))
        print(f"{key}: {len(frames)} frames  {detail}")

    if not rows:
        raise SystemExit("no films to stack yet")

    # Laid out as a grid rather than a column. Four games stacked is 520x960, a
    # shape that makes a reader scroll past three of them to see the fourth;
    # two across is the same content in a header-sized picture.
    width = max(r[2][0].shape[1] for r in rows)
    length = max(len(r[2]) for r in rows)
    columns = args.columns if args.columns else (2 if len(rows) > 2 else 1)
    out = []
    for i in range(length):
        panels = []
        for title, detail, frames in rows:
            frame = frames[min(i, len(frames) - 1)]
            img = Image.fromarray(frame)
            if img.width != width:
                img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
            panel = Image.new("RGB", (width, img.height + HEADER_H), (16, 19, 24))
            panel.paste(banner(width, title, detail), (0, 0))
            panel.paste(img, (0, HEADER_H))
            panels.append(panel)
        cell_h = max(p.height for p in panels)
        n_rows = (len(panels) + columns - 1) // columns
        canvas = Image.new("RGB", (width * columns, cell_h * n_rows), (16, 19, 24))
        for n, panel in enumerate(panels):
            canvas.paste(panel, ((n % columns) * width, (n // columns) * cell_h))
        out.append(np.asarray(canvas))

    write_clip(out, MEDIA / "reel.mp4", fps=args.fps, gif=True)
    print(f"\nwrote {(MEDIA / 'reel.mp4').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
