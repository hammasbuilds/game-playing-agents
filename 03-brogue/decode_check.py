"""Score the decoder against text whose content is already known.

Brogue puts strings on screen that are the same in every game -- the button bar
reads "Explore", "Rest", "Search", "Menu", "Inventory", and the status line
reads "Depth". Checking those is a real accuracy measurement with ground truth
that did not come from the decoder itself.

    python decode_check.py --frames runs/frames
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from brogueagent.screen import Decoder

# Strings Brogue always draws, and roughly where.
EXPECTED = ["Explore", "Rest", "Search", "Menu", "Inventory", "Depth"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/frames")
    ap.add_argument("--out", default="runs/decode_check.json")
    args = ap.parse_args()

    decoder = Decoder.load()
    paths = sorted(Path(args.frames).glob("*.png"))
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")

    found = {s: 0 for s in EXPECTED}
    unknown_share, lit_total, frames_with_bar = [], 0, 0
    sample_lines: list[str] = []
    for i, path in enumerate(paths):
        screen = decoder(Image.open(path))
        whole = "\n".join(screen.text())
        lit = int((screen.chars != " ").sum())
        lit_total += lit
        unknown_share.append(screen.unknown / max(lit, 1))
        hits = 0
        for s in EXPECTED:
            if s in whole:
                found[s] += 1
                hits += 1
        frames_with_bar += int(hits >= 3)
        if i == 0:
            sample_lines = [ln for ln in screen.text() if ln.strip()][:8]

    result = {
        "frames": len(paths),
        "unknown_glyph_share": round(float(np.mean(unknown_share)), 4),
        "lit_cells_per_frame": round(lit_total / len(paths), 1),
        "known_string_hit_rate": {s: round(found[s] / len(paths), 3) for s in EXPECTED},
        "frames_with_three_or_more": round(frames_with_bar / len(paths), 3),
    }
    print(json.dumps(result, indent=1))
    print("\nfirst frame, non-empty lines as decoded:")
    for line in sample_lines:
        print("  |" + line)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
