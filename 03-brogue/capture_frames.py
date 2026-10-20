"""Record raw frames from real games, once, so the decoder can be tuned offline.

    python capture_frames.py --seeds 1 2 3 --frames 30 --out runs/frames

Relaunching the game for every change to a threshold is slow and flaky; a
directory of PNGs is neither.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mss
from PIL import Image

from brogueagent.window import launch

APP = Path(__file__).resolve().parents[2] / "games" / "brogue" / "app" / "BrogueCE-windows"
WANDER = list("hjklyubn") * 3 + ["i", "escape", "x", "escape", "s", "z", "x", "escape"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", default="runs/frames")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with mss.mss() as sct:
        for seed in args.seeds:
            win = launch(APP, seed=seed)
            try:
                for i in range(args.frames):
                    win.send(WANDER[i % len(WANDER)])
                    time.sleep(0.12)
                    frame = win.grab(sct)
                    Image.fromarray(frame).save(out / f"seed{seed}_{i:03d}.png")
                print(f"seed {seed}: {args.frames} frames")
            finally:
                win.close()
            time.sleep(0.4)
    print(f"wrote {len(list(out.glob('*.png')))} frames to {out}")


if __name__ == "__main__":
    main()
