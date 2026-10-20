"""Record the planner and a random agent side by side, at a watchable size.

    python film.py --episodes 1 --length 400

Crafter hands the agent a 64x64 image and that is what the agent keeps seeing —
its tile decoder is calibrated to those 7-pixel tiles and would break on
anything else. But `env.render(size)` draws the same state at any size, so the
camera can run at 512x512 while the agent runs at 64x64. The recording is not
the agent's view blown up; it is the game, drawn properly.

Each panel carries the achievements that run actually unlocked, read from the
environment rather than from what the agent believed it did.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import crafter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from crafteragent.agent import PlannerAgent                      # noqa: E402
from evaluate import RandomAgent                                 # noqa: E402
from film import side_by_side, write_clip                        # noqa: E402


def record(agent, seed: int, length: int, size: int):
    env = crafter.Env(seed=seed, length=length)
    frame = env.reset()
    agent.reset()
    frames = [env.render((size, size))]
    info = {"achievements": {}}
    steps = 0
    while True:
        frame, _, done, info = env.step(agent.act(frame))
        steps += 1
        frames.append(env.render((size, size)))
        if done:
            break
    unlocked = sorted(k for k, v in info["achievements"].items() if v > 0)
    return frames, unlocked, steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--size", type=int, default=512, help="render size, not agent input")
    ap.add_argument("--width", type=int, default=440, help="panel width in the film")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gif-width", type=int, default=340, dest="gif_width")
    ap.add_argument("--gif-frames", type=int, default=60, dest="gif_frames",
                    help="a gif is a thumbnail that moves, not the recording")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    strips = []
    for name, agent in (("random", RandomAgent(seed=args.seed)),
                        ("planned (no training)", PlannerAgent())):
        frames, unlocked, steps = record(agent, args.seed, args.length, args.size)
        print(f"  {name:<22} {steps:>4} steps, {len(unlocked)} achievements: "
              f"{', '.join(unlocked) or 'none'}", flush=True)
        subtitle = (f"{len(unlocked)} achievements · "
                    + (", ".join(a.replace('_', ' ') for a in unlocked[:4])
                       or "nothing unlocked"))
        strips.append((name, subtitle, frames))

    out = Path(args.out or ROOT / "docs" / "media" / "crafter" / "planner_vs_random.mp4")
    frames = side_by_side(strips, width=args.width)
    print(f"\n{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    write_clip(frames, out, fps=args.fps, gif=True,
               gif_width=args.gif_width, gif_max_frames=args.gif_frames)


if __name__ == "__main__":
    main()
