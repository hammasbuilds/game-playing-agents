"""Record the trained agent and an untrained one, side by side, at full size.

    python film.py --scenario defend_the_center --resolution 640x480

The agent is unchanged: it sees the same 60x80 greyscale stack it was trained
on. Only the camera changes — the game is asked to render at 640x480 instead of
160x120, and the recording is taken from that.

One thing this surfaced that is worth knowing before trusting any film as a
demonstration of a score: **the render resolution changes what the agent does.**
The 60x80 input is a box filter of the screen, and a box filter of a 640x480
render is a cleaner image than one of a 160x120 render — the renderer antialiases
differently — so the same weights behave differently. The numbers printed here
are measured at the resolution being filmed, not copied from the training log.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from doomagent.baselines import RandomAgent                      # noqa: E402
from doomagent.env import DoomEnv, EnvConfig                     # noqa: E402
from doomagent.vecenv import fine_timer                          # noqa: E402
from evaluate import load_policy                                 # noqa: E402
from film import side_by_side, write_clip                        # noqa: E402


# What a scenario's reward actually counts. Calling all three "kills" was wrong
# for two of them: health gathering pays for staying alive and deadly corridor
# pays for distance down the corridor.
UNITS = {
    "defend_the_center": "kills",
    "basic": "reward",
    "health_gathering": "ticks survived",
    "health_gathering_supreme": "ticks survived",
    "deadly_corridor": "distance reward",
}


def record(agent, cfg: EnvConfig, seed: int, max_steps: int = 2000):
    """One episode, keeping every colour frame the game drew."""
    env = DoomEnv(EnvConfig(**{**cfg.__dict__, "seed": seed}))
    frames, steps = [], 0
    try:
        obs = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()
        shot = env.screen_rgb()
        if shot is not None:
            frames.append(shot)
        while steps < max_steps:
            obs, _, done = env.step(agent.act(obs))
            steps += 1
            shot = env.screen_rgb()
            if shot is not None:
                frames.append(shot)
            if done:
                break
        return frames, env.episode_return(), steps
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="defend_the_center")
    ap.add_argument("--resolution", default="640x480")
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--concat", type=int, default=1,
                    help="join this many episodes end to end, for short scenarios")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--gif-width", type=int, default=340, dest="gif_width")
    ap.add_argument("--gif-frames", type=int, default=60, dest="gif_frames",
                    help="a gif is a thumbnail that moves, not the recording")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    policy_path = Path(args.policy or HERE / "runs" / args.scenario / "policy.pt")
    trained = load_policy(policy_path)
    cfg = EnvConfig(scenario=args.scenario, resolution=args.resolution)

    strips = []
    with fine_timer():
        for name, agent in (("untrained", RandomAgent(3, np.random.default_rng(0))), ("trained (PPO)", trained)):
            runs, returns = [], []
            for seed in args.seeds:
                frames, ret, steps = record(agent, cfg, seed)
                runs.append(frames)
                returns.append(ret)
                print(f"  {name:<14} seed {seed}: return {ret:>6.1f}  "
                      f"{steps} steps  {len(frames)} frames", flush=True)
            # Deadly corridor is over in about twenty agent steps -- two
            # seconds of film -- so short episodes are joined end to end rather
            # than shown one at a time. Everywhere else the longest single run
            # is the one worth watching.
            if args.concat > 1:
                joined = [frame for run in runs[:args.concat] for frame in run]
                runs, best = [joined], 0
            else:
                best = int(np.argmax([len(f) for f in runs]))
            mean = float(np.mean(returns))
            unit = UNITS.get(args.scenario, "reward")
            strips.append((name,
                           f"{args.scenario.replace('_', ' ')} · "
                           f"{mean:.1f} {unit} over {len(returns)} run"
                           f"{'s' if len(returns) > 1 else ''} at {args.resolution}",
                           runs[best]))

    out = Path(args.out or ROOT / "docs" / "media" / "doom"
               / f"{args.scenario}_{args.resolution}.mp4")
    frames = side_by_side(strips, width=args.width)
    print(f"\n{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    write_clip(frames, out, fps=args.fps, gif=True,
               gif_width=args.gif_width, gif_max_frames=args.gif_frames)


if __name__ == "__main__":
    main()
