"""Train one game from the arcade.

    python -m arcade.train --game boxing --steps 2000000 --envs 12
    python -m arcade.train --all

Checkpoints are written from step 0, so the before-and-after video exists the
moment training ends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .envs import CATALOGUE
from .ppo import PPOConfig, train

ROOT = Path(__file__).resolve().parents[1]

# How long each game needs to show something a viewer can see, measured by
# throughput on this machine rather than guessed: the racing game runs at a
# quarter of the Atari pace because every agent step is four physics steps and a
# render.
BUDGET = {"boxing": 2_000_000, "platformer": 2_000_000, "racing": 800_000,
          "pingpong": 2_000_000}


def occupied(out: Path, overwrite: bool) -> str | None:
    """Refuse to train into a directory that already holds a run.

    A second run started into the same directory silently replaces the first
    one's checkpoints -- including ckpt_000, the untrained network every verdict
    is measured against. That happened here: a racing run launched without
    --out overwrote the untrained checkpoint of the run whose numbers the README
    was quoting, and nothing in the output said a word about it.
    """
    existing = sorted(out.glob("ckpt_*.pt")) if out.exists() else []
    if not existing or overwrite:
        return None
    return (f"{out} already holds {len(existing)} checkpoints from an earlier "
            f"run. Pass --out to train somewhere else, or --overwrite to "
            f"replace them on purpose.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=sorted(CATALOGUE))
    ap.add_argument("--all", action="store_true", help="train every game in turn")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--arch", default="nature", choices=["nature", "impala", "small"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--lr-schedule", default="linear",
                    choices=["linear", "constant", "cosine", "warmup_cosine"])
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--out", default=None, help="where to write the run")
    ap.add_argument("--overwrite", action="store_true",
                    help="train into a directory that already holds a finished run")
    args = ap.parse_args()

    if not args.game and not args.all:
        ap.error("pass --game or --all")
    games = sorted(CATALOGUE) if args.all else [args.game]

    for key in games:
        game = CATALOGUE[key]
        steps = args.steps or BUDGET.get(key, 1_000_000)
        out = Path(args.out) if args.out else ROOT / "runs_arcade" / key
        # A second run started into the same directory silently replaces the
        # first run's checkpoints -- including ckpt_000, the untrained network
        # every verdict is measured against. That happened here: a racing run
        # launched without --out overwrote the untrained checkpoint of the run
        # whose numbers the README was quoting, and nothing said a word.
        problem = occupied(out, args.overwrite)
        if problem:
            ap.error(problem)
        cfg = PPOConfig(total_steps=steps, n_envs=args.envs, width=args.width,
                        arch=args.arch, device=args.device, lr=args.lr,
                        entropy_coef=args.entropy, lr_schedule=args.lr_schedule,
                        rollout=args.rollout,
                        epochs=args.epochs, minibatches=args.minibatches)
        print(f"\n=== {game.title} ({game.env_id}) : {steps:,} steps on "
              f"{args.envs} envs, {args.arch} x{args.width}, lr {args.lr} "
              f"({args.lr_schedule}), entropy {args.entropy} -> {out}", flush=True)
        summary = train(game, cfg, out, seed=args.seed)
        print(json.dumps({k: v for k, v in summary.items() if k != "checkpoints"},
                         indent=1))


if __name__ == "__main__":
    main()
