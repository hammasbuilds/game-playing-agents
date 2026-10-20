"""Score every checkpoint, then film them playing the same level side by side.

This is the artefact the whole arcade exists to produce: one video in which the
untrained network and the trained one play **the same seed at the same time**,
each labelled with how long it had trained and what it is scoring. Nobody has to
read a learning curve to see the difference, and nobody has to press play --
the clips autoplay, muted and looping, wherever they are shown.

    python -m arcade.progress --game boxing

Writes, under docs/media/<game>/:
    ladder.json      score of every checkpoint over the same evaluation seeds
    step_000.mp4     the untrained network, alone
    step_100.mp4     the final network, alone
    progress.mp4     all checkpoints at once, side by side
    progress.gif     the same, small enough for a README
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from film.record import SUB_FONT, TITLE_FONT   # re-exported: the reel uses them
from .envs import CATALOGUE
from .play import load, rollout, score

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
# How wide each agent appears in a progress film. The old value was 260, which
# is narrower than every game in the catalogue renders: CarRacing draws 600x400
# and was being shrunk by more than half before anybody saw it. The camera should
# not be smaller than the thing it is pointed at.
PANEL_W = 420


def compose(strips: list[tuple[str, str, list[np.ndarray]]]) -> list[np.ndarray]:
    """Lay several episodes out left to right, playing together."""
    from film import side_by_side
    return side_by_side(strips, width=PANEL_W)


def _show(path: Path) -> str:
    """A short name for a path that may or may not sit under the repository."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_clip(frames: list[np.ndarray], path: Path, fps: int = 20,
               gif: bool = False, gif_width: int = 340) -> None:
    from film import write_clip as _write
    _write(frames, path, fps=fps, gif=gif, gif_width=gif_width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, choices=sorted(CATALOGUE))
    ap.add_argument("--run", default=None, help="training directory (default runs_arcade/<game>)")
    ap.add_argument("--episodes", type=int, default=8, help="evaluation episodes per checkpoint")
    ap.add_argument("--film-seed", type=int, default=12345)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--panels", type=int, default=3, help="how many checkpoints to film")
    args = ap.parse_args()

    game = CATALOGUE[args.game]
    run_dir = Path(args.run or ROOT / "runs_arcade" / args.game)
    out_dir = MEDIA / args.game
    out_dir.mkdir(parents=True, exist_ok=True)

    # One scorer per game. The watcher that films a run when it finishes and a
    # person filming the same run by hand will otherwise both score it and both
    # write the ladder, and the file ends up a mixture of two runs -- which is
    # how a ladder came to read 75 for the untrained network in one reading and
    # 25 in the next.
    lock = out_dir / ".filming"
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < 3600:
            raise SystemExit(f"{args.game} is already being filmed "
                             f"(started {age / 60:.0f} minutes ago); "
                             f"delete {lock} if that is stale")
        print(f"ignoring a stale lock {age / 60:.0f} minutes old")
    lock.write_text(str(os.getpid()))
    atexit.register(lambda: lock.unlink(missing_ok=True))

    saved = json.loads((run_dir / "checkpoints.json").read_text())

    # Trust the directory over the registry. The weights are the artefact; the
    # registry is a note about them, and a run stopped between the two leaves
    # checkpoints that are real and unlisted.
    listed = {entry["path"] for entry in saved}
    for path in sorted(run_dir.glob("ckpt_*.pt")):
        if path.name in listed:
            continue
        tag = path.stem.replace("ckpt_", "")
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
        saved.append({"tag": tag, "update": None, "steps": int(blob.get("step", 0)),
                      "path": path.name, "return_mean": None})
        print(f"found an unlisted checkpoint on disk: {path.name} "
              f"at {blob.get('step', 0):,} steps")
    saved.sort(key=lambda entry: entry["steps"])
    # -- the ladder --------------------------------------------------------
    seeds = [50_000 + i for i in range(args.episodes)]
    # This ladder exists to pick which checkpoints are worth filming. Scored on
    # a couple of episodes it is not a measurement of anything, and it was read
    # as one: a two-episode run of this script produced a racing ladder that
    # appeared to decline from 330 to 263 across training, and that "decline"
    # reached the README. Under the real protocol -- six selection and six
    # reporting episodes -- the same checkpoints rise to 632. CarRacing's
    # per-track spread is enormous and two tracks say nothing.
    if args.episodes < 6:
        print(f"  note: {args.episodes} episodes per checkpoint is enough to "
              f"choose what to film and not enough to compare checkpoints; "
              f"run `python -m arcade.evaluate --game {args.game}` for that",
              flush=True)
    ladder = [{"tag": "random", "steps": None,
               **score(None, game, seeds, max_steps=args.max_steps)}]
    for entry in saved:
        row = score(run_dir / entry["path"], game, seeds, max_steps=args.max_steps)
        ladder.append({"tag": entry["tag"], "steps": entry["steps"], **row})
        print(f"{entry['tag']:>6}  {entry['steps']:>9} steps  "
              f"score {row['return_mean']:8.2f} +- {row['return_std']:.2f}", flush=True)
    (out_dir / "ladder.json").write_text(json.dumps(
        {"game": game.key, "title": game.title, "env_id": game.env_id,
         "evaluation_seeds": seeds, "ladder": ladder,
         # Named in the file, so anything that reads it knows what it is.
         "measurement": "film-grade" if args.episodes < 6 else "ladder",
         "episodes_per_checkpoint": args.episodes}, indent=1))

    # -- the films ---------------------------------------------------------
    pick = [saved[0]] + ([saved[len(saved) // 2]] if len(saved) > 2 and args.panels > 2 else []) \
        + [saved[-1]]
    strips = []
    for entry in pick:
        model, loaded_game, _ = load(run_dir / entry["path"])
        episode = rollout(model, loaded_game, args.film_seed,
                          max_steps=args.max_steps, record=True)
        trained = "untrained" if entry["steps"] == 0 else f"{entry['steps']:,} steps"
        strips.append((trained, f"score {episode.total_reward:.0f}", episode.frames))
        write_clip(episode.frames, out_dir / f"step_{entry['tag']}.mp4")
        print(f"filmed {entry['tag']}: {len(episode.frames)} frames, "
              f"score {episode.total_reward:.0f}", flush=True)

    write_clip(compose(strips), out_dir / "progress.mp4", gif=True)
    print(f"\nwrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
