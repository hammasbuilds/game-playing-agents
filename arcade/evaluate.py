"""Score an arcade game's checkpoint ladder under the repository protocol.

    python -m arcade.evaluate --game racing --episodes 12

Until this existed, the arcade had two different numbers for the same agent:
the README said racing went from −53.7 to +376.0, while the ladder written
beside the films said −48.6 to +262.7. Neither was wrong — they were different
episode counts on different seeds, scored at different times — but a reader has
no way to tell that, and one of them was always going to be stale.

So scoring happens in one place now, under `protocol.py`: the ladder on one set
of seeds, the winning checkpoint re-scored on a disjoint set, and a verdict
against two standard errors of the difference. The best checkpoint is reported rather than
the last, because a run can peak in the middle — the walking robot did, losing
857 return between its best checkpoint and the end of training.

Racing is the counter-example and the reason the episode count matters. Its
film ladder, scored on two tracks, appeared to fall from 330 to 263 across
training. Scored properly it rises the whole way, to 632.5 ± 92.7. CarRacing's
spread between tracks is larger than the differences being compared.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol import Score, score_ladder                          # noqa: E402

from .envs import CATALOGUE                                       # noqa: E402
from .play import rollout                                         # noqa: E402

RUNS = ROOT / "runs_arcade"


def play_factory(game, max_steps: int):
    """`handle` is a loaded model, or None for the random floor."""
    def play(handle, seeds) -> Score:
        returns, lengths = [], []
        for seed in seeds:
            episode = rollout(handle, game, int(seed), max_steps=max_steps)
            returns.append(episode.total_reward)
            lengths.append(episode.length)
        return Score(returns=returns, seeds=list(seeds), lengths=lengths)
    return play


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True, choices=sorted(CATALOGUE))
    ap.add_argument("--run", default=None)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=2500, dest="max_steps")
    args = ap.parse_args()

    from .play import load

    run = Path(args.run or RUNS / args.game)
    paths = sorted(run.glob("ckpt_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not paths:
        raise SystemExit(f"no checkpoints in {run}")

    game = CATALOGUE[args.game]
    checkpoints, steps_of = [], {}
    # The random policy is a floor, not a competitor. It was in the candidate
    # list once, which had two consequences: the comparison said "against the
    # untrained network" while comparing against random play, and on Kangaroo
    # -- where no checkpoint beats random -- the random policy won the selection
    # and was reported as the best checkpoint.
    extra = [("random", None)]
    steps_of["random"] = None
    for path in paths:
        model, loaded_game, meta = load(path)
        game = loaded_game
        checkpoints.append((path.name, model))
        steps_of[path.name] = meta.get("step")

    select = tuple(2_000 + i for i in range(args.episodes))
    report = tuple(9_000 + i for i in range(args.episodes))
    result = score_ladder(checkpoints, play_factory(game, args.max_steps),
                          select_seeds=select, report_seeds=report,
                          # ckpt_000 is the same network before a single
                          # gradient step: the only fair thing to credit
                          # training against.
                          baseline=checkpoints[0], extra=extra)
    for row in result["ladder"]:
        row["steps"] = steps_of.get(row["checkpoint"])
    result["game"] = args.game
    result["env_id"] = game.env_id

    # The untrained *network* is a different thing from a random policy, and the
    # gap between them is worth keeping: an untrained network is not uniform.
    random_rows = [r for r in result["ladder"] if r["checkpoint"] == "random"]
    if random_rows:
        result["random_floor"] = random_rows[0]

    v = result["verdict"]
    print(f"\n{args.game}: {v['baseline_mean']:.1f} -> {v['challenger_mean']:.1f}: "
          f"{v['label']} ({v['gain']:+.1f} against a bar of {v['bar']:.1f})")
    out = run / "eval.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
