"""Train one game several times over a list of values for one setting.

    python -m arcade.sweep --game racing --param lr --values 1e-4,2.5e-4,5e-4 --steps 200000
    python -m arcade.sweep --game pingpong --param arch --values small,nature,impala

Runs the arms one after another rather than at once, because they would
otherwise be competing for the same cores and every arm would be measuring the
contention as much as the setting. Each arm lands in its own directory with its
own checkpoints, so the ladder of any arm can be scored and filmed exactly like a
single run.

The comparison it writes is the arm's *evaluation* score, not its training
return: training return is measured on whatever the policy happened to meet,
while the ladder plays every arm on the same levels.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SWEEPS = ROOT / "runs_arcade" / "sweeps"

# Which settings may be swept, and how a value is spelled on the command line.
SWEEPABLE = {
    "lr": "--lr", "entropy": "--entropy", "arch": "--arch", "width": "--width",
    "rollout": "--rollout", "epochs": "--epochs", "minibatches": "--minibatches",
    "envs": "--envs", "seed": "--seed", "lr_schedule": "--lr-schedule",
}


def python() -> str:
    return str(PY) if PY.exists() else sys.executable


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--param", required=True, choices=sorted(SWEEPABLE))
    ap.add_argument("--values", required=True,
                    help="comma separated, e.g. 1e-4,2.5e-4,5e-4")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--episodes", type=int, default=5, help="evaluation episodes per arm")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="further switches held constant across the arms")
    args = ap.parse_args()

    values = [v.strip() for v in args.values.split(",") if v.strip()]
    if len(values) < 2:
        raise SystemExit("a sweep needs at least two values to compare")

    sys.path.insert(0, str(ROOT))
    from arcade.envs import CATALOGUE
    from arcade.play import score

    out_root = SWEEPS / args.game / args.param
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {"game": args.game, "param": args.param, "values": values,
                "steps": args.steps, "episodes": args.episodes,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "arms": []}

    for value in values:
        arm_dir = out_root / slug(value)
        cmd = [python(), "-m", "arcade.train", "--game", args.game,
               "--steps", str(args.steps), "--envs", str(args.envs),
               "--out", str(arm_dir), SWEEPABLE[args.param], str(value), *args.extra]
        print(f"\n=== {args.param}={value} ===", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT))
        arm = {"value": value, "dir": str(arm_dir), "returncode": proc.returncode,
               "minutes": round((time.time() - started) / 60, 1)}

        # Score the trained checkpoint and the untrained one on the same levels,
        # so the arms are comparable to each other and not only to themselves.
        seeds = [70_000 + i for i in range(args.episodes)]
        game = CATALOGUE[args.game]
        for tag in ("000", "100"):
            path = arm_dir / f"ckpt_{tag}.pt"
            if path.exists():
                arm[f"score_{tag}"] = score(path, game, seeds)["return_mean"]
        if "score_000" in arm and "score_100" in arm:
            arm["gain"] = round(arm["score_100"] - arm["score_000"], 2)
        manifest["arms"].append(arm)
        (out_root / "sweep.json").write_text(json.dumps(manifest, indent=1))
        print(f"  {args.param}={value}: {arm.get('score_000')} -> {arm.get('score_100')}",
              flush=True)

    ranked = [a for a in manifest["arms"] if a.get("gain") is not None]
    ranked.sort(key=lambda a: -a["gain"])
    manifest["best"] = ranked[0]["value"] if ranked else None
    (out_root / "sweep.json").write_text(json.dumps(manifest, indent=1))

    print(f"\n{'value':<16}{'untrained':>11}{'trained':>10}{'gain':>9}{'minutes':>9}")
    for arm in manifest["arms"]:
        print(f"{str(arm['value']):<16}{str(arm.get('score_000','—')):>11}"
              f"{str(arm.get('score_100','—')):>10}{str(arm.get('gain','—')):>9}"
              f"{arm['minutes']:>9}")
    print(f"\nbest {args.param}: {manifest['best']}")
    print(f"wrote {(out_root / 'sweep.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
