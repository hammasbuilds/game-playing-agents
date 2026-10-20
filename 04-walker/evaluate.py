"""Score the checkpoint ladder under the repository's evaluation protocol.

    python evaluate.py --episodes 20

The protocol lives in `protocol.py` at the root and is the same one every other
project is scored by: the ladder is scored on one set of seeds, the best
checkpoint is re-scored on a disjoint set, and the verdict is the gap against
two standard errors of the difference.

That separation matters here more than anywhere else in the repository, because
this run peaked in the middle: 4,502 at 700k steps against 3,550 at 2M. Picking
the winner and quoting the score that won is how a noisy ladder reports a
checkpoint's luck as its skill.

The policy's mean action is used rather than a sample. Sampling while scoring
measures the exploration noise as much as the policy, and the mean action is
what a deployed controller would do.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from protocol import Score, score_ladder                          # noqa: E402
from walker.checkpoint import load                                # noqa: E402


def play_factory(env_id: str, max_steps: int = 1000):
    def play(handle, seeds) -> Score:
        model, norm = handle
        env = gym.make(env_id)
        returns, lengths = [], []
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            total, steps = 0.0, 0
            while steps < max_steps:
                with torch.no_grad():
                    action, _, _ = model.act(torch.from_numpy(norm(obs[None])),
                                             greedy=True)
                obs, reward, terminated, truncated, _ = env.step(
                    np.clip(action[0], env.action_space.low, env.action_space.high))
                total += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(steps)
        env.close()
        return Score(returns=returns, seeds=list(seeds), lengths=lengths)
    return play


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None)
    ap.add_argument("--env", default="Walker2d-v5")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    run = Path(args.run or HERE / "runs" / args.env.split("-")[0].lower())
    paths = sorted(run.glob("ckpt_*.pt"),
                   key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not paths:
        raise SystemExit(f"no checkpoints in {run}")

    checkpoints, steps_of = [], {}
    for path in paths:
        model, norm, state = load(path)
        checkpoints.append((path.name, (model, norm)))
        steps_of[path.name] = state["step"]

    select = tuple(2_000 + i for i in range(args.episodes))
    report = tuple(9_000 + i for i in range(args.episodes))
    result = score_ladder(checkpoints, play_factory(args.env),
                          select_seeds=select, report_seeds=report,
                          # Said out loud rather than relied on: ckpt_000 is the
                          # untrained network and nothing else may stand in for it.
                          baseline=checkpoints[0])
    for row in result["ladder"]:
        row["steps"] = steps_of[row["checkpoint"]]
    result["best"]["steps"] = steps_of[result["best"]["checkpoint"]]
    result["untrained"]["steps"] = steps_of[result["untrained"]["checkpoint"]]
    result["env_id"] = args.env

    v = result["verdict"]
    print(f"\n{v['baseline_mean']:.1f} -> {v['challenger_mean']:.1f}: {v['label']} "
          f"({v['gain']:+.1f} against a bar of {v['bar']:.1f})")
    (run / "eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {run / 'eval.json'}")


if __name__ == "__main__":
    main()
