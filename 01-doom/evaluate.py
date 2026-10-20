"""Score random / spray / the trained policy on the same episode seeds.

    python evaluate.py --scenario defend_the_center --episodes 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doomagent.baselines import PolicyAgent, RandomAgent, SprayAgent
from doomagent.env import ACTION_SETS, EnvConfig
from doomagent.evaluate import run_episodes
from doomagent.model import ActorCritic


def load_policy(path: Path, greedy: bool = True) -> PolicyAgent:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    # Checkpoints from before --width existed have no entry for it, and they are
    # all width 1; defaulting keeps them loadable.
    model = ActorCritic(tuple(blob["obs_shape"]), blob["n_actions"],
                        width=int(blob.get("width", 1)))
    model.load_state_dict(blob["model"])
    model.eval()
    return PolicyAgent(model, greedy=greedy, name="ppo-greedy" if greedy else "ppo-sampled")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center", choices=sorted(ACTION_SETS))
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--policy", default=None, help="path to policy.pt (default runs/<scenario>/policy.pt)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_cfg = EnvConfig(scenario=args.scenario)
    # Eval seeds are deliberately above the training seeds (train uses 0 + 1000*i)
    # so no evaluated map is one the policy was fitted on.
    seeds = [900_000 + i for i in range(args.episodes)]
    rng = np.random.default_rng(0)
    n_actions = len(ACTION_SETS[args.scenario])

    rows = [run_episodes(RandomAgent(n_actions, rng), env_cfg, seeds),
            run_episodes(SprayAgent(n_actions, rng), env_cfg, seeds)]

    policy_path = Path(args.policy or f"runs/{args.scenario}/policy.pt")
    if policy_path.exists():
        for greedy in (True, False):
            rows.append(run_episodes(load_policy(policy_path, greedy), env_cfg, seeds))
    else:
        print(f"no policy at {policy_path}: reporting baselines only")

    width = max(len(r["agent"]) for r in rows)
    print(f"\n{args.scenario}  ({args.episodes} held-out episodes, seeds 900000+)")
    print(f"{'agent'.ljust(width)}   mean +- sd        min    max   steps")
    for r in rows:
        print(f"{r['agent'].ljust(width)}   {r['return_mean']:7.2f} +- {r['return_std']:5.2f} "
              f"{r['return_min']:7.2f} {r['return_max']:6.2f} {r['steps_mean']:7.1f}")

    out = Path(args.out or f"runs/{args.scenario}/eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scenario": args.scenario, "episodes": args.episodes,
                               "seeds": seeds, "rows": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
