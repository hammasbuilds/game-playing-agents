"""Train the pixel policy on one ViZDoom scenario.

    python train.py --scenario defend_the_center --steps 1500000 --envs 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from doomagent.env import ACTION_SETS, EnvConfig
from doomagent.ppo import PPOConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="defend_the_center", choices=sorted(ACTION_SETS))
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1,
                    help="channel multiplier; >1 is worth it only on a GPU")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"runs/{args.scenario}")
    env_cfg = EnvConfig(scenario=args.scenario, frame_skip=args.frame_skip, seed=args.seed)
    cfg = PPOConfig(total_steps=args.steps, n_envs=args.envs, rollout=args.rollout,
                    lr=args.lr, entropy_coef=args.entropy, width=args.width,
                    device=args.device)
    print(f"training {args.scenario}: {args.steps} agent steps "
          f"({args.steps * args.frame_skip} game tics) on {args.envs} envs -> {out}")
    summary = train(env_cfg, cfg, out, seed=args.seed)
    print(json.dumps(summary, indent=1))
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
