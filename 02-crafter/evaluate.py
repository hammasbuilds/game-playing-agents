"""Score agents on Crafter the way the benchmark defines it.

The headline number is Crafter's score: the geometric mean over the 22
achievements of the percentage of episodes in which each was unlocked, computed
as `exp(mean(log(1 + s_i))) - 1`. It is deliberately unkind -- an agent that
masters ten achievements and never touches the other twelve is capped low,
because the zeros drag the mean down. Reward and achievement *count* are printed
next to it since they move differently.

    python evaluate.py --agent planner --episodes 20
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import crafter
import numpy as np

from crafteragent.agent import PlannerAgent

ACHIEVEMENTS = tuple(crafter.constants.achievements)


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.unlocked: set[str] = set()
        self.chosen = "random"

    def reset(self) -> None:
        self.unlocked = set()

    def act(self, frame) -> int:
        return int(self.rng.integers(len(crafter.constants.actions)))


class ForagerAgent:
    """A hand-written floor that is harder than random: wander, and hit whatever
    is in front of you. `do` is the action that collects wood, drinks water,
    picks saplings and kills cows, so pressing it often gets a surprising amount
    of the way up the board -- which is the point of having this baseline."""

    name = "forager"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.chosen = "forage"
        self.unlocked: set[str] = set()

    def reset(self) -> None:
        self.unlocked = set()

    def act(self, frame) -> int:
        actions = crafter.constants.actions
        roll = self.rng.random()
        if roll < 0.45:
            return actions.index("do")
        return actions.index(["move_left", "move_right", "move_up", "move_down"]
                             [int(self.rng.integers(4))])


def crafter_score(success_rates: np.ndarray) -> float:
    """success_rates in percent, one per achievement."""
    return float(np.exp(np.mean(np.log(1 + success_rates))) - 1)


def run(agent, episodes: int, length: int, seed: int, verbose: bool = True) -> dict:
    unlocked_counts = Counter()
    rewards, lengths, deaths = [], [], 0
    tracker_hits = tracker_claims = tracker_truth = 0
    goal_counts = Counter()

    for ep in range(episodes):
        env = crafter.Env(seed=seed + ep, length=length)
        frame = env.reset()
        agent.reset()
        total = 0.0
        steps = 0
        info = {"achievements": {}}
        while True:
            action = agent.act(frame)
            frame, reward, done, info = env.step(action)
            total += reward
            steps += 1
            if done:
                break
        truth = {k for k, v in info["achievements"].items() if v > 0}
        unlocked_counts.update(truth)
        rewards.append(total)
        lengths.append(steps)
        deaths += int(steps < length)
        claimed = set(getattr(agent, "unlocked", set()))
        tracker_hits += len(claimed & truth)
        tracker_claims += len(claimed)
        tracker_truth += len(truth)
        goal_counts.update(getattr(agent, "goal_counts", {}))
        if verbose:
            print(f"  ep{ep:>3} steps={steps:<5} reward={total:6.1f} "
                  f"achievements={len(truth):>2} {sorted(truth)}", flush=True)

    rates = np.array([100.0 * unlocked_counts[a] / episodes for a in ACHIEVEMENTS])
    return {
        "agent": agent.name,
        "episodes": episodes,
        "length": length,
        "score": round(crafter_score(rates), 2),
        "reward_mean": round(float(np.mean(rewards)), 2),
        "reward_std": round(float(np.std(rewards, ddof=1)) if episodes > 1 else 0.0, 2),
        "achievements_mean": round(float(np.mean([
            sum(1 for a in ACHIEVEMENTS if unlocked_counts[a]) for _ in [0]])), 2),
        "distinct_achievements": int(sum(1 for a in ACHIEVEMENTS if unlocked_counts[a])),
        "steps_mean": round(float(np.mean(lengths)), 1),
        "died": deaths,
        "success_rates": {a: round(100.0 * unlocked_counts[a] / episodes, 1)
                          for a in ACHIEVEMENTS},
        "self_tracking": {
            "precision": round(tracker_hits / tracker_claims, 3) if tracker_claims else None,
            "recall": round(tracker_hits / tracker_truth, 3) if tracker_truth else None,
        },
        "goal_share": {k: round(v / max(sum(goal_counts.values()), 1), 3)
                       for k, v in goal_counts.most_common(12)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="planner", choices=["planner", "random", "forager", "all"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--length", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = ["random", "forager", "planner"] if args.agent == "all" else [args.agent]
    rows = []
    for name in names:
        agent = {"planner": PlannerAgent, "random": RandomAgent, "forager": ForagerAgent}[name]()
        print(f"\n== {name} ==", flush=True)
        rows.append(run(agent, args.episodes, args.length, args.seed))

    print(f"\n{'agent':<10} {'score':>7} {'reward':>8} {'achvs':>6} {'steps':>7} {'died':>5}")
    for r in rows:
        print(f"{r['agent']:<10} {r['score']:>6.2f}% {r['reward_mean']:>8.2f} "
              f"{r['distinct_achievements']:>6} {r['steps_mean']:>7.1f} {r['died']:>5}")

    out = Path(args.out or f"runs/eval_{args.agent}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
