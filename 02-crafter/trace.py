"""Watch one episode decide things, step by step.

    python trace.py --seed 500 --tail 40

Prints the last `--tail` steps before the episode ended plus a goal histogram,
which is the quickest way to see whether the planner is dying, starving or just
walking in circles.
"""
from __future__ import annotations

import argparse
from collections import Counter

import crafter

from crafteragent.agent import PlannerAgent
from crafteragent.shelter import open_sides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--length", type=int, default=2000)
    ap.add_argument("--tail", type=int, default=40)
    args = ap.parse_args()

    env = crafter.Env(seed=args.seed, length=args.length)
    frame = env.reset()
    agent = PlannerAgent()
    agent.reset()
    rows, goals = [], Counter()
    info = {}
    while True:
        action = agent.act(frame)
        name = crafter.constants.actions[action]
        goals[agent.chosen] += 1
        inv = agent.inventory
        rows.append(
            f"s{len(rows):>4} day={agent.perceiver.daylight_now:4.2f} "
            f"hp={inv.get('health', 0)} food={inv.get('food', 0)} "
            f"drink={inv.get('drink', 0)} nrg={inv.get('energy', 0)} "
            f"wood={inv.get('wood', 0)} stone={inv.get('stone', 0)} "
            f"pos={agent.memory.pos} face={agent.memory.facing} "
            f"near={sorted(agent.memory.nearby_materials())} "
            f"threats={len(agent.memory.objects_of({'zombie', 'skeleton'}))} "
            f"goal={agent.chosen:<18} "
            f"shelter={agent.shelter_plan.mode}/{agent.shelter_plan.side} "
            f"open={len(open_sides(agent.memory))} -> {name}")
        frame, reward, done, info = env.step(action)
        if done:
            break

    print("\n".join(rows[-args.tail:]))
    print(f"\nended after {len(rows)} steps; "
          f"health={info['inventory']['health']} "
          f"(0 means killed, otherwise the step limit)")
    print(f"achievements: {sorted(k for k, v in info['achievements'].items() if v)}")
    print(f"tiles known: {len(agent.memory.tiles)}  "
          f"localised: {agent.memory.localised}")
    print("goal share: " + ", ".join(f"{k}={v}" for k, v in goals.most_common()))


if __name__ == "__main__":
    main()
