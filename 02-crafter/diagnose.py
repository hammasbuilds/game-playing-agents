"""Why do episodes end? Run several and report the state at the last step.

    python diagnose.py --episodes 8

Prints, per episode, how it ended and what the agent was doing for the final
twenty steps. Written because the score said "13%" while every single episode
was ending at step 200, and the score alone does not say whether that is zombies,
thirst or starvation.
"""
from __future__ import annotations

import argparse
from collections import Counter

import crafter

from crafteragent.agent import PlannerAgent
from crafteragent.shelter import is_sheltered, open_sides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--length", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=500)
    args = ap.parse_args()

    causes = Counter()
    for ep in range(args.episodes):
        env = crafter.Env(seed=args.seed + ep, length=args.length)
        frame = env.reset()
        agent = PlannerAgent()
        agent.reset()
        recent, info, sheltered_steps, nights = [], {}, 0, 0
        was_night = False
        while True:
            action = agent.act(frame)
            recent.append(agent.chosen)
            if len(recent) > 20:
                recent.pop(0)
            night = agent.perceiver.daylight_now < 0.3
            if night and not was_night:
                nights += 1
            was_night = night
            if is_sheltered(agent.memory):
                sheltered_steps += 1
            frame, _, done, info = env.step(action)
            if done:
                break
        inv = info["inventory"]
        if inv["health"] <= 0:
            if inv["food"] == 0 and inv["drink"] == 0:
                cause = "killed (and starving and parched)"
            elif inv["food"] == 0:
                cause = "killed (and starving)"
            elif inv["drink"] == 0:
                cause = "killed (and parched)"
            else:
                cause = "killed outright"
        else:
            cause = "reached the step limit"
        causes[cause] += 1
        print(f"ep{ep:>3} {len(recent) and env._step:>5} steps  {cause:<34} "
              f"hp={inv['health']} food={inv['food']} drink={inv['drink']} "
              f"energy={inv['energy']} stone={inv['stone']}  "
              f"nights_entered={nights} steps_sealed={sheltered_steps}  "
              f"last_goals={Counter(recent).most_common(3)}")
    print("\n" + "\n".join(f"{v:>3}  {k}" for k, v in causes.most_common()))


if __name__ == "__main__":
    main()
