"""Play seeded games of Brogue and report what the game itself recorded.

    python play.py --seeds 1 2 3 4 5 --steps 300 --agent explorer

The score is not computed here. Brogue keeps its own high-score table and
`brogue.exe --scores` prints it, so the number in the results is the game's,
written by the game, about a run it played against an agent that could only see
pixels and send keys. The baseline presses random keys from the same set, which
is a fairer floor than it sounds: Brogue's `x` and `>` do a lot of work, so even
random presses sometimes descend a level.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import mss
import numpy as np

from brogueagent.agent import BrogueAgent, observe
from brogueagent.screen import Decoder
from brogueagent.window import WindowGone, launch

APP = Path(__file__).resolve().parents[2] / "games" / "brogue" / "app" / "BrogueCE-windows"
# `<` is deliberately absent. On depth 1 it leaves the dungeon and ends the run,
# so a baseline that pressed it would be scored on how fast it quit rather than
# on how it played.
RANDOM_KEYS = ["h", "j", "k", "l", "y", "u", "b", "n", "x", "z", "s", ">"]


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.chosen = "random"

    def reset(self) -> None:
        pass

    def act(self, obs) -> tuple[str, str]:
        return RANDOM_KEYS[int(self.rng.integers(len(RANDOM_KEYS)))], "random"


def read_scores(app: Path) -> list[dict]:
    """Brogue's own high-score table."""
    out = subprocess.run([str(app / "brogue.exe"), "--scores"], cwd=str(app),
                         capture_output=True, text=True, timeout=60).stdout
    rows = []
    for line in out.splitlines():
        # e.g. "1) 1234, on depth 5, killed by a jackal on depth 5"
        m = re.search(r"(\d[\d,]*)\s*,?\s*(?:on\s+)?depth\s+(\d+)", line, re.I)
        if m:
            rows.append({"score": int(m.group(1).replace(",", "")),
                         "depth": int(m.group(2)), "line": line.strip()})
    return rows


def play_one(agent, seed: int, steps: int, delay: float, decoder) -> dict:
    with mss.mss() as sct:
        win = launch(APP, seed=seed)
        reasons: dict[str, int] = {}
        coverage: list[float] = []
        health: list[float] = []
        # A new dungeon level starts almost entirely unexplored, so the count of
        # non-blank map cells collapses the moment the agent takes a staircase.
        # That is a descent, observed from the screen like everything else.
        descents = 0
        last_lit = None
        try:
            agent.reset()
            died_at = None
            for step in range(steps):
                if not win.alive:
                    died_at = step
                    break
                try:
                    frame = win.grab(sct)
                except WindowGone:
                    # The game ended, by death or by quit, and took its window.
                    died_at = step
                    break
                obs = observe(frame, decoder)
                lit = int((obs.screen.map != " ").sum())
                if last_lit is not None and last_lit > 120 and lit < 0.45 * last_lit:
                    descents += 1
                last_lit = lit
                coverage.append(obs.coverage)
                health.append(obs.health)
                key, why = agent.act(obs)
                reasons[why] = reasons.get(why, 0) + 1
                win.send(key)
                time.sleep(delay)
            # Quit cleanly so Brogue writes the run into its score table: Q,
            # then confirm with y, then dismiss the summary.
            for key in ["escape", "Q", "y", "escape", "enter", "space"]:
                if win.alive:
                    win.send(key)
                    time.sleep(0.35)
            time.sleep(1.5)
        finally:
            win.close()
            time.sleep(0.6)
    return {"seed": seed, "steps": steps, "died_at": died_at, "descents": descents,
            "agent_depth": getattr(agent, "depth", None),
            "coverage_mean": round(float(np.mean(coverage)), 4) if coverage else None,
            "health_mean": round(float(np.mean(health)), 3) if health else None,
            "health_final": round(float(health[-1]), 3) if health else None,
            "reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--delay", type=float, default=0.10)
    ap.add_argument("--agent", default="explorer",
                    choices=["explorer", "random", "cautious", "both", "all"])
    ap.add_argument("--out", default="runs/play.json")
    args = ap.parse_args()

    decoder = Decoder.load()
    names = {"both": ["random", "explorer"],
             "all": ["random", "explorer", "cautious"]}.get(args.agent, [args.agent])
    before = read_scores(APP)
    results = []
    for name in names:
        rows = []
        for seed in args.seeds:
            if name == "random":
                agent = RandomAgent(seed)
            else:
                # `cautious` is the same policy with one switch off: it never
                # walks toward a monster it can see, and only fights what is
                # already next to it.
                agent = BrogueAgent(name=name, seek_trouble=(name == "explorer"))
            row = play_one(agent, seed, args.steps, args.delay, decoder)
            rows.append(row)
            print(f"{name} seed {seed}: descents {row['descents']} "
                  f"coverage {row['coverage_mean']} health {row['health_final']} "
                  f"reasons {row['reasons']}", flush=True)
        results.append({"agent": name, "episodes": rows})

    after = read_scores(APP)
    new = after[:max(0, len(after) - len(before))] if len(after) > len(before) else after
    summary = {
        "agents": results,
        "scores_before": before,
        "scores_after": after,
        "new_entries": new,
        "deepest_recorded": max((r["depth"] for r in after), default=None),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nBrogue's own score table now holds {len(after)} runs; "
          f"deepest depth {summary['deepest_recorded']}")
    for row in after[:5]:
        print("  " + row["line"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
