"""Score the pixel decoder against the game's own tile map.

This is the only file in the project that reads `info['semantic']`. The planner
never sees it; it exists so the claim "the agent plays from pixels" comes with a
number attached, broken down by daylight because night is where a pixel decoder
is supposed to struggle.

    python perception_check.py --episodes 3 --steps 400
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import crafter
import numpy as np

from crafteragent.perceive import MATERIALS, OVERLAYS, Perceiver


def semantic_window(info, view_cols: int, view_rows: int, sem_ids: dict) -> np.ndarray:
    """The true (material, overlay) names for the tiles the frame shows."""
    sem = info["semantic"]
    px, py = info["player_pos"]
    ox, oy = view_cols // 2, view_rows // 2
    out = np.empty((view_cols, view_rows), dtype=object)
    for x in range(view_cols):
        for y in range(view_rows):
            wx, wy = px + x - ox, py + y - oy
            if 0 <= wx < sem.shape[0] and 0 <= wy < sem.shape[1]:
                out[x, y] = sem_ids[int(sem[wx, wy])]
            else:
                out[x, y] = "none"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/perception.json")
    args = ap.parse_args()

    buckets: dict[str, list[float]] = defaultdict(list)
    inv_hits = inv_total = 0
    confusions: dict[tuple[str, str], int] = defaultdict(int)

    for ep in range(args.episodes):
        env = crafter.Env(seed=args.seed + ep, length=args.steps + 1)
        obs = env.reset()
        per = Perceiver()
        # Map the semantic ids back to names once; the view is a private detail
        # of the env, so this is read here and nowhere in the agent.
        sem = env._sem_view
        ids = {v: k for k, v in sem._mat_ids.items()}
        ids.update({v: k.__name__.lower() for k, v in sem._obj_ids.items()})
        ids = {k: ("none" if v is None else v) for k, v in ids.items()}

        rng = np.random.default_rng(args.seed + ep)
        for step in range(args.steps):
            action = int(rng.integers(env.action_space.n))
            obs, _, done, info = env.step(action)
            per.step = env._step
            p = per(obs)
            cols, rows = p.material.shape
            truth = semantic_window(info, cols, rows, ids)

            correct = 0
            for x in range(cols):
                for y in range(rows):
                    mat, obj = p.material_at(x, y), p.overlay_at(x, y)
                    # The semantic view reports one label per tile: the object if
                    # there is one, else the material. Collapse the decode the
                    # same way before comparing.
                    if obj is not None:
                        got = "player" if obj.startswith("player") else obj.split("-")[0]
                    else:
                        got = "none" if mat is None else mat
                    want = truth[x, y]
                    if got == want:
                        correct += 1
                    else:
                        confusions[(want, got)] += 1
            bucket = ("day" if p.daylight > 0.8 else
                      "dusk" if p.daylight > 0.5 else
                      "night" if p.daylight > 0.2 else "midnight")
            buckets[bucket].append(correct / (cols * rows))
            buckets["all"].append(correct / (cols * rows))

            true_inv = {k: min(v, 10) for k, v in info["inventory"].items()}
            for k, v in true_inv.items():
                inv_total += 1
                inv_hits += int(p.inventory.get(k, 0) == v)
            if done:
                break

    rows_out = []
    for name in ("all", "day", "dusk", "night", "midnight"):
        if buckets[name]:
            arr = np.array(buckets[name])
            rows_out.append({"daylight": name, "frames": len(arr),
                             "tile_accuracy": round(float(arr.mean()), 4),
                             "worst_frame": round(float(arr.min()), 4)})

    worst = sorted(confusions.items(), key=lambda kv: -kv[1])[:8]
    result = {
        "episodes": args.episodes, "steps_per_episode": args.steps,
        "tile_accuracy": rows_out,
        "inventory_slot_accuracy": round(inv_hits / max(inv_total, 1), 4),
        "top_confusions": [{"true": a, "decoded": b, "count": c} for (a, b), c in worst],
    }
    print(json.dumps(result, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
