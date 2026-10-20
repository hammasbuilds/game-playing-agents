"""Carry hand labels from an old prototype atlas to a newly built one.

Prototypes are keyed by their rank, so rebuilding the atlas renumbers everything
and the hand-written labels no longer point at the glyph they were written for.
They are not lost, though: both atlases are sets of vectors in the same space, so
each new prototype can inherit the label of the old prototype it matches.

    python migrate_labels.py --old runs/atlas_old.npz --old-labels runs/labels_old.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brogueagent.screen import ATLAS_PATH, LABELS_PATH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--old-labels", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    old = np.load(args.old)["prototypes"]
    new = np.load(ATLAS_PATH)["prototypes"]
    table = json.loads(Path(args.old_labels).read_text())
    old_labels = {int(k): v for k, v in table.items() if k.isdigit()}

    sims = new @ old.T
    best = sims.argmax(axis=1)
    score = sims[np.arange(len(best)), best]

    migrated: dict[str, str] = {}
    for i, (j, s) in enumerate(zip(best, score)):
        if s >= args.threshold and int(j) in old_labels:
            migrated[str(i)] = old_labels[int(j)]
    print(f"{len(new)} new prototypes, {len(old_labels)} old labels, "
          f"{len(migrated)} carried over at similarity >= {args.threshold}")
    by_char: dict[str, int] = {}
    for ch in migrated.values():
        by_char[ch] = by_char.get(ch, 0) + 1
    print("labels by character:", dict(sorted(by_char.items(), key=lambda kv: -kv[1])))

    if args.write:
        out = {"_comment": table.get("_comment", "")}
        out.update({k: migrated[k] for k in sorted(migrated, key=int)})
        LABELS_PATH.write_text(json.dumps(out, indent=0, ensure_ascii=False))
        print(f"wrote {LABELS_PATH}")


if __name__ == "__main__":
    main()
