"""Label the glyph prototypes from text Brogue itself puts on screen.

Reading 174 prototypes off a contact sheet by eye is quick but wrong about one
letter in ten -- `a` and `n`, `o` and `u`, `e` and `c` are a few pixels apart at
this font size, and the first hand-labelled atlas decoded "adventurer" as
"ndverturer".

Brogue prints strings that are identical in every game: the opening messages, the
button bar, the status line. Each one is a labelled example. This script slides
every known string along every decoded row, and where a row matches well enough
to be that string, each cell casts a vote for the character it ought to be. The
prototype keeps the character with the most votes.

    python auto_label.py --frames runs/frames

Map glyphs (#, @, >, <, ~, +) appear in no sentence, so they stay hand-labelled;
anything the vote covers overrides the hand label, and the script prints every
disagreement so the two can be compared.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from brogueagent.screen import LABELS_PATH, MIN_INK, Decoder, cell_vectors

KNOWN_STRINGS = [
    "Hello and welcome, adventurer, to the Dungeons of Doom!",
    "Retrieve the Amulet of Yendor from the 26th floor and escape with it!",
    "Press <?> for help at any time.",
    "The dungeon exit is magically sealed!",
    "Only the bearer of the Amulet of Yendor may pass.",
    "Rest (z)", "Search (s)", "Stealth range:", "You (lit)",
    "gold pieces", "A sword", "The dungeon exit", "Str: 12", "Armor: 3",
]
# Every entry has at least two words on purpose. A single word is matched by its
# length alone, so "Health" lines up with "kobold" and teaches the decoder that
# an H is a k. Requiring a multi-word pattern is what makes an alignment
# self-evidencing.
def lit_runs(lit_row: np.ndarray) -> list[tuple[int, int]]:
    """(start column, length) of each run of non-blank cells in a row."""
    runs, start = [], None
    for c, on in enumerate(lit_row):
        if on and start is None:
            start = c
        elif not on and start is not None:
            runs.append((start, c - start))
            start = None
    if start is not None:
        runs.append((start, len(lit_row) - start))
    return runs


def locate(want: str, runs: list[tuple[int, int]]) -> int | None:
    """Where this string starts in a row, found from word lengths alone.

    Nothing here reads a character, which is the point: the labels are what is
    being learned, so the alignment cannot depend on them. "Hello and welcome,
    adventurer, to the Dungeons of Doom!" is a run of words of length
    5, 3, 8, 11, 2, 3, 8, 2, 5 -- a pattern that does not occur twice on a Brogue
    screen. An alignment is only used when the pattern matches exactly once in
    the row, so an ambiguous match teaches nothing rather than teaching noise.
    """
    words = want.split(" ")
    if len(words) < 2:
        return None
    lengths = [len(w) for w in words]
    hits = []
    for i in range(len(runs) - len(lengths) + 1):
        window = runs[i:i + len(lengths)]
        if [length for _, length in window] != lengths:
            continue
        # The gaps between words must be single spaces, as in the string.
        gaps_ok = all(window[j + 1][0] == window[j][0] + window[j][1] + 1
                      for j in range(len(window) - 1))
        if gaps_ok:
            hits.append(window[0][0])
    return hits[0] if len(hits) == 1 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/frames")
    ap.add_argument("--write", action="store_true", help="update labels.json")
    args = ap.parse_args()

    decoder = Decoder.load()
    paths = sorted(Path(args.frames).glob("*.png"))
    votes: dict[int, Counter] = defaultdict(Counter)
    matched = 0

    for path in paths:
        image = Image.open(path)
        vectors, norms, _ = cell_vectors(image)
        flat = vectors.reshape(-1, vectors.shape[-1])
        best = (flat @ decoder.prototypes.T).argmax(axis=1)
        best = best.reshape(vectors.shape[:2])
        lit = norms > MIN_INK
        screen = decoder(image)

        rows, _ = screen.chars.shape
        for r in range(rows):
            runs = lit_runs(lit[r])
            for want in KNOWN_STRINGS:
                start = locate(want, runs)
                if start is None:
                    continue
                matched += 1
                for i, ch in enumerate(want):
                    if ch != " ":
                        votes[int(best[r, start + i])][ch] += 1

    hand = {k: v for k, v in json.loads(LABELS_PATH.read_text()).items() if k.isdigit()}
    voted = {idx: counter.most_common(1)[0][0] for idx, counter in votes.items()
             if counter.most_common(1)[0][1] >= 3}
    agreed = sum(1 for i, ch in voted.items() if hand.get(str(i)) == ch)
    print(f"{matched} string alignments accepted over {len(paths)} frames")
    print(f"{len(voted)} prototypes labelled by vote; {agreed} agree with the hand labels")
    print("\ndisagreements (index: hand -> voted, votes):")
    for idx in sorted(voted):
        if hand.get(str(idx)) != voted[idx]:
            print(f"  {idx:>4}: {hand.get(str(idx), '-')!r} -> {voted[idx]!r} "
                  f"{dict(votes[idx].most_common(3))}")

    if args.write:
        merged = dict(hand)
        merged.update({str(i): ch for i, ch in voted.items()})
        table = json.loads(LABELS_PATH.read_text())
        comment = table.get("_comment", "")
        out = {"_comment": comment}
        out.update({k: merged[k] for k in sorted(merged, key=int)})
        LABELS_PATH.write_text(json.dumps(out, indent=0, ensure_ascii=False))
        print(f"\nwrote {LABELS_PATH} with {len(merged)} labels")


if __name__ == "__main__":
    main()
