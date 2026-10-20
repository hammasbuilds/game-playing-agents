"""Learn the glyph prototypes from recorded frames, and write a labelling sheet.

    python capture_frames.py --seeds 1 2 3 --frames 28     # record
    python build_atlas.py                                   # cluster + sheet

Clusters every lit cell greedily by cosine similarity, frequency first, and
saves the prototypes to `brogueagent/atlas.npz`. The characters themselves are
not inferred -- `atlas_sheet.png` shows each prototype with its index, and
`brogueagent/labels.json` maps index to character. That file is written once, by
a person reading the sheet, which is cheaper and more exact than training an OCR
model to read a font that is already pixel-perfect on screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brogueagent.screen import ATLAS_PATH, BOX_H, BOX_W, LABELS_PATH, MIN_INK, cell_vectors

THRESHOLD = 0.80   # below this a cell starts a new prototype


def cluster(vectors: np.ndarray, threshold: float):
    protos = np.zeros((0, vectors.shape[1]), dtype=np.float32)
    assign = np.zeros(len(vectors), dtype=np.int32)
    for i, v in enumerate(vectors):
        if len(protos):
            sims = protos @ v
            j = int(sims.argmax())
            if sims[j] >= threshold:
                assign[i] = j
                continue
        protos = np.vstack([protos, v[None]])
        assign[i] = len(protos) - 1
    return protos, assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/frames")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--sheet", default="atlas_sheet.png")
    args = ap.parse_args()

    paths = sorted(Path(args.frames).glob("*.png"))[::args.stride]
    if not paths:
        raise SystemExit(f"no frames in {args.frames} -- run capture_frames.py first")

    collected = []
    for path in paths:
        vectors, norms, _ = cell_vectors(Image.open(path))
        flat = vectors.reshape(-1, vectors.shape[-1])
        collected.append(flat[norms.reshape(-1) > MIN_INK])
    data = np.vstack(collected).astype(np.float32)
    print(f"{len(paths)} frames, {len(data)} lit cells")

    protos, assign = cluster(data, args.threshold)
    weight = np.bincount(assign, minlength=len(protos))
    order = np.argsort(weight)[::-1]
    protos, weight = protos[order], weight[order]
    share = weight.cumsum() / weight.sum()
    print(f"{len(protos)} prototypes; "
          f"{int(np.searchsorted(share, 0.96)) + 1} cover 96% of cells, "
          f"{int(np.searchsorted(share, 0.99)) + 1} cover 99%")

    np.savez_compressed(ATLAS_PATH, prototypes=protos, weight=weight)
    print(f"wrote {ATLAS_PATH}")

    labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    cols, scale, pad, label_h = 16, 4, 10, 13
    cell_w, cell_h = BOX_W * scale + pad, BOX_H * scale + pad + label_h
    rows = (len(protos) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (16, 16, 24))
    draw = ImageDraw.Draw(sheet)
    for i, proto in enumerate(protos):
        tile = proto.reshape(BOX_H, BOX_W)
        tile = tile - tile.min()
        tile = (255 * tile / (tile.max() + 1e-6)).astype(np.uint8)
        img = Image.fromarray(tile).convert("RGB").resize(
            (BOX_W * scale, BOX_H * scale), Image.NEAREST)
        x, y = (i % cols) * cell_w + pad // 2, (i // cols) * cell_h + pad // 2
        sheet.paste(img, (x, y))
        known = str(i) in labels
        draw.text((x, y + BOX_H * scale), f"{i}{'' if known else '*'}",
                  fill=(150, 210, 255) if known else (255, 170, 120))
    sheet.save(args.sheet)
    print(f"wrote {args.sheet} -- indices marked * are not in labels.json yet")


if __name__ == "__main__":
    main()
