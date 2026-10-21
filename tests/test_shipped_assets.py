"""The data files the agents cannot work without are actually in the repository.

The Brogue agent reads its screen by matching each cell against a glyph atlas,
`brogueagent/atlas.npz`. A blanket `*.npz` line in .gitignore meant that file
was in no clone of this repository, so the agent could not decode a single
frame anywhere but on the machine that built it -- and the three decoder tests
passed there for the same reason, which is why nothing said so.

It cannot be regenerated from what is here: `build_atlas.py` clusters glyphs out
of frames captured from a running game. So the file is committed, and this test
exists to fail loudly if it ever stops being, instead of three unrelated tests
erroring with FileNotFoundError.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BROGUE = next((p for p in (ROOT / "projects" / "03-brogue-screen-agent",
                           ROOT / "03-brogue") if p.is_dir()), None)


@pytest.mark.skipif(BROGUE is None, reason="this repository has no Brogue project")
def test_the_glyph_atlas_is_in_the_repository():
    atlas = BROGUE / "brogueagent" / "atlas.npz"
    assert atlas.is_file(), f"{atlas} is missing: the screen decoder cannot run"
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                             str(atlas.relative_to(ROOT)).replace("\\", "/")],
                            capture_output=True, text=True)
    assert listed.returncode == 0, (
        f"{atlas.relative_to(ROOT)} exists here but git does not track it, so it is "
        f"in no clone: {listed.stderr.strip()}")


@pytest.mark.skipif(BROGUE is None, reason="this repository has no Brogue project")
def test_the_atlas_matches_what_the_decoder_expects():
    """One prototype per label, and a prototype the right length to compare."""
    blob = np.load(BROGUE / "brogueagent" / "atlas.npz")
    prototypes, weight = blob["prototypes"], blob["weight"]
    assert prototypes.ndim == 2 and weight.ndim == 1
    assert len(prototypes) == len(weight), (
        f"{len(prototypes)} prototypes against {len(weight)} weights: the atlas was "
        "written by two different runs")
    assert (weight > 0).all(), "a prototype no frame ever supported"

    labels = json.loads((BROGUE / "brogueagent" / "labels.json").read_text("utf-8"))
    # The same filter the decoder uses: labels.json carries a "_comment" key
    # explaining what the file is, and only the digit keys are prototypes.
    indices = [int(k) for k in labels if k.isdigit()]
    assert indices, "labels.json names no prototype at all"
    assert max(indices) < len(prototypes), (
        f"labels.json names prototype {max(indices)} and the atlas holds "
        f"{len(prototypes)}: the two files are out of step")
