"""A run must not quietly replace another run's checkpoints.

This happened: a second racing run was started without `--out`, wrote
`ckpt_000.pt` into the directory the first run lived in, and destroyed the
untrained network that every verdict for that run was measured against. Nothing
in the output mentioned it. The trainer now refuses, and the refusal names the
two ways to mean it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_trainer(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "arcade.train", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_training_into_an_occupied_directory_is_refused(tmp_path):
    (tmp_path / "ckpt_000.pt").write_bytes(b"not really a checkpoint")
    out = run_trainer("--game", "racing", "--steps", "1000", "--out", str(tmp_path))
    assert out.returncode != 0
    assert "already holds" in out.stderr
    # The refusal has to say what to do instead, or it is just an obstacle.
    assert "--out" in out.stderr and "--overwrite" in out.stderr


def test_the_checkpoint_it_would_have_replaced_is_still_there(tmp_path):
    victim = tmp_path / "ckpt_000.pt"
    victim.write_bytes(b"the untrained network")
    run_trainer("--game", "racing", "--steps", "1000", "--out", str(tmp_path))
    assert victim.read_bytes() == b"the untrained network"


def test_an_empty_directory_is_fine(tmp_path):
    """The guard must not block the ordinary case.

    Checked against the function rather than by starting a trainer: a real run
    of even a thousand CarRacing steps takes longer than this whole suite.
    """
    from arcade.train import occupied
    assert occupied(tmp_path / "fresh", overwrite=False) is None


def test_overwrite_means_overwrite(tmp_path):
    from arcade.train import occupied
    (tmp_path / "ckpt_000.pt").write_bytes(b"x")
    assert occupied(tmp_path, overwrite=False) is not None
    assert occupied(tmp_path, overwrite=True) is None


def test_a_directory_with_other_files_is_not_a_run(tmp_path):
    """Only checkpoints mean a run lives here; a log or a json does not."""
    from arcade.train import occupied
    (tmp_path / "train.log").write_text("hello")
    (tmp_path / "history.json").write_text("[]")
    assert occupied(tmp_path, overwrite=False) is None
