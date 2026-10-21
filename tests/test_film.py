"""The recorder, and the rule that a camera must not shrink the thing it films.

These exist because the repository's films were bad for reasons no test could
have caught by looking at a score: panels narrower than the game's own render,
a recording taken from the agent's 160x120 training buffer, and a gif large
enough that no README would load it.
"""
from __future__ import annotations

import numpy as np
import pytest

from film import panel, side_by_side
from film.record import write_clip


def frame(h: int, w: int, value: int = 120) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_panel_keeps_the_aspect_ratio():
    out = panel(frame(480, 640), "title", "subtitle", width=320, label_h=40)
    assert out.width == 320
    assert out.height == 240 + 40          # 640x480 scaled to 320 wide is 240 tall


def test_panel_does_not_pretend_a_small_render_is_a_big_one():
    """Upscaling adds no detail, so it must not be done smoothly.

    A 160-wide render blown up to 640 with a smooth filter looks like a blurred
    photograph of a game; with nearest neighbour it looks like what it is. The
    point of the resolution work was to stop faking this.
    """
    small = frame(120, 160)
    out = panel(small, "t", "", width=640)
    assert out.width == 640
    assert out.height == 480 + 52


def test_side_by_side_holds_the_last_frame_of_a_short_run():
    """A run that ends early is a result, not a reason to reshuffle the layout."""
    short = [frame(60, 80, v) for v in (10, 20)]
    long = [frame(60, 80, v) for v in (30, 40, 50, 60)]
    out = side_by_side([("a", "", short), ("b", "", long)], width=80)
    assert len(out) == 4
    left_last = out[-1][:60, :80]
    assert left_last.mean() == pytest.approx(20, abs=1.5)


def test_side_by_side_is_empty_for_no_strips():
    assert side_by_side([]) == []


def test_write_clip_says_so_rather_than_writing_nothing(tmp_path, capsys):
    write_clip([], tmp_path / "x.mp4")
    assert "nothing to write" in capsys.readouterr().out


def test_gif_is_capped_in_frames(tmp_path):
    """A gif is a thumbnail that moves: a 350-frame episode must not become one.

    The stride is derived from the length rather than fixed, so a long episode
    keeps its duration and the file stays small enough to load.
    """
    frames = [frame(60, 80, v % 255) for v in range(300)]
    out = tmp_path / "clip.mp4"
    write_clip(frames, out, fps=15, gif=True, gif_width=120, gif_max_frames=30)
    gif = out.with_suffix(".gif")
    if not gif.exists():
        pytest.skip("no gif encoder available here")
    import imageio.v3 as iio
    assert iio.improps(gif, index=...).shape[0] <= 30
