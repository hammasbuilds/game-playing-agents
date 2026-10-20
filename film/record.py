"""Film an agent at the size the game renders, not the size it thinks in.

Every demo in this repository was recorded from the agent's own input — 84x84
greyscale for the arcade games, a 160x120 screen for DOOM — and then scaled up
for the page. That is why they looked like 1996 even when the game underneath
did not: a 320x240 recording of DOOM is a choice, and it was the wrong one.

The agent's input and the recording are separate things. The agent still sees
its small frame; the camera sees whatever the game can draw. Nothing here
touches a policy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
INK = (236, 240, 247)
DIM = (150, 175, 215)
BACK = (12, 14, 18)


def _font(size: int):
    """A readable font, falling back to PIL's built-in one.

    The default bitmap font is about 11 pixels tall, which survives a README and
    disappears in a feed where the video plays at a third of its size.
    """
    from PIL import ImageFont

    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = _font(17)
SUB_FONT = _font(14)


def panel(frame: np.ndarray, title: str, subtitle: str = "",
          width: int = 640, label_h: int = 52) -> Image.Image:
    """One agent's screen, with who it is written underneath.

    The frame keeps its aspect ratio and is never enlarged past its own
    resolution: blowing a 160-pixel-wide render up to 640 does not add detail, it
    only makes the lack of it fill more of the screen.
    """
    img = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")
    if img.width != width:
        height = round(img.height * width / img.width)
        resample = Image.LANCZOS if img.width > width else Image.NEAREST
        img = img.resize((width, height), resample)
    out = Image.new("RGB", (width, img.height + label_h), BACK)
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((12, img.height + 7), title, fill=INK, font=_font(max(14, width // 34)))
    if subtitle:
        draw.text((12, img.height + 7 + max(18, width // 26)), subtitle, fill=DIM,
                  font=_font(max(12, width // 44)))
    return out


def side_by_side(strips: list[tuple[str, str, list[np.ndarray]]],
                 width: int = 640, gap: int = 10) -> list[np.ndarray]:
    """Play several runs together, left to right.

    Runs end at different times — that is itself the result, since an untrained
    agent usually dies first — so a finished one holds its last frame rather than
    disappearing and shuffling the others sideways.
    """
    if not strips:
        return []
    length = max(len(frames) for _, _, frames in strips)
    out = []
    for i in range(length):
        panels = [panel(frames[min(i, len(frames) - 1)], title, subtitle, width)
                  for title, subtitle, frames in strips]
        height = max(p.height for p in panels)
        canvas = Image.new("RGB", (sum(p.width for p in panels)
                                   + gap * (len(panels) - 1), height), BACK)
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + gap
        out.append(np.asarray(canvas))
    return out


def _show(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_clip(frames: list[np.ndarray], path: Path, fps: int = 20,
               gif: bool = False, gif_width: int = 440, quality: int = 5,
               gif_every: int = 4, gif_max_frames: int = 90) -> None:
    """Write an mp4, and optionally a gif for places that will not play video.

    Size is a feature here: a 20 MB clip of a 15-second demo is one nobody waits
    for, and GitHub will not render it inline at all. Quality 6 at 640 wide keeps
    the textures legible at a few MB; the gif is narrower and samples every third
    frame, because a gif is a thumbnail that moves, not the recording.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        print(f"  (nothing to write for {path.name})")
        return
    try:
        import imageio.v2 as imageio
        # H.264 refuses odd dimensions.
        even = [f[:f.shape[0] // 2 * 2, :f.shape[1] // 2 * 2] for f in frames]
        imageio.mimwrite(path, even, fps=fps, quality=quality, macro_block_size=None)
    except Exception as exc:
        # Only the encoding belongs in here. Wrapping the logging too meant a
        # failure to *print* the filename was reported as "no mp4" for a video
        # that had been written perfectly well.
        print(f"  (no mp4 for {path.name}: {type(exc).__name__}: {exc})")
    else:
        print(f"  {_show(path)}  {path.stat().st_size / 1e6:.1f} MB")

    if not gif:
        return
    try:
        import imageio.v2 as imageio
        # A gif of a 350-step episode is six megabytes however carefully it is
        # encoded, so the frame count is capped and the sampling stride derived
        # from the length rather than fixed: the clip stays the same duration
        # and the file stays small enough for a README to load.
        import math
        step = max(gif_every, math.ceil(len(frames) / max(1, gif_max_frames)))
        scale = gif_width / frames[0].shape[1]
        small = [np.asarray(Image.fromarray(f).resize(
            (gif_width, round(f.shape[0] * scale)), Image.LANCZOS))
            for f in frames[::step]]
        gif_path = path.with_suffix(".gif")
        imageio.mimwrite(gif_path, small,
                         duration=1000 * step / max(1, fps), loop=0)
        print(f"  {_show(gif_path)}  {gif_path.stat().st_size / 1e6:.1f} MB")
    except Exception as exc:
        print(f"  (no gif for {path.name}: {type(exc).__name__}: {exc})")
