"""Start Brogue, find its window, screenshot it, and type into it.

Nothing here is a hook into the game. Brogue CE is an ordinary SDL application
downloaded from its release page; the agent sees it the way a person with a
screen recorder would, and types the way a keyboard does. Two Windows details
make that workable without taking over the machine:

* **Input goes to the window, not to the desktop.** Keys are delivered with
  `PostMessage` to the game's own window handle, so they land in Brogue's message
  queue whether or not it is the focused window. `SendInput`, the usual approach,
  types into whatever is focused -- which, on a machine where someone else is
  working, is not necessarily the game.
* **Capture is of the client area.** `GetClientRect` plus `ClientToScreen` gives
  the drawable region without the title bar, so the character grid starts at a
  predictable corner.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()

WM_KEYDOWN, WM_KEYUP, WM_CHAR = 0x0100, 0x0101, 0x0102

# Virtual-key codes for the keys Brogue needs that are not plain characters.
gdi32 = ctypes.windll.gdi32

# PrintWindow flag that renders the window's full contents, including the parts
# another window is covering. Windows 8.1 and later.
PW_RENDERFULLCONTENT = 0x00000002


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class WindowGone(RuntimeError):
    """The game window is gone or minimised, so there is nothing to capture."""


VK = {
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "enter": 0x0D, "escape": 0x1B, "space": 0x20, "tab": 0x09,
}


@dataclass
class BrogueWindow:
    """A running Brogue process and the window it drew."""

    process: subprocess.Popen
    hwnd: int

    # -- geometry ---------------------------------------------------------
    def client_rect(self) -> tuple[int, int, int, int]:
        r = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y, r.right, r.bottom

    @property
    def alive(self) -> bool:
        """Whether there is still a window worth photographing.

        `IsWindow` is not enough. When a Brogue game ends the window can go to
        the minimised position Windows uses -- a client rect of 0x0 at
        (-32000, -32000) -- while the handle stays valid, and asking for a
        screenshot of that raises deep inside the capture library. One game
        ending that way took a whole six-game batch with it.
        """
        if self.process.poll() is not None or not user32.IsWindow(self.hwnd):
            return False
        x, y, w, h = self.client_rect()
        return w > 0 and h > 0 and x > -30000 and y > -30000

    # -- output -----------------------------------------------------------
    def grab(self, sct) -> np.ndarray:
        """A picture of the game, whatever else is happening on the desktop.

        Screenshotting the patch of desktop where the window sits is the obvious
        way and it is fragile in two directions that matter when a run takes
        twenty minutes: minimise the window and there is nothing at those
        coordinates at all, and merely *cover* it and the capture quietly returns
        the covering window's pixels, so the agent plays on against a picture of
        something else. One game in a batch stopped at step 52 that way.

        `PrintWindow` with PW_RENDERFULLCONTENT asks the window to draw itself
        into a bitmap, which is independent of what is in front of it. It does
        not work for every renderer, so the result is sanity-checked and the
        desktop grab is kept as the fallback.
        """
        printed = self._print_window()
        if printed is not None:
            return printed
        x, y, w, h = self.client_rect()
        if w <= 0 or h <= 0 or x <= -30000:
            raise WindowGone(f"the window is minimised or gone: {(x, y, w, h)}")
        shot = sct.grab({"left": x, "top": y, "width": w, "height": h})
        return np.array(Image.frombytes("RGB", shot.size, shot.rgb))

    def _print_window(self) -> np.ndarray | None:
        """Ask the window to render itself. Returns None if it will not."""
        if not self.use_print_window:
            return None
        r = wintypes.RECT()
        if not user32.GetClientRect(self.hwnd, ctypes.byref(r)):
            return None
        width, height = r.right, r.bottom
        if width <= 0 or height <= 0:
            return None

        window_dc = user32.GetDC(self.hwnd)
        if not window_dc:
            return None
        memory_dc = bitmap = None
        try:
            memory_dc = gdi32.CreateCompatibleDC(window_dc)
            bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
            if not memory_dc or not bitmap:
                return None
            gdi32.SelectObject(memory_dc, bitmap)
            if not user32.PrintWindow(self.hwnd, memory_dc, PW_RENDERFULLCONTENT):
                return None

            header = BITMAPINFOHEADER()
            header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            header.biWidth = width
            header.biHeight = -height          # negative: top-down rows
            header.biPlanes = 1
            header.biBitCount = 32
            header.biCompression = 0           # BI_RGB
            buffer = ctypes.create_string_buffer(width * height * 4)
            info = BITMAPINFO()
            info.bmiHeader = header
            if not gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer,
                                   ctypes.byref(info), 0):
                return None
            pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
            frame = pixels[:, :, :3][:, :, ::-1]          # BGRA -> RGB
            # Some renderers hand back an empty bitmap rather than failing; a
            # Brogue screen is never uniformly one colour.
            if int(frame.max()) - int(frame.min()) < 8:
                self.use_print_window = False
                return None
            return np.ascontiguousarray(frame)
        finally:
            if bitmap:
                gdi32.DeleteObject(bitmap)
            if memory_dc:
                gdi32.DeleteDC(memory_dc)
            user32.ReleaseDC(self.hwnd, window_dc)

    # -- input ------------------------------------------------------------
    def send(self, key: str, hold: float = 0.02) -> None:
        """Send one keystroke. `key` is a single character or a VK name."""
        if key in VK:
            code, char = VK[key], None
        elif len(key) == 1:
            code, char = ord(key.upper()), key
        else:
            raise ValueError(f"unknown key {key!r}")
        user32.PostMessageW(self.hwnd, WM_KEYDOWN, code, 0)
        if char is not None:
            # SDL raises a text-input event from WM_CHAR, and Brogue reads
            # letter commands from that rather than from the raw key code.
            user32.PostMessageW(self.hwnd, WM_CHAR, ord(char), 0)
        time.sleep(hold)
        user32.PostMessageW(self.hwnd, WM_KEYUP, code, 0)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _windows_of(pid: int) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(callback, 0)
    return found


def launch(app_dir: Path, seed: int | None = None, font_size: int = 4,
           timeout: float = 25.0, extra: list[str] | None = None) -> BrogueWindow:
    """Start a new game and wait for its window to appear.

    `-n` skips the main menu and `-s` fixes the dungeon seed, which is what makes
    a run reproducible: the same seed is the same dungeon, so two agents can be
    compared on the same hundred dungeons rather than on their luck.
    """
    exe = Path(app_dir) / "brogue.exe"
    if not exe.exists():
        raise FileNotFoundError(f"{exe} -- run scripts/get_brogue.py first")
    args = [str(exe), "-n", "--size", str(font_size)]
    if seed is not None:
        args += ["-s", str(seed)]
    args += extra or []
    proc = subprocess.Popen(args, cwd=str(app_dir))

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"brogue exited with {proc.returncode}")
        windows = _windows_of(proc.pid)
        if windows:
            hwnd = windows[0][0]
            win = BrogueWindow(proc, hwnd)
            x, y, w, h = win.client_rect()
            if w > 100 and h > 100:
                time.sleep(1.5)  # let the first frame finish drawing
                return win
        time.sleep(0.25)
    proc.terminate()
    raise TimeoutError("brogue window did not appear")
