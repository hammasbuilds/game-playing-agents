"""Download Brogue CE into games/brogue and unpack it.

The game is not committed to this repository -- it is someone else's GPL binary
release, and it is 1.7 MB away from any machine that wants it.

    python scripts/get_brogue.py
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

REPO = "tmewett/BrogueCE"
TARGET = Path(__file__).resolve().parents[1] / "games" / "brogue"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
            f"https://api.github.com/repos/{REPO}/releases/latest", timeout=60) as fh:
        release = json.load(fh)
    assets = [a for a in release["assets"]
              if a["name"].endswith("windows-x86_64.zip")]
    if not assets:
        raise SystemExit(f"no Windows build in {release['tag_name']}: "
                         f"{[a['name'] for a in release['assets']]}")
    asset = assets[0]
    print(f"{release['tag_name']}: {asset['name']} "
          f"({asset['size'] / 1e6:.1f} MB)")

    with urllib.request.urlopen(asset["browser_download_url"], timeout=300) as fh:
        blob = fh.read()
    (TARGET / asset["name"]).write_bytes(blob)
    app = TARGET / "app"
    app.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(app)
    exes = list(app.rglob("brogue.exe"))
    print(f"unpacked to {exes[0].parent if exes else app}")


if __name__ == "__main__":
    main()
