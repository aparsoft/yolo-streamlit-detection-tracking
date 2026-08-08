"""Fetch an aerial test image for the OBB section of yolo26_playground.ipynb.

The original three Wikimedia URLs in this file were guesses and all returned 404. These are
verified (checked with the Commons API), and the default one is the only candidate of the six
I tried that a DOTA model actually reads well — see the note on nadir vs oblique below.

    ./venv/bin/python docs/obb_images.py

Images land in ``images/`` with a filename the notebook's resolver recognises (`ship`,
`container`, `port`, `harbor`, `aerial`, `satellite`).
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# name -> (url, licence). Nadir (straight down) beats oblique every time: DOTA is trained on
# satellite imagery, so a photo taken out of a plane window detects almost nothing. Of the six
# images tried, the Planet Labs one returned 35 boxes; the oblique ones returned 0-2.
IMAGES = {
    "port_long_beach_satellite.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/"
        "Port_of_Long_Beach%2C_California_by_Planet_Labs.jpg/"
        "1920px-Port_of_Long_Beach%2C_California_by_Planet_Labs.jpg",
        "CC BY-SA 4.0, Planet Labs Inc.",
    ),
    # Oblique, kept only as a counter-example — it detects poorly on purpose.
    "container_terminal_hamburg_oblique.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/"
        "Container_Terminal_Tollerort_%28Hamburg-Steinwerder%29.Bianca_Rambow.phb.ajb.jpg/"
        "1920px-Container_Terminal_Tollerort_%28Hamburg-Steinwerder%29.Bianca_Rambow.phb.ajb.jpg",
        "CC BY-SA 3.0 de, Bianca Rambow",
    ),
}

UA = {"User-Agent": "yolo-vision-studio/1.0 (notebook demo; contact: repo owner)"}


def fetch(name: str, url: str, dest_dir: Path = config.IMAGES_DIR) -> Path:
    """Download *url* to ``dest_dir/name`` unless it is already there."""
    dest = dest_dir / name
    if dest.exists():
        print(f"{name:44s} already present ({dest.stat().st_size // 1024} KB)")
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as f:
        f.write(response.read())
    print(f"{name:44s} {dest.stat().st_size // 1024:>6} KB")
    return dest


if __name__ == "__main__":
    only_nadir = "--all" not in sys.argv
    for name, (url, licence) in IMAGES.items():
        if only_nadir and "oblique" in name:
            continue
        fetch(name, url)
        print(f"{'':44s} {licence}")
    print(
        "\nAttribution belongs in images/ATTRIBUTION.md — these are CC BY-SA, not public domain."
    )
    print("Pass --all to also fetch the oblique counter-example.")
