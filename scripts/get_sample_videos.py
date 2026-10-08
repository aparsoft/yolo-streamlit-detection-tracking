"""Download the sample videos into videos/ (they live on a GitHub release, not in git, so a clone stays small).

    python scripts/get_sample_videos.py            # all clips that aren't here yet
    python scripts/get_sample_videos.py --force    # re-download everything

Every file is checked against the release's SHA256SUMS, so all dev machines run the exact same clips.
Standard library only: no extra installs. Sources and licence: videos/ATTRIBUTION.md (Pexels).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/aparsoft/yolo-streamlit-detection-tracking/releases/download/sample-videos"
VIDEOS = Path(__file__).resolve().parent.parent / "videos"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "yolo-vision-studio"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="download even if a matching file is already there")
    args = ap.parse_args()

    sums = {}
    for line in fetch(f"{RELEASE}/SHA256SUMS").decode().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip().lstrip("*")] = digest

    VIDEOS.mkdir(exist_ok=True)
    failed = 0
    for name, digest in sorted(sums.items()):
        dest = VIDEOS / name
        if dest.exists() and not args.force and sha256(dest) == digest:
            print(f"  ok        {name}")
            continue
        print(f"  download  {name} …", end="", flush=True)
        data = fetch(f"{RELEASE}/{name}")
        if hashlib.sha256(data).hexdigest() != digest:
            print(" checksum mismatch, skipped")
            failed += 1
            continue
        dest.write_bytes(data)
        print(f" {len(data) / 1e6:.1f} MB")
    print(f"{len(sums) - failed} of {len(sums)} clips ready in {VIDEOS}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
