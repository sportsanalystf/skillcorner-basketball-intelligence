"""Download the SkillCorner basketball open data into `data/raw/opendata/`.

The per-game tracking files are Git LFS objects. This script:

1. shallow-clones the repository (docs, events, aggregates, LFS pointers),
2. tries `git lfs pull` for the tracking objects,
3. falls back to GitHub's resolved-LFS endpoint (`media.githubusercontent.com`)
   for any file that is still a pointer, downloading in parallel.

Step 3 matters in environments where the LFS batch API is unavailable but plain
HTTPS is not — a common situation behind a proxy.
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "opendata"
REPO = "SkillCorner/opendata-basketball"
BRANCH = "main"
CLONE = f"https://github.com/{REPO}.git"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{BRANCH}"

POINTER_MAX_BYTES = 10_000   # an LFS pointer is ~130 bytes; a real file is 30-45 MB
MIN_TRACKING_BYTES = 20_000_000
MAX_ATTEMPTS = 4


def is_complete(path: Path) -> bool:
    """A tracking file is only usable if it is large AND a readable gzip stream.

    Large parallel downloads truncate silently often enough that checking the
    size alone is not sufficient — we decompress the tail to prove the stream
    terminates cleanly.
    """
    if not path.exists() or path.stat().st_size < MIN_TRACKING_BYTES:
        return False
    try:
        with gzip.open(path, "rb") as f:
            last = b""
            while chunk := f.read(1 << 22):
                last = chunk
        return last.rstrip().endswith(b"}")
    except (OSError, EOFError, gzip.BadGzipFile):
        return False


def _run(cmd, cwd=None, check=False):
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def clone_repo() -> None:
    if (RAW / "matches").exists() and (RAW / "aggregates").exists():
        print("✓ repository content already present")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"→ cloning {REPO}")
        r = _run(["git", "clone", "--depth", "1", "-b", BRANCH, CLONE, f"{tmp}/repo"])
        if r.returncode != 0:
            print(r.stderr[-800:], file=sys.stderr)
            raise SystemExit("git clone failed")
        src = Path(tmp) / "repo"
        print("→ pulling Git LFS objects (may be unavailable behind a proxy)")
        _run(["git", "lfs", "pull"], cwd=str(src))
        if RAW.exists():
            shutil.rmtree(RAW)
        shutil.copytree(src / "data", RAW)
        shutil.copytree(src / "docs", RAW / "docs")
        for f in ("README.md", "LICENSE"):
            if (src / f).exists():
                shutil.copy2(src / f, RAW / f"SKILLCORNER_{f}")
    print(f"  content in {RAW}")


def missing() -> list[tuple[str, Path]]:
    """Games whose tracking file is absent, a pointer, or truncated."""
    out = []
    for d in sorted((RAW / "matches").iterdir()):
        if not d.is_dir():
            continue
        p = d / f"{d.name}_tracking_data.jsonl.gz"
        if not is_complete(p):
            out.append((d.name, p))
    return out


def fetch_one(item: tuple[str, Path]) -> tuple[str, int, bool]:
    """Download to a temp file, verify, then move into place. Retries on failure."""
    gid, dest = item
    url = f"{MEDIA}/data/matches/{gid}/{gid}_tracking_data.jsonl.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=900) as r, tmp.open("wb") as f:
                expected = r.headers.get("Content-Length")
                n = 0
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            if expected and n != int(expected):
                raise OSError(f"short read: {n} of {expected} bytes")
            tmp.replace(dest)
            if is_complete(dest):
                return gid, dest.stat().st_size, True
            raise OSError("downloaded file failed the gzip integrity check")
        except Exception as exc:          # noqa: BLE001 - retry on anything transient
            tmp.unlink(missing_ok=True)
            if attempt == MAX_ATTEMPTS:
                print(f"  {gid}  FAILED after {attempt} attempts: {exc}", file=sys.stderr)
                return gid, 0, False
            time.sleep(2 * attempt)
    return gid, 0, False


def main() -> int:
    clone_repo()
    todo = missing()
    if not todo:
        print("✓ all tracking files present and verified")
    else:
        print(f"→ {len(todo)} tracking files to fetch (~{len(todo) * 35} MB) "
              f"from media.githubusercontent.com")
        # 3 workers rather than 5: higher concurrency truncates on this endpoint
        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            for gid, n, ok in ex.map(fetch_one, todo):
                print(f"  {gid}  {n / 1e6:6.1f} MB  {'ok' if ok else 'FAILED'}")
        still = [g for g, _ in missing()]
        if still:
            print(f"✗ incomplete after retries: {still}", file=sys.stderr)
            return 1

    n_games = sum(1 for d in (RAW / "matches").iterdir() if d.is_dir())
    total = sum(f.stat().st_size for f in RAW.rglob("*") if f.is_file())
    print(f"✓ {n_games} games, {total / 1e6:.0f} MB in {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
