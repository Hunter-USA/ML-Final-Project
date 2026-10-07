"""Download the Harmonix Set annotations (GitHub) and mel-spectrograms (Dropbox, ~1.2 GB).

    python -m sharp.download                       # both
    python -m sharp.download --what annotations
    python -m sharp.download --what melspecs
    python -m sharp.download --mel-archive ~/Downloads/Harmonix_melspecs.tgz   # already downloaded by hand

Result:
    data/raw/harmonixset/dataset/{metadata.csv, segments/*.txt, ...}
    data/raw/melspecs/**/<id>-mel.npy  (+ info.json)

By downloading the spectrograms you agree to the license included in the archive.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from . import config
from .data import check_mel_info, find_mel_files

ANNOTATIONS_REPO = "https://github.com/urinieto/harmonixset.git"
ANNOTATIONS_URL = "https://github.com/urinieto/harmonixset/archive/refs/heads/master.zip"
MELSPECS_URL = "https://www.dropbox.com/s/zxnqlx0hxz0lsyc/Harmonix_melspecs.tgz?dl=1"
N_SONGS = 912


def fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (sharp-song-structure)"})
    print(f"downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_report = 0, 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done >= next_report:
                pct = f" ({done / total:.0%})" if total else ""
                print(f"  {done / 1e6:,.0f} MB{pct}", flush=True)
                next_report += 50 * (1 << 20)
    tmp.replace(dest)
    return dest


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:          # Python without tarfile extraction filters
            tar.extractall(dest)


def get_annotations(raw_dir: Path = config.RAW_DIR) -> Path:
    target = raw_dir / "harmonixset"
    if (target / "dataset" / "metadata.csv").exists():
        print(f"annotations already present in {target}")
        return target
    if target.exists():
        shutil.rmtree(target)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cloned = False
    if shutil.which("git"):
        print(f"cloning {ANNOTATIONS_REPO}")
        cloned = subprocess.run(["git", "clone", "--depth", "1", ANNOTATIONS_REPO, str(target)]).returncode == 0
    if not cloned:
        archive = fetch(ANNOTATIONS_URL, raw_dir / "harmonixset-master.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(raw_dir)
        (raw_dir / "harmonixset-master").rename(target)
        archive.unlink()
    n = len(list((target / "dataset" / "segments").glob("*.txt")))
    print(f"annotations ready: {n} segment files in {target / 'dataset'}")
    return target


def get_melspecs(raw_dir: Path = config.RAW_DIR, archive: Optional[Path] = None,
                 keep_archive: bool = False) -> Path:
    mel_dir = raw_dir / "melspecs"
    if len(find_mel_files(mel_dir)) >= N_SONGS:
        print(f"mel-spectrograms already present in {mel_dir}")
        return mel_dir
    downloaded = archive is None
    if archive is None:
        archive = fetch(MELSPECS_URL, raw_dir / "Harmonix_melspecs.tgz")
    print(f"extracting {archive} (this takes a few minutes)...")
    mel_dir.mkdir(parents=True, exist_ok=True)
    _extract_tar(Path(archive), mel_dir)
    if downloaded and not keep_archive:
        Path(archive).unlink()
    n = len(find_mel_files(mel_dir))
    info = check_mel_info(mel_dir)
    print(f"mel-spectrograms ready: {n} files in {mel_dir}")
    if info:
        print("  info.json: " + ", ".join(f"{k}={v}" for k, v in info.items() if k.isupper()))
    if n < N_SONGS:
        print(f"  warning: expected {N_SONGS} files", file=sys.stderr)
    return mel_dir


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", choices=["annotations", "melspecs", "all"], default="all")
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--mel-archive", type=Path, default=None,
                   help="use a Harmonix_melspecs.tgz you already downloaded instead of fetching it")
    p.add_argument("--keep-archive", action="store_true")
    args = p.parse_args(argv)
    if args.what in ("annotations", "all"):
        get_annotations(args.raw_dir)
    if args.what in ("melspecs", "all") or args.mel_archive:
        get_melspecs(args.raw_dir, args.mel_archive, args.keep_archive)
    print("next: python -m sharp.prepare")


if __name__ == "__main__":
    main()
