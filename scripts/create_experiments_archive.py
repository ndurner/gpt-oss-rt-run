#!/usr/bin/env python3
"""Build experiments-raw tarball with desired layout."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

DATASET_ROOT = Path("datasets/raw")
EXP_COMBINED_DIR = Path("exp-combined")
TOURNAMENT_DIR = Path("tournament/exp-tournament/tournament_output-sister")
LINGUISTIC_FILES = [
    Path("linguistic/experiments.json"),
    Path("linguistic/findings_1-4_run_eval.py"),
]
TOURNAMENT_NEST = DATASET_ROOT / "tournament_output"
SKIP_NAMES = {".DS_Store"}


def require_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected path: {path}")


def add_exp_combined(tar: tarfile.TarFile) -> None:
    for child in sorted(EXP_COMBINED_DIR.iterdir()):
        if child.name in SKIP_NAMES:
            continue
        arcname = DATASET_ROOT / child.name
        tar.add(child, arcname=str(arcname))


def add_tournament(tar: tarfile.TarFile) -> None:
    for child in sorted(TOURNAMENT_DIR.iterdir()):
        if child.name in SKIP_NAMES:
            continue
        if child.name == "tournament_output":
            tar.add(child, arcname=str(TOURNAMENT_NEST))
        else:
            arcname = TOURNAMENT_NEST / child.name
            tar.add(child, arcname=str(arcname))


def add_linguistic(tar: tarfile.TarFile) -> None:
    for path in LINGUISTIC_FILES:
        tar.add(path, arcname=str(DATASET_ROOT / path.name))


def build_archive(archive: Path) -> None:
    require_exists(EXP_COMBINED_DIR)
    require_exists(TOURNAMENT_DIR)
    for path in LINGUISTIC_FILES:
        require_exists(path)

    if archive.exists():
        archive.unlink()

    with tarfile.open(archive, mode="w:xz") as tar:
        add_exp_combined(tar)
        add_tournament(tar)
        add_linguistic(tar)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to output tar.xz archive")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_archive(args.archive)
