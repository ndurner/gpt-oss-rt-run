#!/usr/bin/env python3
"""
Update a Kaggle submission JSON by replacing its harmony_response_walkthroughs
with the output_text from all phase_3 runs of a given experiment.

Usage:
  python scripts/update_harmony_walkthroughs.py \
    --target kaggle_submission/example-harmony-findings.json \
    --experiment zipb-barebones

This will read files from:
  exp-combined/experiment_<experiment>/phase_3/*.json
and set target['harmony_response_walkthroughs'] to a list of the
`output_text` values found in numeric run order.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


RUN_FILE_RE = re.compile(r"^run_(\d+)\.json$")


def find_phase3_run_files(exp_name: str, base_dir: Path) -> List[Tuple[int, Path]]:
    """Return list of (run_index, file_path) for phase_3 run files, sorted by index."""
    phase3_dir = base_dir / f"experiment_{exp_name}" / "phase_3"
    if not phase3_dir.is_dir():
        raise FileNotFoundError(f"Phase 3 directory not found: {phase3_dir}")

    runs: List[Tuple[int, Path]] = []
    for p in phase3_dir.iterdir():
        if not p.is_file():
            continue
        m = RUN_FILE_RE.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        runs.append((idx, p))

    runs.sort(key=lambda x: x[0])
    if not runs:
        raise FileNotFoundError(f"No run_*.json files found in {phase3_dir}")
    return runs


def extract_output_texts(run_files: List[Tuple[int, Path]]) -> List[str]:
    """Load each run JSON and collect its `output_text` string if present."""
    outputs: List[str] = []
    for idx, path in run_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read {path}: {e}") from e

        if "output_text" not in data:
            raise KeyError(f"Missing 'output_text' in {path}")
        text = data["output_text"]
        if not isinstance(text, str):
            raise TypeError(f"Field 'output_text' in {path} is not a string")
        outputs.append(text)

    return outputs


def update_target_json(target_path: Path, walkthroughs: List[str]) -> None:
    """Replace harmony_response_walkthroughs in target JSON and write back in place."""
    if not target_path.is_file():
        raise FileNotFoundError(f"Target JSON not found: {target_path}")

    try:
        with target_path.open("r", encoding="utf-8") as f:
            target = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read target JSON {target_path}: {e}") from e

    if "harmony_response_walkthroughs" not in target:
        raise KeyError(
            f"Target JSON missing 'harmony_response_walkthroughs': {target_path}"
        )

    target["harmony_response_walkthroughs"] = walkthroughs

    # Preserve a stable, readable formatting.
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(target, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the target Kaggle-style JSON to update",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name (without 'experiment_' prefix)",
    )
    parser.add_argument(
        "--exp-base",
        default="exp-combined",
        help="Base directory containing experiment_* folders (default: exp-combined)",
    )

    args = parser.parse_args()

    base_dir = Path(args.exp_base)
    run_files = find_phase3_run_files(args.experiment, base_dir)
    walkthroughs = extract_output_texts(run_files)
    update_target_json(Path(args.target), walkthroughs)

    print(
        f"Updated {args.target} with {len(walkthroughs)} walkthrough(s) "
        f"from experiment_{args.experiment}/phase_3"
    )


if __name__ == "__main__":
    main()

