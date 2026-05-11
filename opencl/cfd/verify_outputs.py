#!/usr/bin/env python3
"""
Compare CFD output files against a fixed reference with explicit failure.
Usage:
  python3 verify_outputs.py --ref-prefix nvidia_ref_iter64
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

OUTPUT_NAMES = ("density.txt", "momentum.txt", "density_energy.txt")
REL_TOL = 1e-4
ABS_TOL = 1e-5
PASS = "\033[92mPASS\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-prefix", required=True)
    return parser.parse_args()


def read_tokens(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return path.read_text().split()


def compare_tokens(output_path: Path, reference_path: Path) -> None:
    output_tokens = read_tokens(output_path)
    reference_tokens = read_tokens(reference_path)
    if len(output_tokens) != len(reference_tokens):
        raise ValueError(
            f"{output_path.name}: token count mismatch, got {len(output_tokens)}, "
            f"expected {len(reference_tokens)}"
        )
    for index, (actual, expected) in enumerate(zip(output_tokens, reference_tokens)):
        if index < 2:
            if actual != expected:
                raise ValueError(
                    f"{output_path.name}: header mismatch at token {index}, "
                    f"got {actual}, expected {expected}"
                )
            continue
        actual_value = float(actual)
        expected_value = float(expected)
        if math.isclose(actual_value, expected_value, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            continue
        raise ValueError(
            f"{output_path.name}: value mismatch at token {index}, "
            f"got {actual_value}, expected {expected_value}"
        )


def main() -> int:
    args = parse_args()
    ref_prefix = Path(args.ref_prefix)
    for output_name in OUTPUT_NAMES:
        compare_tokens(Path(output_name), ref_prefix.with_name(f"{ref_prefix.name}_{output_name}"))
    print(f"Verification: {PASS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
