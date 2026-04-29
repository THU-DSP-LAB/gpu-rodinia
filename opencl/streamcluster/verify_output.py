#!/usr/bin/env python3
"""
Compare streamcluster output against a fixed reference with explicit failure.
Usage:
  python3 verify_output.py --output output.txt --reference nvidia_ref.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

REL_TOL = 1e-5
ABS_TOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    return parser.parse_args()


def read_tokens(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    return path.read_text().split()


def compare_tokens(output_tokens: list[str], reference_tokens: list[str]) -> None:
    if len(output_tokens) != len(reference_tokens):
        raise ValueError(
            f"token count mismatch, got {len(output_tokens)}, expected {len(reference_tokens)}"
        )
    for index, (actual, expected) in enumerate(zip(output_tokens, reference_tokens)):
        if "." not in expected and "e" not in expected.lower():
            if actual != expected:
                raise ValueError(
                    f"integer token mismatch at index {index}, got {actual}, expected {expected}"
                )
            continue
        actual_value = float(actual)
        expected_value = float(expected)
        if math.isclose(actual_value, expected_value, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            continue
        raise ValueError(
            f"float token mismatch at index {index}, got {actual_value}, expected {expected_value}"
        )


def main() -> int:
    args = parse_args()
    compare_tokens(read_tokens(Path(args.output)), read_tokens(Path(args.reference)))
    print("Verification: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
