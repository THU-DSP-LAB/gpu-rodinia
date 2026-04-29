#!/usr/bin/env python3

import argparse
import math
import re
import sys
from pathlib import Path

VALUE_PATTERN = re.compile(r"=\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.IGNORECASE)
REL_TOL = 1e-4
ABS_TOL = 1e-6
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def load_values(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8")
    return [float(token) for token in VALUE_PATTERN.findall(text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    output_path = Path(args.output)
    if not output_path.is_file():
        print(f"{FAIL} Missing output file: {output_path}", file=sys.stderr)
        return 1

    reference = load_values(reference_path)
    output = load_values(output_path)
    if len(reference) != len(output):
        print(
            f"{FAIL} Value count mismatch: ref={len(reference)} output={len(output)}",
            file=sys.stderr,
        )
        return 1

    for index, (ref_value, out_value) in enumerate(zip(reference, output)):
        if not math.isclose(ref_value, out_value, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            print(
                f"{FAIL} Mismatch at value {index}: ref={ref_value:.10e} output={out_value:.10e}",
                file=sys.stderr,
            )
            return 1

    print(f"Validation {PASS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
