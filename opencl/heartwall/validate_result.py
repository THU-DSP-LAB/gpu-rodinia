#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

INT_PATTERN = re.compile(r"-?\d+")


def load_values(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    return [int(token) for token in INT_PATTERN.findall(text)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    output_path = Path(args.output)
    if not output_path.is_file():
        print(f"Missing output file: {output_path}", file=sys.stderr)
        return 1

    reference = load_values(reference_path)
    output = load_values(output_path)
    if len(reference) != len(output):
        print(
            f"Value count mismatch: ref={len(reference)} output={len(output)}",
            file=sys.stderr,
        )
        return 1

    for index, (ref_value, out_value) in enumerate(zip(reference, output)):
        if ref_value != out_value:
            print(
                f"Mismatch at value {index}: ref={ref_value} output={out_value}",
                file=sys.stderr,
            )
            return 1

    print("Validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
