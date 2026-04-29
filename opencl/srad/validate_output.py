#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def load_pgm(path: Path) -> tuple[tuple[int, int, int], list[int]]:
    tokens = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(line.split())

    if len(tokens) < 4 or tokens[0] != "P2":
        raise ValueError(f"Unsupported PGM file: {path}")

    width = int(tokens[1])
    height = int(tokens[2])
    max_value = int(tokens[3])
    pixels = [int(token) for token in tokens[4:]]
    return (width, height, max_value), pixels


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

    reference_header, reference_pixels = load_pgm(reference_path)
    output_header, output_pixels = load_pgm(output_path)
    if reference_header != output_header:
        print(
            f"{FAIL} Header mismatch: ref={reference_header} output={output_header}",
            file=sys.stderr,
        )
        return 1

    if len(reference_pixels) != len(output_pixels):
        print(
            f"{FAIL} Pixel count mismatch: ref={len(reference_pixels)} output={len(output_pixels)}",
            file=sys.stderr,
        )
        return 1

    for index, (ref_value, out_value) in enumerate(zip(reference_pixels, output_pixels)):
        if ref_value != out_value:
            print(
                f"{FAIL} Pixel mismatch at {index}: ref={ref_value} output={out_value}",
                file=sys.stderr,
            )
            return 1

    print(f"Validation {PASS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
