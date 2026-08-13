#!/usr/bin/env python3
"""Generate the signal-level synthetic grain-clustering benchmark corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from quantem.diffraction.grain_clustering_synthetic import generate_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--num-random", type=int, default=1000)
    parser.add_argument("--shape", type=int, nargs=2, metavar=("RX", "RY"), default=(64, 64))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quicklooks", choices=("none", "canonical", "all"), default="canonical"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse existing samples, generating only deterministic missing files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_random < 0:
        raise SystemExit("--num-random must be non-negative")
    if any(value <= 0 for value in args.shape):
        raise SystemExit("--shape values must be positive")
    manifest = generate_dataset(
        args.output,
        num_random=args.num_random,
        map_shape=tuple(args.shape),
        seed=args.seed,
        quicklooks=args.quicklooks,
        resume=args.resume,
    )
    print("output:", args.output.resolve())
    print("splits:", manifest["splits"])
    print("samples:", len(manifest["samples"]))
    print("last run:", manifest["last_run"])


if __name__ == "__main__":
    main()
