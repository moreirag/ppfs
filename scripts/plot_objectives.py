#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot solutions in objective space from .npz files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more .npz result files containing array F",
    )
    parser.add_argument("--title", default="Objective Space")
    parser.add_argument("--xlabel", default="f1")
    parser.add_argument("--ylabel", default="f2")
    parser.add_argument("--zlabel", default="f3")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--size", type=float, default=18)
    parser.add_argument("--output", type=Path, default=None, help="Save plot image")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    return parser.parse_args()


def load_front(path: Path):
    data = np.load(path)
    if "F" not in data:
        raise ValueError(f"File {path} does not contain 'F'")
    F = np.asarray(data["F"], dtype=float)
    if F.ndim != 2:
        raise ValueError(f"Array F in {path} must be 2D, got shape {F.shape}")
    return F


def main():
    args = parse_args()

    fronts = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        fronts.append((path.stem, load_front(path)))

    n_obj = fronts[0][1].shape[1]
    if n_obj not in (2, 3):
        raise ValueError(f"Only 2D or 3D objective spaces are supported. Found M={n_obj}")

    for name, F in fronts[1:]:
        if F.shape[1] != n_obj:
            raise ValueError(
                f"All inputs must have same number of objectives. {name} has M={F.shape[1]}, expected M={n_obj}"
            )

    fig = plt.figure(figsize=(8, 6))

    if n_obj == 2:
        ax = fig.add_subplot(111)
        for name, F in fronts:
            ax.scatter(F[:, 0], F[:, 1], s=args.size, alpha=args.alpha, label=name)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
    else:
        ax = fig.add_subplot(111, projection="3d")
        for name, F in fronts:
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=args.size, alpha=args.alpha, label=name)
        ax.set_xlabel(args.xlabel)
        ax.set_ylabel(args.ylabel)
        ax.set_zlabel(args.zlabel)

    ax.set_title(args.title)
    ax.grid(True, linestyle="--", alpha=0.3)
    if len(fronts) > 1:
        ax.legend(frameon=True)

    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot to: {args.output}")

    if args.show or args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
