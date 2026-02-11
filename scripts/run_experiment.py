#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from pymoo.optimize import minimize
from pymoo.problems import get_problem

from mdmc_mop_py import ROIDWUMOEA, ROINSGA2


def build_algorithm(name, pop_size, axis, theta, seed):
    if name == "roinsga2":
        return ROINSGA2(pop_size=pop_size, axis=axis, theta=theta)
    if name == "roidwu":
        return ROIDWUMOEA(pop_size=pop_size, axis=axis, theta=theta)
    raise ValueError(f"Unknown algorithm: {name}")


def build_problem(name, n_obj, n_var):
    if name.startswith("dtlz"):
        return get_problem(name, n_obj=n_obj, n_var=n_var)
    if name.startswith("wfg"):
        return get_problem(name, n_obj=n_obj, n_var=n_var)
    raise ValueError("Use a pymoo problem name, e.g., dtlz2, wfg4, wfg9")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MDMC-MOP Python versions (pymoo)."
    )
    parser.add_argument("--algorithm", choices=["roinsga2", "roidwu"], default="roidwu")
    parser.add_argument("--problem", default="wfg9")
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--n-obj", type=int, default=2)
    parser.add_argument("--n-var", type=int, default=5)
    parser.add_argument("--n-evals", type=int, default=100000)
    parser.add_argument("--theta", type=float, default=0.3)
    parser.add_argument(
        "--axis",
        type=float,
        nargs="+",
        default=None,
        help="ROI axis vector in objective space (default: ones).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.axis is not None and len(args.axis) != args.n_obj:
        raise ValueError("--axis length must match --n-obj")

    axis = np.array(args.axis, dtype=float) if args.axis is not None else None

    problem = build_problem(args.problem, args.n_obj, args.n_var)
    algorithm = build_algorithm(args.algorithm, args.pop_size, axis, args.theta, args.seed)

    result = minimize(
        problem,
        algorithm,
        termination=("n_eval", args.n_evals),
        seed=args.seed,
        save_history=False,
        verbose=args.verbose,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / f"{args.algorithm}_{args.problem}_M{args.n_obj}_D{args.n_var}_seed{args.seed}.npz"

    X = result.pop.get("X")
    F = result.pop.get("F")
    np.savez(out_file, X=X, F=F)

    print(f"Saved results to: {out_file}")
    print(f"Population size: {len(result.pop)}")


if __name__ == "__main__":
    main()
