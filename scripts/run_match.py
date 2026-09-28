#!/usr/bin/env python3
"""Run a toy thermodynamic chess match."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermo_chess.simulation import STRATEGY_NAMES, MatchConfig, simulate_match


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--beta-white", type=float, default=4.0)
    parser.add_argument("--beta-black", type=float, default=4.0)
    parser.add_argument(
        "--white-strategy",
        choices=STRATEGY_NAMES,
        default="material_conservative",
        help="Style preset used by White (default: material_conservative).",
    )
    parser.add_argument(
        "--black-strategy",
        choices=STRATEGY_NAMES,
        default="activity_aggressive",
        help="Style preset used by Black (default: activity_aggressive).",
    )
    parser.add_argument(
        "--solidness-white",
        type=unit_float,
        default=None,
        help="Override White's preset solidness sigma in [0, 1].",
    )
    parser.add_argument(
        "--solidness-black",
        type=unit_float,
        default=None,
        help="Override Black's preset solidness sigma in [0, 1].",
    )
    parser.add_argument("--depth", type=nonnegative_int, default=4)
    parser.add_argument("--adaptive-c", type=positive_float, default=0.3)
    parser.add_argument("--viewer-workers", type=positive_int, default=2, help="Worker processes for White/Black viewer panels; use 1 for sequential generation.")
    parser.add_argument("--name", default="thermo_match")
    parser.add_argument("--ignore-threefold", action="store_true", help="Continue through threefold repetition until mate/stalemate or max-plies.")
    parser.add_argument("--allow-draw-claims", action="store_true", help="Stop on other claimable/automatic draw rules as well.")
    args = parser.parse_args()

    result = simulate_match(
        config=MatchConfig(
            max_plies=args.max_plies,
            seed=args.seed,
            beta_white=args.beta_white,
            beta_black=args.beta_black,
            white_strategy=args.white_strategy,
            black_strategy=args.black_strategy,
            white_solidness=args.solidness_white,
            black_solidness=args.solidness_black,
            depth=args.depth,
            adaptive_c=args.adaptive_c,
            viewer_workers=args.viewer_workers,
            match_name=args.name,
            stop_only_on_mate_or_stalemate=not args.allow_draw_claims,
            stop_on_threefold_repetition=not args.ignore_threefold,
            results_dir=Path("data/results"),
            games_dir=Path("data/games"),
        )
    )
    print(f"CSV:  {result['csv']}")
    print(f"JSON: {result['json']}")
    print(f"PGN:  {result['pgn']}")


if __name__ == "__main__":
    main()
