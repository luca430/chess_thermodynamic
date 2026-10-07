#!/usr/bin/env python3
"""Run a toy thermodynamic chess match."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermo_chess.simulation import STRATEGY_NAMES, MatchConfig, simulate_match


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def optional_workers(value: str) -> int | None:
    if value == "auto":
        return None
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1 or auto")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def probability_quantile(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must satisfy 0 < q <= 1")
    return parsed


def format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {remaining:.1f}s"


def display_termination(reason: object) -> str:
    text = str(reason)
    if text == "max_plies":
        return "max plies reached"
    return text.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--beta-white", type=float, default=4.0)
    parser.add_argument("--beta-black", type=float, default=4.0)
    parser.add_argument("--kappa", type=positive_float, default=1.0, help="Boltzmann-like constant in pawn units (default: 1.0).")
    parser.add_argument(
        "--white-strategy",
        choices=STRATEGY_NAMES,
        default="material_conservative",
        help="Style preset used by White (default: material_conservative).",
    )
    parser.add_argument(
        "--black-strategy",
        choices=STRATEGY_NAMES,
        default="pressure_aggressive",
        help="Style preset used by Black (default: pressure_aggressive).",
    )
    parser.add_argument("--cdepth", type=positive_int, default=2, help="Complete move-response cycles for search (default: 2).")
    parser.add_argument("--deepening-quantile", type=probability_quantile, default=0.8)
    parser.add_argument("--viewer-workers", type=positive_int, default=1, help="Deprecated for normal saved-data viewer generation; retained for compatibility.")
    parser.add_argument("--search-workers", type=optional_workers, default=1, help="Process workers for root own-move branches; use 1 for serial or auto for CPU count.")
    parser.add_argument("--parallel-min-branches", type=positive_int, default=8, help="Minimum root branch count before search worker parallelism activates.")
    parser.add_argument("--name", default=None, help="Base filename for outputs; generated from strategies, betas, and cycle depth by default.")
    parser.add_argument("--ignore-threefold", action="store_true", help="Continue through threefold repetition until mate/stalemate or max-plies.")
    parser.add_argument("--allow-draw-claims", action="store_true", help="Stop on other claimable/automatic draw rules as well.")
    args = parser.parse_args()
    started_at = time.perf_counter()
    result = simulate_match(
        config=MatchConfig(
            max_plies=args.max_plies,
            seed=args.seed,
            beta_white=args.beta_white,
            beta_black=args.beta_black,
            kappa=args.kappa,
            white_strategy=args.white_strategy,
            black_strategy=args.black_strategy,
            cdepth=args.cdepth,
            deepening_quantile=args.deepening_quantile,
            viewer_workers=args.viewer_workers,
            search_workers=args.search_workers,
            parallel_min_branches=args.parallel_min_branches,
            match_name=args.name,
            stop_only_on_mate_or_stalemate=not args.allow_draw_claims,
            stop_on_threefold_repetition=not args.ignore_threefold,
            games_dir=Path("data/games"),
        ),
        progress=lambda line: print(line, flush=True),
    )
    elapsed = time.perf_counter() - started_at
    plies_played = int(result["plies_played"])
    print()
    print("Match finished")
    print(f"Result: {result['result']}")
    print(f"Termination: {display_termination(result['terminal_reason'])}")
    print(f"Plies played: {plies_played}")
    print(f"Full moves: {result['full_moves_played']}")
    print(f"Elapsed time: {format_duration(elapsed)}")
    if plies_played:
        print(f"Average time per ply: {elapsed / plies_played:.2f}s")
    print(f"White strategy: {args.white_strategy}")
    print(f"Black strategy: {args.black_strategy}")
    print(f"cdepth: {args.cdepth}")
    print(f"Deepening quantile: {args.deepening_quantile}")
    print()
    print(f"CSV:  {result['csv']}")
    print(f"JSON: {result['json']}")
    print(f"PGN:  {result['pgn']}")


if __name__ == "__main__":
    main()
