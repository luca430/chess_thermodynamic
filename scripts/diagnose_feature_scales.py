#!/usr/bin/env python3
"""Report fixed move-feature scale diagnostics on sampled legal positions."""

from __future__ import annotations

import argparse
import random
from statistics import median
from typing import Iterable

import chess

from thermo_chess.features import board_context, move_feature_breakdown
from thermo_chess.measure import STRATEGY_FEATURES


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def sample_positions(count: int, seed: int, max_random_plies: int) -> list[chess.Board]:
    rng = random.Random(seed)
    positions: list[chess.Board] = []
    board = chess.Board()
    while len(positions) < count:
        if board.is_game_over(claim_draw=False) or board.ply() >= max_random_plies:
            board = chess.Board()
            continue
        moves = list(board.legal_moves)
        if not moves:
            board = chess.Board()
            continue
        board.push(rng.choice(moves))
        if board.legal_moves.count() > 0:
            positions.append(board.copy(stack=False))
    return positions


def feature_values(board: chess.Board) -> dict[str, list[float]]:
    before = board_context(board)
    values = {name: [] for name in STRATEGY_FEATURES}
    for move in board.legal_moves:
        after = board.copy(stack=False)
        after.push(move)
        breakdown = move_feature_breakdown(
            board,
            move,
            before_context=before,
            after_context=board_context(after),
            after_board=after,
        )
        pawn = breakdown["pawn"]
        for name in STRATEGY_FEATURES:
            values[name].append(float(pawn[name]))
    return values


def nonzero_abs(values: Iterable[float]) -> list[float]:
    return [abs(value) for value in values if abs(value) > 1e-12]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--max-random-plies", type=int, default=80)
    args = parser.parse_args()

    samples = sample_positions(args.positions, args.seed, args.max_random_plies)
    nonzero = {name: [] for name in STRATEGY_FEATURES}
    ranges = {name: [] for name in STRATEGY_FEATURES}
    totals = {name: 0 for name in STRATEGY_FEATURES}

    for board in samples:
        by_feature = feature_values(board)
        for name, values in by_feature.items():
            totals[name] += len(values)
            nonzero[name].extend(nonzero_abs(values))
            if values:
                ranges[name].append(max(values) - min(values))

    print(f"sampled_positions={len(samples)} seed={args.seed}")
    print(
        "| feature | nonzero % | median | p75 | p90 | max | median range | p90 range |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in STRATEGY_FEATURES:
        nz = nonzero[name]
        feature_ranges = ranges[name]
        nonzero_pct = 100.0 * len(nz) / totals[name] if totals[name] else 0.0
        print(
            f"| {name} | {nonzero_pct:.1f} | {median(nz) if nz else 0.0:.3f} "
            f"| {percentile(nz, 0.75):.3f} | {percentile(nz, 0.90):.3f} "
            f"| {max(nz) if nz else 0.0:.3f} | {median(feature_ranges) if feature_ranges else 0.0:.3f} "
            f"| {percentile(feature_ranges, 0.90):.3f} |"
        )


if __name__ == "__main__":
    main()
