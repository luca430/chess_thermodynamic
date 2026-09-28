#!/usr/bin/env python3
"""Benchmark entropy-adaptive expected values on a representative position."""

from __future__ import annotations

import argparse
import time

import chess

from thermo_chess.evaluation import StaticEvaluator
from thermo_chess.measure import Style, move_distribution
from thermo_chess.search import AdaptiveExpectedValue


REPRESENTATIVE_FEN = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 8"


class CountingEvaluator(StaticEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, board: chess.Board) -> float:
        self.calls += 1
        return super().evaluate(board)


def full_expected_value(
    board: chess.Board,
    style: Style,
    beta: float,
    evaluator: CountingEvaluator,
    depth: int,
) -> tuple[float, int]:
    if depth == 0 or board.is_game_over(claim_draw=True):
        return evaluator.evaluate(board), 1
    landscape = move_distribution(board, style, beta, evaluator)
    nodes = 1
    value = 0.0
    for record in landscape.records:
        child = board.copy(stack=False)
        child.push(record.move)
        child_value, child_nodes = full_expected_value(
            child, style, beta, evaluator, depth - 1
        )
        value += record.probability * child_value
        nodes += child_nodes
    return value, nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fen", default=REPRESENTATIVE_FEN)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--adaptive-c", type=float, default=0.3)
    args = parser.parse_args()

    board = chess.Board(args.fen)
    style = Style()
    print("depth  seconds    nodes   static   recursive   avg_K   cache_hits   value")
    for depth in range(1, 5):
        search = AdaptiveExpectedValue(
            style,
            args.beta,
            StaticEvaluator(),
            depth=depth,
            adaptive_c=args.adaptive_c,
        )
        search.begin_diagnostics()
        value = search.expected_value(board)
        diagnostics = search.finish_diagnostics()
        print(
            f"{depth:>5}  {diagnostics['elapsed_time']:>7.3f}  "
            f"{diagnostics['nodes_evaluated']:>7}  "
            f"{diagnostics['static_evaluations']:>7}  "
            f"{diagnostics['recursive_nodes']:>9}  "
            f"{diagnostics['average_k']:>6.2f}  "
            f"{diagnostics['cache_hits']:>10}  {value:>8.3f}"
        )

    evaluator = CountingEvaluator()
    started = time.perf_counter()
    value, nodes = full_expected_value(board, style, args.beta, evaluator, depth=2)
    elapsed = time.perf_counter() - started
    print(
        f"full2  {elapsed:>7.3f}  {nodes:>7}  {evaluator.calls:>7}  "
        f"{'-':>9}  {'-':>6}  {'-':>10}  {value:>8.3f}"
    )


if __name__ == "__main__":
    main()
